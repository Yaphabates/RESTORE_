import os.path as osp
from collections import OrderedDict
import math
import json
import copy
import time
import datetime
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.utils import (
    MetricMeter, AverageMeter)
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import numpy as np
from .imagenet_templates import IMAGENET_TEMPLATES

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'MaPLe_FSA',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0,
                      "maple_length": cfg.TRAINER.MAPLE_FSA.N_CTX}
    # import pdb; pdb.set_trace()
    # print(next(model.parameters()).device)
    model = clip.build_model(state_dict or model.state_dict(), design_details)
    # import pdb; pdb.set_trace()
    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # Pass as the list, as nn.sequential cannot process multiple arguments in the forward pass
        # import pdb; pdb.set_trace()
        combined = [x, compound_prompts_deeper_text, 0]  # third argument is the counter which denotes depth of prompt
        outputs = self.transformer(combined)
        # import pdb; pdb.set_trace()
        x = outputs[0]  # extract the x back from here
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)   ### [n_cls, 77, 512]
        # import pdb; pdb.set_trace()
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        ### x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] get a [n_cls, transformer.width] tensor

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        
        return x   ### [n_cls, 512]

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, int(c_in // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(int(c_in // reduction), c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # import pdb; pdb.set_trace()
        x = self.fc(x)
        return x

class Cross_attn_block(nn.Module):
    def __init__(self, dtype=torch.float16, cfg=None):
        super().__init__()
        self.dtype = dtype
        # self.prompt_bank = {}
        prompt_bank = np.load(cfg.DATASET.PROMPT_PATH)
        # for key in prompt_bank.keys():
        #     self.prompt_bank[key] = torch.from_numpy(prompt_bank[key]).type(self.dtype)
        #     self.prompt_bank[key] /= self.prompt_bank[key].norm(dim=-1, keepdim=True)
        #     self.prompt_bank[key] = nn.Parameter(self.prompt_bank[key])

        prompt_bank = np.concatenate([np.mean(prompt_bank[key], axis=0, keepdims=True) for key in prompt_bank.keys()], axis=0)
        # import pdb; pdb.set_trace()
        prompt_bank = torch.from_numpy(prompt_bank).type(self.dtype)
        prompt_bank /= prompt_bank.norm(dim=-1, keepdim=True)
        self.prompt_bank = nn.Parameter(prompt_bank)

        self.cross_attn = nn.MultiheadAttention(embed_dim=512, num_heads=8)
        self.pre_ln = nn.LayerNorm(512)

        self.FFN = nn.Sequential(
            nn.Linear(512, 2048),
            QuickGELU(),
            nn.Linear(2048, 512),
            QuickGELU()
        )
        self.post_ln = nn.LayerNorm(512)

    def forward(self, x):
        # import pdb; pdb.set_trace()
        # classnames = [name.replace('_', ' ') for name in classnames]
        # x = x.type(torch.float32)
        # self.prompt_bank.to(dtype=x.dtype)
        # attn_mask = self.build_attention_mask().to(dtype=x.dtype, device=x.device)
        # import pdb; pdb.set_trace()
        # classnames = np.array(classnames)
        # aug_prompt = torch.cat([self.prompt_bank[key] for key in [classnames[i] for i in label]], dim=0).to(x.device)
        # x = x + self.cross_attn(self.pre_ln(x), self.pre_ln(self.prompt_bank), 
        #                         self.pre_ln(self.prompt_bank), need_weights=False, attn_mask=None)[0]
        x = x + self.FFN(self.post_ln(x))
        return x

class Feature_Calibrator(nn.Module):
    def __init__(self):
        super().__init__()
        self.FFN = nn.Sequential(
            nn.Linear(512, 2048),
            QuickGELU(),
            nn.Linear(2048, 512),
            # QuickGELU()
        )
        self.post_ln = nn.LayerNorm(512)

    def forward(self, x):
        x = x + self.FFN(self.post_ln(x))
        return x

class MultiModalPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.MAPLE_FSA.N_CTX
        ctx_init = cfg.TRAINER.MAPLE_FSA.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        # Default is 1, which is compound shallow prompting
        assert cfg.TRAINER.MAPLE.PROMPT_DEPTH >= 1, "For MaPLe, PROMPT_DEPTH should be >= 1"
        self.compound_prompts_depth = cfg.TRAINER.MAPLE_FSA.PROMPT_DEPTH  # max=12, but will create 11 such shared prompts
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        # import pdb; pdb.set_trace()
        if ctx_init and n_ctx <= 4:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print('MaPLe design: Multi-modal Prompt Learning')
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of MaPLe context words (tokens): {n_ctx}")
        # These below, related to the shallow prompts
        # Linear layer so that the tokens will project to 512 and will be initialized from 768
        # import pdb; pdb.set_trace()
        self.ctx = nn.Parameter(ctx_vectors)
        # import pdb; pdb.set_trace()
        # self.ctx = nn.Parameter(self.ctx)
        self.proj = nn.Linear(ctx_dim, 768)
        # self.proj = nn.Sequential(Adapter(c_in=512, reduction=0.25), nn.Linear(ctx_dim, 768))
        self.proj.half()
        # self.cross_attn = Cross_attn_block(dtype=dtype, cfg=cfg).half()
        # These below parameters related to the shared prompts
        # Define the compound prompts for the deeper layers

        # Minimum can be 1, which defaults to shallow MaPLe
        # compound prompts
        self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, 512))
                                                      for _ in range(self.compound_prompts_depth - 1)])

        # self.compound_prompts_vision = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, 768))
        #                                         for _ in range(self.compound_prompts_depth - 1)])
        # import pdb; pdb.set_trace()
        for single_para in self.compound_prompts_text:
            nn.init.normal_(single_para, std=0.02)
            # nn.init.constant_(single_para, 0)
        
        # for single_para in self.compound_prompts_vision:
        #     nn.init.normal_(single_para, std=0.02)
            # import pdb; pdb.set_trace()
        # Also make corresponding projection layers, for each prompt

        # single_layer = nn.Sequential(Adapter(c_in=512, reduction=0.25), nn.Linear(ctx_dim, 768))
        single_layer = nn.Linear(ctx_dim, 768)
        self.compound_prompt_projections = _get_clones(single_layer, self.compound_prompts_depth - 1)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, 77)

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)  ### [n_cls, 77, 512]

        # import pdb; pdb.set_trace()
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self):
        ctx = self.ctx  ### embedding of "a photo of a"

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)   ### embedding of "[SOS] a photo of a [CLASS] [EOS]"

        # Before returning, need to transform
        # prompts to 768 for the visual side
        visual_deep_prompts = []
        # import pdb; pdb.set_trace()
        # total_cpt = torch.cat([cpt for cpt in self.compound_prompts_text], dim=0)
        # total_cpv = self.single_layer(total_cpt)

        for index, layer in enumerate(self.compound_prompt_projections):
            # import pdb; pdb.set_trace()
            # visual_deep_prompts.append(total_cpv[index*self.n_ctx: (index+1)*self.n_ctx, :])
            visual_deep_prompts.append(layer(self.compound_prompts_text[index]))
            # visual_deep_prompts.append(layer(self.cross_attn(self.compound_prompts_text[index])))
        # Now the other way around
        # We will project the textual prompts from 512 to 768

        ### return [embedding of "[SOS] a photo of a [CLASS] [EOS]", embedding of "a photo of a", random init context, compound_proj(random init context)]
        return prompts, self.proj(self.ctx), self.compound_prompts_text, visual_deep_prompts
        # return prompts, self.proj(self.ctx), self.compound_prompts_text, self.compound_prompts_vision
        # return prompts, self.proj(self.cross_attn(self.ctx)), self.compound_prompts_text, visual_deep_prompts   # pass here original, as for visual 768 is required


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.classnames = classnames
        self.prompt_learner = MultiModalPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts  ###  tokenized "a photo of a"
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.init_visual_prompt_norm = None
        self.init_text_prompt_norm = None
        self.text_adapter = Feature_Calibrator().half()
        self.visual_adapter = Feature_Calibrator().half()
        self.text_cali_ratio = nn.Parameter(torch.tensor(cfg.TRAINER.MAPLE_FSA.TEXT_CALIB_R, dtype=torch.float64))
        self.vis_cali_ratio = nn.Parameter(torch.tensor(cfg.TRAINER.MAPLE_FSA.VIS_CALIB_R, dtype=torch.float64))
        self.fs_loss_r = cfg.TRAINER.MAPLE_FSA.FS_LOSS_R
        self.vis_cali_updated = False
        self.text_cali_updated = False
    
    def get_current_shift(self):
        with torch.no_grad():
            prompts, shared_ctx, deep_compound_prompts_text, \
                deep_compound_prompts_vision = self.prompt_learner()
            text_shift = 0.0
            vis_shift = 0.0
            for i in range(len(deep_compound_prompts_text)):
                text_shift += torch.norm(deep_compound_prompts_text[i].t()@deep_compound_prompts_text[i]) / len(deep_compound_prompts_text)
                vis_shift += torch.norm(deep_compound_prompts_vision[i].t()@deep_compound_prompts_vision[i]) / len(deep_compound_prompts_vision)

        return text_shift, vis_shift

    def update_calib(self):
        # import pdb; pdb.set_trace()
        text_shift, vis_shift = self.get_current_shift()
        if self.prompt_learner.training and not self.text_cali_updated:
            self.text_cali_ratio = nn.Parameter(torch.tensor(self.text_cali_ratio.item() + text_shift.item()*0.02))
            self.text_cali_updated = True
        if self.prompt_learner.training and not self.vis_cali_updated:
            self.vis_cali_ratio = nn.Parameter(torch.tensor(self.vis_cali_ratio.item() + vis_shift.item()*0.005))
            self.vis_cali_updated = True       

    def forward(self, image, label=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        prompts, shared_ctx, deep_compound_prompts_text, \
            deep_compound_prompts_vision = self.prompt_learner()

        ## layer-wise loss

        if self.init_visual_prompt_norm == None:
            self.init_visual_prompt_norm = [torch.norm(vision_prompt.t() @ vision_prompt).detach()
                                    for vision_prompt in deep_compound_prompts_vision]
            self.init_text_prompt_norm = [torch.norm(text_prompt.t() @ text_prompt).detach()
                                    for text_prompt in deep_compound_prompts_text]
        reg_loss = 0.0
        
        for i in range(len(deep_compound_prompts_text)):
            text_shift = torch.norm(deep_compound_prompts_text[i].t()@deep_compound_prompts_text[i])/self.init_text_prompt_norm[i]
            vis_shift = torch.norm(deep_compound_prompts_vision[i].t()@deep_compound_prompts_vision[i])/self.init_visual_prompt_norm[i]
            reg_loss += (text_shift-vis_shift) ** 2
            # text_shift = torch.norm(deep_compound_prompts_text[i])
            # vis_shift = torch.norm(deep_compound_prompts_vision[i])
            # # import pdb; pdb.set_trace()
            # reg_loss += text_shift * 0.00001 + vis_shift * 0.00001

        text_features = self.text_encoder(prompts, tokenized_prompts, deep_compound_prompts_text)
        # text_scale = (scale * text_prompt_reg / self.init_text_prompt_norm).detach() + 0.08
        # self.text_cali_ratio = 0.1
        text_features = self.text_cali_ratio.data.item() * self.text_adapter(text_features) + \
            (1-self.text_cali_ratio.data.item()) * text_features


        image_features = self.image_encoder(image.type(self.dtype), shared_ctx, deep_compound_prompts_vision)

        image_features = self.vis_cali_ratio.data.item() * self.visual_adapter(image_features) + \
            (1-self.vis_cali_ratio.data.item()) * image_features


        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # import pdb; pdb.set_trace()
        logits = logit_scale * image_features @ text_features.t()
        # import pdb; pdb.set_trace()
        # print((vision_promnpt_reg/self.init_visual_prompt_norm - text_prompt_reg/self.init_text_prompt_norm) ** 2)
        # print(self.init_text_prompt_norm)
        # print(vision_promnpt_reg/self.init_visual_prompt_norm, text_prompt_reg/self.init_text_prompt_norm)
        # import pdb; pdb.set_trace()
        if self.prompt_learner.training:
            return F.cross_entropy(logits, label) + self.fs_loss_r * reg_loss
        return logits


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


@TRAINER_REGISTRY.register()
class MaPLe_FSA(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.MAPLE.PREC in ["fp16", "fp32", "amp"]
    def train(self):
        self.before_train()
        for self.epoch in range(self.start_epoch, self.max_epoch):
            print("text_cali", self.model.text_cali_ratio.item(), "vis_cali", self.model.vis_cali_ratio.item())
            self.before_epoch()
            self.run_epoch()
            self.after_epoch()
            if self.max_epoch // 3 == self.epoch:
                self.model.update_calib()
        self.after_train()

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.MAPLE.PREC == "fp32" or cfg.TRAINER.MAPLE.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = ["prompt_learner", "adapter", "VPT"]
        # name_to_update = ["prompt_learner", "VPT"]

        for name, param in self.model.named_parameters():
            for upd_name in name_to_update:
                if upd_name in name:
                    param.requires_grad_(True)
                    break
                else:
                    param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("MultiModalPromptLearner", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.MAPLE.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)
        # import pdb; pdb.set_trace()

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.MAPLE.PREC
        if prec == "amp":
            with autocast():
                loss = model(image, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss = model(image, label)
            optim.zero_grad()
            loss.backward()
            optim.step()

        loss_summary = {"loss": loss.item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
