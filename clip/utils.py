import json
import torch
import clip
from tqdm import tqdm

# 加载CLIP模型和预处理器
device = "cuda" if torch.cuda.is_available() else "cpu"
model, transform = clip.load("ViT-B/16", device=device)

# 读取JSON文件
with open("your_json_file.json", "r") as json_file:
    data = json.load(json_file)

# 创建一个空字典来存储句子描述和编码
sentence_encodings = {}

# 循环遍历每个类别的句子描述
for category, sentences in data.items():
    for sentence in tqdm(sentences, desc=f"Encoding {category}"):
        # 预处理并编码文本
        text_input = transform([sentence]).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_input)

        # 存储编码结果
        sentence_encodings[sentence] = text_features[0].cpu().numpy()

# 现在，sentence_encodings字典包含了每个句子的单独编码
# 您可以根据需要使用这些编码
import pdb; pdb.set_trace()

# data=read_json_file("/home/yyc/yyc_workspace/CVPR2024/CaFo/gpt_file/caltech_prompt.json")
















