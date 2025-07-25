#!/bin/bash

# custom config
DATA=/home/yyc/yyc_workspace/CVPR2024/Data/
TRAINER=KgCoOp
DATASET=$1
SEED=$2
#CFG=rn50_ep100  # config file
CFG=vit_b16_ep10_ctxv1
CTP=end  # class token position (end or middle)
NCTX=4  # number of context tokens
SHOTS=16  # number of shots (1, 2, 4, 8, 16)
CSC=False  # class-specific context (False or True)
WEIGHT=1.0

LOADEP=10
SUB=new


COMMON_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}_${WEIGHT}/${CFG}/seed${SEED}
MODEL_DIR=output/base2new/train_base/${COMMON_DIR}
DIR=output/base2new/test_${SUB}/${COMMON_DIR}

echo "Run this job and save the output to ${DIR}"
python train.py \
--root ${DATA} \
--seed ${SEED} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${TRAINER}/${CFG}.yaml \
--output-dir ${DIR} \
--model-dir ${MODEL_DIR} \
--load-epoch ${LOADEP} \
--eval-only \
TRAINER.COOP.N_CTX ${NCTX} \
TRAINER.COOP.CSC ${CSC} \
TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
DATASET.NUM_SHOTS ${SHOTS} \
DATASET.SUBSAMPLE_CLASSES ${SUB}


