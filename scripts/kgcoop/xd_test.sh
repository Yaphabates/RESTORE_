#!/bin/bash

#cd ../..

# custom config
DATA=/home/yyc/yyc_workspace/CVPR2024/Data/
TRAINER=KgCoOp

DATASET=$1
SEED=$2

CFG=vit_b16_ep2_ctxv1_crossdataset
SHOTS=16


DIR=output/evaluation/${TRAINER}/${CFG}_${SHOTS}shots/${DATASET}/seed${SEED}

echo "Run this job and save the output to ${DIR}"

python train.py \
--root ${DATA} \
--seed ${SEED} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${TRAINER}/${CFG}.yaml \
--output-dir ${DIR} \
--model-dir output/imagenet/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED} \
--load-epoch 2 \
--eval-only
