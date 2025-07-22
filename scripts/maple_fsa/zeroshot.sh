#!/bin/bash

#cd ../..

# custom config
DATA=/home/yyc/yyc_workspace/CVPR2024/Data/
TRAINER=MaPLe_FSA
DATASET=$1
CFG=vit_b16_c2_ep5_batch4_2ctx  # rn50, rn101, vit_b32 or vit_b16
DIR=output/${DATASET}/${TRAINER}/zero_shots/

python train.py \
--root ${DATA} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${TRAINER}/${CFG}.yaml \
--output-dir output/${TRAINER}/${CFG}/${DATASET} \
--eval-only