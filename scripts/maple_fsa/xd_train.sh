#!/bin/bash

#cd ../..

# custom config
DATA=/home/yyc/yyc_workspace/CVPR2024/Data/
TRAINER=MaPLe_FSA

DATASET=imagenet
SEED=$2
VIS_CALIB_R=$3
TEXT_CALIB_R=$4
FS_LOSS_LR=$5

CFG=vit_b16_c2_ep5_batch4_2ctx_cross_datasets
SHOTS=16


DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}."
else
    echo "Run this job and save the output to ${DIR}"

    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --vis_calib_r ${VIS_CALIB_R} \
    --text_calib_r ${TEXT_CALIB_R} \
    --fs_loss_r ${FS_LOSS_LR} \
    DATASET.NUM_SHOTS ${SHOTS}
fi