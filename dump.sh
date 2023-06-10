#!/bin/bash

DUMP_PATH="dump"
EXP_PATH="exp/clustering"


exp="avhubert_large_lrs3_iter5_12_layer_feat_kmeans"

start_stage=1
stop_stage=1


if [ ${start_stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
	echo "Stage 0: Dumping text files into ${DUMP_PATH}"
	python src/dump_data.py
fi
if [ ${start_stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
		echo "Stage 1: Dumping cluster IDs into ${DUMP_PATH}"
		mkdir -pv ${DUMP_PATH}/${exp}
		cp ${EXP_PATH}/${exp}/labels/* ${DUMP_PATH}/${exp}

fi
