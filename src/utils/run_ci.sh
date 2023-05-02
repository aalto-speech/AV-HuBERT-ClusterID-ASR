#!/bin/bash

#module load miniconda
#source activate avhubert

start_stage=1
stop_stage=1

if [ ${start_stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    EXP_DIR="/scratch/work/sarvasm1/AV-HuBERT-ClusterID-ASR/exp/clustering/avhubert_base_lrs3_iter5_output_head/labels"
    for file in $(ls $EXP_DIR | grep gram); do
        echo $file
        python3 coincidence_index.py --file "${EXP_DIR}/${file}" >> "${EXP_DIR}/index_of_coincedence"
    done
fi
if [ ${start_stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    python3 match.py --text /m/teamwork/t40511_asr/c/LRS3-TED/lrs3/30h_data/test.wrd \
                     --clusters /scratch/work/sarvasm1/av_hubert/ussee/exp/clustering/avhubert_base_lrs3_iter5_output_head/labels/test_0_1_rm_dup \
                     --exp_dir /scratch/work/sarvasm1/av_hubert/ussee/exp/clustering/avhubert_base_lrs3_iter5_output_head
fi
