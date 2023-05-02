#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --output=clusters.out
#SBATCH --job-name=clustering

#module load miniconda
#source activate avhubert



EXP_PATH="/scratch/work/sarvasm1/av_hubert/ussee/lstm_id_to_cluster/exp"

split="train"
text_data_path="/scratch/work/sarvasm1/av_hubert/ussee/exp/clustering/${split}.wrd_clean"
cluster_data_path="/scratch/work/sarvasm1/av_hubert/ussee/exp/clustering/avhubert_base_lrs3_iter5_output_head/labels/${split}_0_1"
exp_name="test_train"

mkdir -pv ${EXP_PATH}/${exp_name}
#python3 convert_numers.py --input_text_file "/scratch/work/sarvasm1/av_hubert/ussee/exp/clustering/${split}.wrd" \
						#--output_text_file ${text_data_path}
python train.py --text ${text_data_path} \
                --clusters ${cluster_data_path} \
                --exp_dir ${EXP_PATH}/${exp_name}




