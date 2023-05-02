#!/bin/bash

#SBATCH --time=10:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --output=lstm_1024_base4AVH.log
#SBATCH --job-name=lstm_1024_base4AVH

module load miniconda
source activate avhubert

#dumpdir="/scratch/work/sarvasm1/AV-HuBERT-ClusterID-ASR/dump/avhubert_large_lrs3_iter5_output_head/"
exp_dir="/scratch/work/sarvasm1/AV-HuBERT-ClusterID-ASR/exp/openNMT_id2char/avhubert_base_iter4/lstm_0.001lr_1024embed_1024hidden_sent_norm_nodup_sep"
conf="conf/lstm_train.yaml"

cp ${conf} ${exp_dir}/conf.yaml
onmt_build_vocab -config ${conf} -n_sample 30782
onmt_train -config ${conf}

#onmt_translate -model ${exp_dir}/checkpoints/model_step_6500.pt \
#			-src ${dumpdir}/test_0_1_rm_dup \
#				-output ${exp_dir}/test_0_1_predicted_6500checkpoint \
#				--attn_debug \
#				--verbose &> 1024_lstm_6500checkpoint.log
#

