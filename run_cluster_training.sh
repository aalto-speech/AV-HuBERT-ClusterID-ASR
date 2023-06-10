#!/bin/bash

#SBATCH --time=8:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --output=lstm_1024_kmeans_large5.log
#SBATCH --job-name=lstm_1024_kmeans_large5

module load miniconda
source activate avhubert

EXP_PATH="exp/openNMT_id2char/"

exp_dir="${EXP_PATH}/kmeans_avhubert_large_iter5/lstm_0.001lr_1024embed_1024hidden_sent_norm_nodup_sep"
conf="conf/lstm_train.yaml"

mkdir -pv ${exp_dir}
cp ${conf} ${exp_dir}/conf.yaml

onmt_build_vocab -config ${conf} -n_sample 30782
onmt_train -config ${conf}

#onmt_translate -model ${exp_dir}/checkpoints/model_step_6500.pt \
#			-src ${dumpdir}/test_0_1_rm_dup \
#				-output ${exp_dir}/test_0_1_predicted_6500checkpoint \
#				--attn_debug \
#				--verbose &> 1024_lstm_6500checkpoint.log
#

