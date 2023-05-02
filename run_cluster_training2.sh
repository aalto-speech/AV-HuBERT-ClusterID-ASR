#!/bin/bash

#SBATCH --time=10:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --output=lstm_500_base4AVH.log
#SBATCH --job-name=lstm_500_base4AVH

module load miniconda
source activate avhubert

exp_dir="/scratch/work/sarvasm1/AV-HuBERT-ClusterID-ASR/exp/openNMT_id2char/avhubert_large_iter4/lstm_0.001lr_500embed_10_early_stop_nodup_sep"
conf="conf/fix_train_rm_dup_sep_model.yaml"

cp ${conf} ${exp_dir}/conf.yaml
onmt_build_vocab -config ${conf} -n_sample 30782
onmt_train -config ${conf}

#onmt_translate -model ${exp_dir}/checkpoints/model_step_4000.pt \
#			-src ${dumpdir}/test_0_1_rm_dup \
#				-output ${exp_dir}/test_0_1_predicted_4000checkpoint \
#				--attn_debug \
#				--verbose &> sep_rm_dup_attentions_4000checkpoint.log
#

