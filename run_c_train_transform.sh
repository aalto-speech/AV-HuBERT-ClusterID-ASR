#!/bin/bash

#SBATCH --time=16:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --output=transformer_train.log
#SBATCH --job-name=transformer

module load miniconda
source activate avhubert

dumpdir="/scratch/work/sarvasm1/av_hubert/ussee/dump/avhubert_base_lrs3_iter5_output_head/"
exp_dir="/scratch/work/sarvasm1/av_hubert/ussee/exp/openNMT_id2char/transformer_0.01lr_4enc_dec_layers_6heads_512hidden_clean_nodup_sep"
conf="conf/transformer.yaml"

cp ${conf} ${exp_dir}/conf.yaml
#onmt_build_vocab -config ${conf} -n_sample 30782

#onmt_train -config ${conf}

onmt_translate -model /scratch/work/groszt1/ASR/ussee/run/model_step_100000.pt -src clustering/avhubert_base_lrs3_iter4_output_head/labels/train_0_1_rm_dup -output clustering/avhubert_base_lrs3_iter4_output_head/labels/train_0_1_rm_dup_predicted --verbose

onmt_translate -model ${exp_dir}/checkpoints/model_step_200000.pt \
				-src ${dumpdir}/test_0_1_rm_dup \
				-output ${exp_dir}/test_0_1_predicted \
				-verbose 


