#!/bin/bash

#SBATCH --time=1:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --output=decoding.log
#SBATCH --job-name=decoding

module load miniconda
source activate avhubert

BASE_DUMPDIR="dump"
BASE_EXPDIR="exp/openNMT_id2char/kmeans_avhubert_large_iter5"

src_clusters="${BASE_DUMPDIR}/avhubert_large_lrs3_iter5_12_layer_feat_kmeans/test_0_1_rm_dup"
exp_dirs=("lstm_0.001lr_1024embed_1024hidden_sent_norm_nodup_sep" "lstm_0.001lr_500embed_10_early_stop_nodup_sep")
models=("model_step_6000" "model_step_7000")

block_repeat=0
beam_size=15

for i in "${!exp_dirs[@]}"
do
    onmt_translate -model ${BASE_EXPDIR}/${exp_dirs[i]}/checkpoints/${models[i]}.pt \
  				-src ${src_clusters} \
  				-output ${BASE_EXPDIR}/${exp_dirs[i]}/test_0_1_predicted_${models[i]}_block_ngram_repeat_${block_repeat}_beam_size_${beam_size} \
                --block_ngram_repeat ${block_repeat} \
                --beam_size ${beam_size} \
  				--attn_debug \
  				--verbose &> ${BASE_EXPDIR}/${exp_dirs[i]}/attentions_${models[i]}_block_ngram_repeat_${block_repeat}_beam_size_${beam_size}.log

	python3 src/compute_metric.py --pred ${BASE_EXPDIR}/${exp_dirs[i]}/test_0_1_predicted_${models[i]}_block_ngram_repeat_${block_repeat}_beam_size_${beam_size} \
								--target ${BASE_DUMPDIR}/test.wrd > ${BASE_EXPDIR}/${exp_dirs[i]}/test_0_1_predicted_${models[i]}_block_ngram_repeat_${block_repeat}_beam_size_${beam_size}_metric

    mkdir -p ${BASE_EXPDIR}/${exp_dirs[i]}/attentions_${models[i]}_block_ngram_repeat_${block_repeat}_beam_size_${beam_size}
    python3 src/attention_plot.py --attention_log_file ${BASE_EXPDIR}/${exp_dirs[i]}/attentions_${models[i]}_block_ngram_repeat_${block_repeat}_beam_size_${beam_size}.log \
                               --output_folder ${BASE_EXPDIR}/${exp_dirs[i]}/attentions_${models[i]}_block_ngram_repeat_${block_repeat}_beam_size_${beam_size}

done



