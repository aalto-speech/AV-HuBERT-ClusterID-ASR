#!/bin/bash
# Helping script for printing WER statistics from experiments, removing the need to go through them manually

#SBATCH --time=1:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --output=decoding.log
#SBATCH --job-name=decoding

module load miniconda
source activate avhubert

BASE_DUMPDIR="/scratch/work/sarvasm1/AV-HuBERT-ClusterID-ASR/dump"
BASE_EXPDIR="/scratch/work/sarvasm1/AV-HuBERT-ClusterID-ASR/exp/openNMT_id2char"

src_clusters="${BASE_DUMPDIR}/avhubert_large_lrs3_iter5_output_head/labels/test_0_1_rm_dup"
HYP=("avhubert_base_iter4/lstm_0.001lr_1024embed_1024hidden_sent_norm_nodup_sep/test_0_1_predicted_model_step_6000_block_ngram_repeat_0_beam_size_15" 
"avhubert_base_iter4/lstm_0.001lr_500embed_10_early_stop_nodup_sep/test_0_1_predicted_model_step_7000_block_ngram_repeat_0_beam_size_15" 
"avhubert_large_iter5/lstm_0.001lr_1024embed_1024hidden_sent_norm_nodup_sep/test_0_1_predicted_model_step_6500_block_ngram_repeat_0_beam_size_15" 
"avhubert_large_iter5/lstm_0.001lr_500embed_10_early_stop_nodup_sep/test_0_1_predicted_model_step_7000_block_ngram_repeat_0_beam_size_15" 
"avhubert_base_iter5/lstm_0.001lr_10early_stop_1024embed_1024hidden_nodup_sep/test_0_1_predicted_model_step_7500_block_ngram_repeat_0" 
"avhubert_base_iter5/lstm_0.001lr_10early_stop_nodup_sep_256batch/test_0_1_predicted_model_step_7500_block_ngram_repeat_0_beam_size_15")

for i in "${!HYP[@]}"
do
	python3 src/asr_stats.py --hyp ${BASE_EXPDIR}/${HYP[i]} \
							 --ref ${BASE_DUMPDIR}/test.wrd > ${BASE_EXPDIR}/${HYP[i]}_jiwer_stats
    echo ${HYP[i]}
    tail -n 8 ${BASE_EXPDIR}/${HYP[i]}_jiwer_stats | head -n 2
    echo "======================================"
done



