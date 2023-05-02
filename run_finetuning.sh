#!/bin/bash

#SBATCH --time=16:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --output=finetune_av_base_iter5.out
#SBATCH --job-name=finetune_av_base_iter5
#SBATCH -n 1


module load miniconda

source activate avhubert

HUBERT_PATH="/scratch/work/sarvasm1/av_hubert/avhubert"
EXP_PATH="/scratch/work/sarvasm1/AV-HuBERT-ClusterID-ASR/exp/finetuning"

cd ${HUBERT_PATH}

export HYDRA_FULL_ERROR=1

conf_path="/scratch/work/sarvasm1/av_hubert/avhubert/conf/av-finetune"
conf_name="base_lrs3_30h.yaml"
datadir="/m/teamwork/t40511_asr/c/LRS3-TED/lrs3/30h_data"
labeldir="/m/teamwork/t40511_asr/c/LRS3-TED/lrs3/30h_data"
tokenizer="/m/teamwork/t40511_asr/c/LRS3-TED/lrs3/spm5000/spm_unigram5000.model"
model_path="/scratch/work/sarvasm1/av_hubert/models/base_lrs3_iter5.pt"

#model_path="/scratch/work/sarvasm1/av_hubert/models/finetuned/base_lrs3_30h.pt"
exp="finetune_AV_lrs3_base_iter5_5000vocab"
start_stage=0
stop_stage=1

if [ ${start_stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then

	cp ${conf_path}/${conf_name} ${EXP_PATH}/${exp}/config.yaml

	fairseq-hydra-train --config-dir ${conf_path} \
				--config-name ${conf_name} \
				task.data=${datadir} \
				task.label_dir=${labeldir} \
				task.tokenizer_bpe_model=${tokenizer} \
				model.w2v_path=${model_path} \
				hydra.run.dir=${EXP_PATH}/${exp} \
				common.user_dir=`pwd`
fi

if [ ${start_stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
	decode_conf_name="s2s_decode.yaml"

	decode_dir=${EXP_PATH}/${exp}/decode_audio/s2s
	
	mkdir -pv  ${decode_dir}

	#cp ${conf_path}/${decode_conf_name} ${decode_dir}/config.yaml

	python -B infer_s2s.py --config-dir "/scratch/work/sarvasm1/av_hubert/avhubert/conf/"  \
						--config-name ${decode_conf_name} \
						dataset.gen_subset=test \
						common_eval.path=${model_path} \
						common_eval.results_path=${decode_dir} \
						override.modalities=['audio','video'] \
						+override.data="/m/teamwork/t40511_asr/c/LRS3-TED/lrs3/30h_data" +override.label_dir="/m/teamwork/t40511_asr/c/LRS3-TED/lrs3/30h_data" \
						common.user_dir=`pwd` > ${decode_dir}/ref_hyp.out
fi







