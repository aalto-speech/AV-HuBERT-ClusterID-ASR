#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --output=finetune_av_v_large_iter5.out
#SBATCH --job-name=finetune_av_v_large_iter5
#SBATCH -n 1


module load miniconda
source activate avhubert

PROJ_PATH="$(pwd)"
HUBERT_PATH="avhubert"
EXP_PATH="${PROJ_PATH}/exp/finetuning"
cd ${HUBERT_PATH}

export HYDRA_FULL_ERROR=1

# data
datadir="${PROJ_PATH}/lrs3_dataset/30h_data"
labeldir="${PROJ_PATH}/lrs3_dataset/30h_data"

# models
tokenizer="${PROJ_PATH}/lrs3_dataset/spm1000/spm_unigram1000.model"
finetuning_model_path="${PROJ_PATH}/pretrained_models/large_lrs3_iter5.pt"

# config and experiments
conf_path="${PROJ_PATH}/${HUBERT_PATH}/conf/finetune"
conf_name="large_lrs3_30h.yaml"
exp="finetune_V_lrs3_large_iter5_1000vocab_pathtest"

start_stage=0
stop_stage=0

# finetune model on 30h lrs3 data
if [ ${start_stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then

	cp ${conf_path}/${conf_name} ${EXP_PATH}/${exp}/config.yaml

	fairseq-hydra-train --config-dir ${conf_path} \
				--config-name ${conf_name} \
				task.data=${datadir} \
				task.label_dir=${labeldir} \
				task.tokenizer_bpe_model=${tokenizer} \
				model.w2v_path=${finetuning_model_path} \
				hydra.run.dir=${EXP_PATH}/${exp} \
				common.user_dir=`pwd`
fi

# run decoding on model from ${exp} directory
if [ ${start_stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then

	# choose model for decoding - depending on experiment directory
	decode_model_path="${EXP_PATH}/${exp}/checkpoints/checkpoint_best.pt"

	# configuration and result dir
	modality=['audio']  # ['video','audio'] or ['video'] or ['video']
	decode_conf_name="s2s_decode.yaml"
	# rename decode dir based on modality so results are not overwritten
	decode_dir=${EXP_PATH}/${exp}/decode_A/s2s
	
	mkdir -pv  ${decode_dir}

	python -B infer_s2s.py --config-dir "${PROJ_PATH}/${HUBERT_PATH}/conf/" \
						--config-name ${decode_conf_name} \
						dataset.gen_subset=test \
						common_eval.path=${decode_model_path} \
						common_eval.results_path=${decode_dir} \
						override.modalities=${modality} \
						+override.data="${PROJ_PATH}/lrs3_dataset/30h_data" \
						+override.label_dir="${PROJ_PATH}/lrs3_dataset/30h_data" \
						common.user_dir=`pwd` > ${decode_dir}/ref_hyp.out
fi
