#!/bin/bash
# Author: Marek Sarvas

#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --output=clusters.out
#SBATCH --job-name=clustering

module load miniconda
source activate avhubert
module load openblas/0.3.7-gcc-6.5.0-openmp 

RUN_PATH="avhubert/clustering"
PROJ_DIR=$(pwd)
TOOLS="${PROJ_DIR}/tools"
EXP_PATH="$(pwd)/exp"

tsv_dir="${PROJ_DIR}/lrs3_dataset/30h_data"
model="large_lrs3_iter5" 											# type of a model
ckpt_path="${PROJ_DIR}/pretrained_models/${model}.pt"  	# path to a model checkpoint
# dataset split to exctract features from
split="valid"
nshard=1

rank=0
# features to cluster on
is_mfcc=false
# "k_means" for doing k_means on top of the avhubert features, "cluster_ids" to extract clusters from output layer
extract="k_means" 
# '-1' to extract labels from output head, otherwise 'layer number' to extract features from
layer=3

# Create experiment folder name where the clusters will be saved
if [ ${is_mfcc} == true ]; then
	exp_dir="${EXP_PATH}/clustering/mfcc"
else
	if [ "${extract}" == "cluster_ids" ]; then
		exp_dir="${EXP_PATH}/clustering/avhubert_${model}_${layer}_layer_AVHubertIDs"
	else
		exp_dir="${EXP_PATH}/clustering/avhubert_${model}_${layer}_layer_feat_kmeans"
	fi
fi
lab_dir="${exp_dir}/labels"

mkdir -pv ${exp_dir}/feats
mkdir -pv ${lab_dir}

cd ${RUN_PATH}

if [[ ${extract} == "k_means" ]]; then
	start_stage=0
	stop_stage=3

	# variable definition
	km_path="${exp_dir}/k_means_model"
	n_cluster=100
	

	if [ ${start_stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
		if [[ ${is_mfcc} == true ]]; then
			echo "Stage 0: Extracting mfcc features."
			python dump_mfcc_feature.py ${tsv_dir} ${split} ${nshard} ${rank} ${exp_dir}/feats
		else
			echo "Stage 0: Exctracting features from ${layer} layer of the model."
			python dump_hubert_feature.py ${tsv_dir} ${split} ${ckpt_path} ${layer} ${nshard} ${rank} ${exp_dir}/feats --user_dir "/scratch/work/sarvasm1/av_hubert/avhubert"
		fi
	fi
	if [ ${start_stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
		if [ ${split} == "train" ]; then
			echo "Stage 1: Performing k-means clustering."
			python learn_kmeans.py ${exp_dir}/feats ${split} ${nshard} ${km_path} ${n_cluster} --percent 0.1 
		else
			echo "Skipping training the k-means model for ${split} split."
		fi
	fi

	if [ ${start_stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
		echo "Stage 2: Apply learned k-means."
		python dump_km_label.py ${exp_dir}/feats ${split} ${km_path} ${nshard} ${rank} ${lab_dir}

	fi

	if [ ${start_stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
		echo "Stage 3: Dump cluster IDs from numpy to text file"
		# converts stored binary to text file where each line is sentence of avhubert cluster IDs
		file_name="${split}_${rank}_${nshard}"
		
		python ${PROJ_DIR}/src/get_labels.py --feats ${exp_dir}/labels/${file_name}.km \
																		--type "kmeans" \
																		--labels ${lab_dir}/${file_name}

	fi
fi

if [ "${extract}" == "cluster_ids" ]; then
	echo "Stage 0: Exctracting labels from output layer of the model."
	python dump_hubert_labels.py ${tsv_dir} ${split} ${ckpt_path} ${layer} ${nshard} ${rank} ${exp_dir}/feats --user_dir "../"
	file_name="${split}_${rank}_${nshard}"
	# converts stored binary to text file where each line is sentence of avhubert cluster IDs
	python ${PROJ_DIR}/src/get_labels.py --feats ${exp_dir}/feats/${file_name}.npy \
																	--lens ${exp_dir}/feats/${file_name}.len  \
																	--labels ${lab_dir}/${file_name}
fi
