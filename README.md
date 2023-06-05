# Leveraging cluster IDs of AV-HuBERT pre-trained models
Training small LSTM model to produce text from cluster IDs extracted from AV-HuBERT pre-trained models. Using OpenNMT's LSTM models. Trained on LRS3-30h dataset.
## Installation
- download and set up working AV-HuBERT directory as described in https://github.com/facebookresearch/av_hubert 
- have LRS3 data 
- clone this directory (change paths to data and other files in bash scripts)

## Prepare the data
Change paths in the script to match you working dir. Stages of data preparation are described in https://github.com/facebookresearch/av_hubert/tree/main/avhubert/preparation "**LRS3 Preprocessing**" part
```
bash run_lrs3_dataprep.sh
```
## Fine-tune pre-trained AV-HuBERT models
For finetuning pre-trained model just run the script below. Path to finetuning configuration and python environment should be changed if needed.
```
bash run_finetuning.sh
```

## Train LSTM models for ID-to-char translation
Feature / cluster IDs extraction from the AV-HuBERT models. 
- option for extracting intermediate features from n-th layer and then running k-means on top of them to get cluster IDs
- option for directly extracting cluster IDs from last layer of the model
Cluster IDs are stored in the specified experiments folder and used data are copied in *dump* directory for further experiments.
```
bash run_avhubert_feature_extraction.sh
bash dump.sh
```
Run to train OpenNMT model on cluster IDs. Model, training hyperparams and corpus selection is in configuration file in **conf/** directory (config is selected in the bash script)
```
bash run_cluster_training.sh
```
