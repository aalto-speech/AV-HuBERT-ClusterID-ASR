#!/bin/bash

#SBATCH --time=6:00:00
#SBATCH --mem=8G
#SBATCH --output=dataprep_landmarks_xae.out
#SBATCH --job-name=landmarks_xae

module load miniconda
module load ffmpeg

source activate avhubert

PROJ_DIR=$(pwd)
PREP_PATH="avhubert/preparation"
TOOLS="${PROJ_DIR}/tools" # path to landmark and face predictor models


cd $PREP_PATH

# in case of splitting the data files into more subfiles
file_prefix=xae

# data preparation variables
start_stage=3
stop_stage=3

lrs3="${PROJ_DIR}/lrs3_dataset"
ffmpeg_path=$(which ffmpeg)
landmark_dir=landmark
rank=0
nshard=1



# stage 3
vocab_size=1000


if [ ${start_stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "Stage 0: Data preparation, extracting audio and creating file.list and label.list files"
    #python lrs3_prepare.py --lrs3 ${lrs3} --ffmpeg ${ffmpeg_path} --rank ${rank} --nshard ${nshard} --step 1
    #python lrs3_prepare.py --lrs3 ${lrs3} --ffmpeg ${ffmpeg_path} --rank ${rank} --nshard ${nshard} --step 2
    #python lrs3_prepare.py --lrs3 ${lrs3} --ffmpeg ${ffmpeg_path} --rank ${rank} --nshard ${nshard} --step 3 
    python lrs3_prepare.py --lrs3 ${lrs3} --ffmpeg ${ffmpeg_path} --rank ${rank} --nshard ${nshard} --step 4 
fi


if [ ${start_stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Stage 1: Detect facial landmark and crop mouth ROIs, creating \${lrs3}/video"
    python detect_landmark.py --root ${lrs3} \
                            --landmark ${lrs3}/${landmark_dir} \
                            --manifest ${lrs3}/${file_prefix}_file.list \
                            --ffmpeg ${ffmpeg_path} \
                            --rank ${rank} --nshard ${nshard} \
                            --cnn_detector ${TOOLS}/mmod_human_face_detector.dat \
                            --face_predictor ${TOOLS}/shape_predictor_68_face_landmarks.dat \

    python align_mouth.py --video-direc ${lrs3} \
                        --landmark-direc ${lrs3}/${landmark_dir} \
                        --filename-path ${lrs3}/${file_prefix}_file.list \
                        --save-direc ${lrs3}/video \
                        --ffmpeg ${ffmpeg_path} \
                        --rank ${rank} --nshard ${nshard} \
                        --mean-face ${TOOLS}/20words_mean_face.npy
fi
if [ ${start_stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "Stage 2: Count number of frames per clip. "

    python count_frames.py --root ${lrs3} --manifest ${lrs3}/file.list --nshard ${nshard} --rank ${rank}
    
    echo "Stage 2: Merging shards"
    for rank in $(seq 0 $((nshard - 1)));do cat ${lrs3}/nframes.audio.${rank}; done > ${lrs3}/nframes.audio
    for rank in $(seq 0 $((nshard - 1)));do cat ${lrs3}/nframes.video.${rank}; done > ${lrs3}/nframes.video
fi
if [ ${start_stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "Stage 3: Set up data directory"

    python lrs3_manifest.py --lrs3 ${lrs3} \
                            --vocab-size ${vocab_size} \
                            --valid-ids "${PROJ_DIR}/avhubert/preparation/data/lrs3-valid.id"
fi
