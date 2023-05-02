PREDICTIONS="/scratch/work/sarvasm1/av_hubert/ussee/exp/openNMT_id2char/transformer_0.01lr_4enc_dec_layers_6heads_512hidden_nodup_sep/test_0_1_predicted"
TARGET="/scratch/work/sarvasm1/av_hubert/ussee/dump/test.wrd"

metric="/scratch/work/sarvasm1/av_hubert/ussee/exp/openNMT_id2char/transformer_0.01lr_4enc_dec_layers_6heads_512hidden_clean_nodup_sep/metrics.out"

#python3 compute_metric.py --pred ${PREDICTIONS} --target ${TARGET} > ${metric}
python3 compute_metric.py --pred ${PREDICTIONS} --target ${TARGET}
