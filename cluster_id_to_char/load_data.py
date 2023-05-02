import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext
from torchtext.data import get_tokenizer
from torchvision.transforms import ToTensor
from tqdm import tqdm
import collections

from cluster_dataloader import clusterDataset
from convert_numers import convert

import numpy as np


clusterID_file_path = "/scratch/work/sarvasm1/av_hubert/ussee/exp/clustering/avhubert_base_lrs3_iter4_output_head/labels/train_0_1"
text_file_path = "/scratch/work/sarvasm1/av_hubert/ussee/exp/clustering/train.wrd"
clean_text_path = "/scratch/work/sarvasm1/av_hubert/ussee/exp/clustering/train.wrd_clean"

class ClusterDatasetVocab():
    def __init__(self, text_file, cluster_id_file, specials=["<sos>", "<eos>", "<pad>", "<unk>"]):
        self.text = []
        self.clusters = []
        self.specials = specials


        text = self._load_data(text_file)
        cluster_ids = self._load_data(cluster_id_file)
        self.clusterID_vocab, self.text_vocab = self._build_vocabs(text, cluster_ids)

        self.text = np.asarray(self.text)
        self.clusters = np.asarray(self.clusters)

        self.src_maxlen = len(max(self.clusters, key=len))
        self.tgt_maxlen = len(max(self.text, key=len))
        self.src_vocab_size = len(self.clusterID_vocab)
        self.tgt_vocab_size = len(self.text_vocab)

    def _load_data(self, path):
        with open(path, "r") as f:
            data = f.read().split("\n")
        return data

    def _build_vocabs(self, text_data, cluster_data):
        char_counter = collections.Counter()
        id_counter = collections.Counter()

        for txt, c_id in zip(text_data, cluster_data):
            txt = list(txt)
            c_id = c_id.split()
            char_counter.update(txt)
            id_counter.update(c_id)

            # store data
            self.text.append(txt)
            self.clusters.append(c_id)

        # somehow empty array was created at the end -> remove
        self.text = self.text[:-1]
        self.clusters = self.clusters[:-1]

        return torchtext.vocab.vocab(id_counter, specials=self.specials),  torchtext.vocab.vocab(char_counter, specials=self.specials)

    def get_max_val(self, which_data):
        if which_data == "text":
            pass
        else:
            pass

