import numpy
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class clusterDataset2(Dataset):
    def __init__(self, vocab, device="cpu", transform=None, sos=0, eos=1):
        self.transform = transform
        self.device = device

        self.sos = sos
        self.eos = eos

        self.text_utts = vocab.text
        self.ids_utts = vocab.clusters
        self.vocab = vocab
        #self.txt_len = max([len(x) for x in self.text_utts])
        #self.c_len = max([len(x) for x in self.ids_utts])
        #self.max_len = max(self.txt_len, self.c_len)


    #def load_data(self, text_path, ids_path):
    #    c_data = []
    #    text_data = []
    #    txt_lens, c_lens = [], []

    #    with open(text_path, "r") as f_text, open(ids_path, "r") as f_clusters:
    #        for txt_line, ids_line in zip(f_text, f_clusters):
    #            text_data.append([self.sos]+list(txt_line.replace("\n", "").replace(" ", "").replace("'", ""))+[self.eos])
    #            c_data.append([self.sos]+ids_line.replace("\n", "").split(" ")+[self.eos])


    #    return text_data, c_data

    def __len__(self):
        return len(self.text_utts)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        ids_utt = self.vocab.clusterID_vocab.lookup_indices(self.ids_utts[idx])
        #list(map(int, self.ids_utts[idx]))
        text_utt = self.vocab.text_vocab.lookup_indices(self.text_utts[idx])
        #[self.char_mapping(x) for x in self.text_utts[idx]]

        if self.transform is not None:
            ids_utt = self.transform(ids_utt, device=self.device)
            text_utt = self.transform(text_utt, device=self.device)

        #ids_utt = F.pad(ids_utt,(0, self.max_len - ids_utt.shape[-1]))
        #text_utt = F.pad(text_utt, (0, self.max_len - text_utt.shape[-1]))

        return ids_utt, text_utt
class clusterDataset(Dataset):
    def __init__(self, text_path, ids_path, char_mapping, device="cpu", transform=None, sos=0, eos=1):
        self.char_mapping = char_mapping
        self.transform = transform
        self.device = device

        self.sos = sos
        self.eos = eos

        self.text_utts, self.ids_utts = self.load_data(text_path, ids_path)
        self.txt_len = max([len(x) for x in self.text_utts])
        self.c_len = max([len(x) for x in self.ids_utts])
        self.max_len = max(self.txt_len, self.c_len)


    def load_data(self, text_path, ids_path):
        c_data = []
        text_data = []
        txt_lens, c_lens = [], []

        with open(text_path, "r") as f_text, open(ids_path, "r") as f_clusters:
            for txt_line, ids_line in zip(f_text, f_clusters):
                text_data.append([self.sos]+list(txt_line.replace("\n", "").replace(" ", "").replace("'", ""))+[self.eos])
                c_data.append([self.sos]+ids_line.replace("\n", "").split(" ")+[self.eos])


        return text_data, c_data

    def __len__(self):
        return len(self.text_utts)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        ids_utt = list(map(int, self.ids_utts[idx]))
        text_utt = [self.char_mapping(x) for x in self.text_utts[idx]]

        if self.transform is not None:
            ids_utt = self.transform(ids_utt, device=self.device)
            text_utt = self.transform(text_utt, device=self.device)

        #ids_utt = F.pad(ids_utt,(0, self.max_len - ids_utt.shape[-1]))
        #text_utt = F.pad(text_utt, (0, self.max_len - text_utt.shape[-1]))

        return ids_utt, text_utt
