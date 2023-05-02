import os
import argparse
import collections
from math import ceil

class CustomDataLoader():
    def __init__(self, cluster_path, text_path):
        with open(text_path, "r") as f_text, open(cluster_path, "r") as f_clusters:
            self.txt_data = f_text.readlines()
            self.cluster_data = f_clusters.readlines()

        self.cluster_data, self.txt_data = self.prep_data()


    def prep_data(self):
        new_cluster_data, new_text_data = [], []

        for id_line, txt_line in zip(self.cluster_data, self.txt_data):
            new_cluster_data.append(id_line.replace("\n", "").split(" "))
            new_text_data.append(list(txt_line.replace("\n", "").replace(" ", "")))

        return new_cluster_data, new_text_data

    def get_utt(self):
        # todo yield text and cluster ID utterence
        pass



class MatchClusters():

    def __init__(self, epochs=2):
        self.sos = "<sos>"
        self.eos = "<eos>"
        self.clusters = {}
        self.epochs = epochs


    def assign_sos_eos(self, id_utt):
        sos = id_utt[0]
        eos = id_utt[-1]
        for ids, char in zip([sos, eos], [self.sos, self.eos]):
            if ids in self.clusters:
                self.clusters[ids].update(char)
            else:
                self.clusters[ids] = collections.Counter({char: 1})
        return id_utt[1:-1]

    def init_dict(self, id_utt, txt_utt):
        id_utt = self.assign_sos_eos(id_utt)

        id_len, txt_len = len(id_utt), len(txt_utt)
        n_to_assign = round(id_len/txt_len)

        char_idx = 0
        i = 0
        for ids in id_utt:
            if i == n_to_assign: # start assigning next character
                i = 0
                char_idx += 1 # next char
                txt_len -= 1
                # if only last char in text utt left, assign it to all remaining clusters
                if txt_len == 1:
                    n_to_assign = -1
                    char_idx -= 1

            # assign characters from text utterence to cluster ID
            char_to_assign = txt_utt[char_idx]
            if ids in self.clusters:
                self.clusters[ids].update(char_to_assign)
            else:
                self.clusters[ids] = collections.Counter({char_to_assign: 1})
            i += 1


    def compute_cer(self):
        pass

    def run(self):
        for i in range(self.epochs):
            self.run_one_epoch()

    def run_one_epoch(self):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str)
    parser.add_argument('--clusters', type=str)
    parser.add_argument('--exp_dir', type=str)
    parser.add_argument('--iters', type=int, default=5)

    args = parser.parse_args()

    with open(args.text, "r") as f_text, open(args.clusters, "r") as f_clusters:
        txt_data = f_text.readlines()
        cluster_data = f_clusters.readlines()

    mc = MatchClusters()

    for id_line, txt_line in zip(cluster_data, txt_data):
        c_id = id_line.replace("\n", "").split(" ")
        txt = list(txt_line.replace("\n", "").replace(" ", ""))

        mc.init_dict(c_id, txt)

    for key, val in mc.clusters.items():
        print(key, val)

    for id_line, txt_line in zip(cluster_data, txt_data):
        c_id = id_line.replace("\n", "").split(" ")
        txt = list(txt_line.replace("\n", "").replace(" ", ""))

        mc.init_dict(c_id, txt)
