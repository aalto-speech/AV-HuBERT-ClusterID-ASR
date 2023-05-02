import argparse
import numpy as np

def get_labels(args):

    index = 0
    data = np.load(args.feats)
    with open(args.lens, "r") as f, open(args.labels, "w") as lab_f, open(args.labels+"_rm_dup", "w") as dup_f:
        for line in f.readlines():
            feat_len = int(line)
            labels = data[index:index+feat_len]
            no_dup_labs = remove_duplicates(labels)
            lab_f.write(' '.join(map(str, labels))+"\n")
            dup_f.write(' '.join(map(str, no_dup_labs))+"\n")
            index += feat_len

def remove_duplicates(labels):
    labels = labels[np.insert(np.diff(labels).astype(bool), 0, True)]
    return labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--feats', type=str, help="Path to file with .npy files to extract labels from")
    parser.add_argument('--lens', type=str, help="Path to file lengths for feats content")
    parser.add_argument('--labels', type=str, help="Output file for the labels")

    args = parser.parse_args()
    get_labels(args)
