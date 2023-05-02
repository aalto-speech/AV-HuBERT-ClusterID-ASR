import argparse
import numpy as np

def get_labels(args):
    with open(args.labels, "r") as lab_f, open(args.labels+"_rm_dup", "w") as dup_f:
        for line in lab_f.readlines():
            no_dup_labs = remove_duplicates(line)
            dup_f.write(' '.join(map(str, no_dup_labs))+"\n")

def remove_duplicates(labels):
    labels = np.asarray(labels.replace("\n", "").split(" "), dtype=int)
    labels = labels[np.insert(np.diff(labels).astype(bool), 0, True)]
    return labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels')

    args = parser.parse_args()
    get_labels(args)
