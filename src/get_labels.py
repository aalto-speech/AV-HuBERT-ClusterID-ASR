import argparse
import numpy as np
import shutil

def get_av_labels(args):
    # TODO: rename 'labels' to cluster IDs
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



def get_kmeans_labels(args):
    shutil.copyfile(args.feats, args.labels)

    with open(args.feats, "r") as f, open(args.labels+"_rm_dup", "w") as dup_f:
        for line in f.readlines():
            clusters = line.strip().split(" ")
            clusters = np.array(list(map(int, clusters)))
            no_dup_clust = remove_duplicates(clusters)
            dup_f.write(' '.join(map(str, no_dup_clust))+"\n")


def remove_duplicates(labels):
    labels = labels[np.insert(np.diff(labels).astype(bool), 0, True)]
    return labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--feats', type=str, help="Path to file with .npy files to extract labels from")
    parser.add_argument('--lens', type=str, default="",  help="Path to file lengths for feats content")
    parser.add_argument('--labels', type=str, help="Output file for the labels")
    parser.add_argument('--type', type=str, default='av', choices=['av', 'kmeans'], help="Type of 'feats' file, either numpy binary or k_means text file")

    args = parser.parse_args()
    if args.type == "av":
        get_av_labels(args)
    elif args.type == "kmeans":
        get_kmeans_labels(args)
    else:
        pass

