import nltk
from nltk.util import ngrams
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures

import os
import argparse
import collections

def compute_ngrams(N, data, data_format):
    counts = collections.Counter()

    for sent in data:
        if data_format == "char":
            sent = sent.replace("\n", "").replace(" ", "").replace("'", "")
            sent = list(sent)
            counts.update(nltk.ngrams(sent, N))
        else:
            counts.update(nltk.ngrams(sent.split(), N))

    return counts

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels', type=str, "Path to file with labels or text")
    parser.add_argument('--ngram', type=int, default=1)
    parser.add_argument('--format', type=str, default="id" choices=["char", "id"], help="If set to char some things such as \"'\" are removed)
    parser.add_argument('--exp_dir', type=str, "Path to directory to store the results")

    args = parser.parse_args()

    label_name = os.path.basename(args.data)

    with open(args.data, "r") as f:
        data = f.readlines()

    counts = compute_ngrams(args.ngram, data, args.format)

    filename = f"{label_name}.{args.format}.{args.ngram}gram"
    with open(os.path.join(args.exp_dir, filename), "w") as f:
        for key, c in counts.most_common():
            f.write("{}: {}\n".format(key, c))

