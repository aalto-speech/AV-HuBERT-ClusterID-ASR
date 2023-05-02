import argparse
import os

def compute_CI(freq_dict, c, N):
    freq_sum = 0
    for key, val in freq_dict.items():
        val = int(val)
        freq_sum += val*(val-1)

    denominator = N*(N-1)

    print("IC: ", freq_sum / (denominator / c))
   # print("IC (no normalization): ", freq_sum / denominator)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, help="path to a file with ngram frequencies")

    args = parser.parse_args()

    freq_dict = {}
    c = 0
    N = 0

    file_name = os.path.basename(args.file)
    print("================", file_name, "================")

    with open(args.file, "r") as f:
        for line in f.readlines():
            key, val = line.replace("\n", "").split(":")
            freq_dict[key] = val
            c += 1
            N += int(val)

    compute_CI(freq_dict, c, N)
