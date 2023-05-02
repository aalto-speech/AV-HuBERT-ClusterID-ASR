import argparse
import torch
import torchmetrics


from jiwer import wer as j_wer
from jiwer import cer as j_cer

def parse_arguments():
    parser = argparse.ArgumentParser(description='Load files containing model predicitons and target sentences to compute WER and CER')
    parser.add_argument('--pred', type=str, help='Path to the first file')
    parser.add_argument('--target', type=str, help='Path to the second file')
    args = parser.parse_args()
    return args.pred, args.target

def compute_error_rates(pred_list, target_list):
    """
    Computes the Word Error Rate (WER) and Character Error Rate (CER)
    between two lists of sentences.

    Args:
        pred_list (list): A list of sentences predicted by the model.
        target_list (list): A list of target sentences.

    Returns:
        (float, float): A tuple containing the WER and CER.
    """
    new_pred_list  = []
    new_target_list  = []

    trim = False
    for src, tgt in zip(pred_list, target_list):
        s_len = len(src)
        t_len = len(tgt)
        print("PRED  : ", src)
        if trim:
            if s_len + 10 > t_len:
                src = src[:t_len]
                new_pred_list.append(src)
                new_target_list.append(tgt)
            print("PRED* : ", src)
        print("TARGET: ", tgt)
        wer_tmp = torchmetrics.functional.word_error_rate(src, tgt)
        cer_tmp = torchmetrics.functional.char_error_rate(src, tgt)
        print(f"WER: {wer_tmp}%, CER: {cer_tmp}% ")
        print("================================================")

    if trim:
        wer = j_wer(new_target_list, new_pred_list)
        cer = j_cer(new_target_list, new_pred_list)
        print("jiwer  : ", wer, cer)
        wer = torchmetrics.functional.word_error_rate(new_pred_list, new_target_list)
        cer = torchmetrics.functional.char_error_rate(new_pred_list, new_target_list)
        print("pytorch: ", wer, cer)
    else:
        wer = j_wer(target_list, pred_list)
        cer = j_cer(target_list, pred_list)
        print("jiwer  : ", wer, cer)
        wer = torchmetrics.functional.word_error_rate(pred_list, target_list)
        cer = torchmetrics.functional.char_error_rate(pred_list, target_list)
        print("pytorch: ", wer, cer)
    return wer, cer


def load_sentences(file_path):
    with open(file_path, 'r', encoding="utf-8") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    return lines

def load_pred_sentences(file_path, sep="|"):
    with open(file_path, 'r', encoding="utf-8") as f:
        lines = f.readlines()
    lines = [line.strip().replace(" ", "").replace(sep, " ") for line in lines]
    return lines


if __name__ == '__main__':
    pred_path, target_path = parse_arguments()
    wer, cer = compute_error_rates(load_pred_sentences(pred_path), load_sentences(target_path))
    #print(wer, cer)
