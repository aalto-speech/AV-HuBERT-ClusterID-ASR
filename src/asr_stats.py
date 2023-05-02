import argparse

from jiwer import wer as j_wer
from jiwer import cer as j_cer
import jiwer

def parse_arguments():
    parser = argparse.ArgumentParser(description='Load files containing model predicitons and target sentences to compute WER and CER')
    parser.add_argument('--hyp', type=str, help='Path to the first file')
    parser.add_argument('--ref', type=str, help='Path to the second file')
    args = parser.parse_args()
    return args.hyp, args.ref

def compute_error_rates(hyp_list, ref_list):
    """
    Computes the Word Error Rate (WER) and Character Error Rate (CER)
    between two lists of sentences.

    Args:
        hyp_list (list): A list of hypitheses predicted by the model.
        ref_list (list): A list of reference sentences.

    Returns:
        (float, float): A tuple containing the WER and CER.
    """
    new_hyp_list  = []
    new_ref_list  = []

    trim = False
    #for src, tgt in zip(pred_list, target_list):
    #    s_len = len(src)
    #    t_len = len(tgt)
    #    print("PRED  : ", src)
    #    print("TARGET: ", tgt)
    #    wer_tmp = torchmetrics.functional.word_error_rate(src, tgt)
    #    cer_tmp = torchmetrics.functional.char_error_rate(src, tgt)
    #    print(f"WER: {wer_tmp}%, CER: {cer_tmp}% ")
    #    print("================================================")

    #wer = j_wer(pred_list, target_list)
    #cer = j_cer(pred_list, target_list)
    #print("jiwer  : ", wer, cer)

    out = jiwer.process_words(ref_list, hyp_list)

    print(jiwer.visualize_alignment(out))
    return None, None
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
