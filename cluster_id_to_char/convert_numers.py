import os
import argparse
from num2words import num2words

def convert(input_f, output_f):
    with open(input_f, "r") as f1, open(output_f, "w") as f2:
        translate_chars = str.maketrans('', '', "\{\}_-',")
        print("Cleaning text file: ", input_f)
        for i, line in enumerate(f1):
            new_line = ""
            for word in line.split(" "):
                try:
                   word = num2words(word)
                except:
                    pass
                new_line += word+" "
            new_line = new_line[:-1].translate(translate_chars).replace("\n", "")
            f2.write(new_line+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_text_file', type=str, help="Path to file with text data")
    parser.add_argument('--output_text_file', type=str, help="Path to file for clean text")


    args = parser.parse_args()
    convert(args.input_text_file, args.output_text_file)
