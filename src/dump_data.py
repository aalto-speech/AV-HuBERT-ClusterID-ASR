from num2words import num2words
train_path="/scratch/work/sarvasm1/AV-HuBERT-ClusterID-ASR/dump/valid.wrd"
prep_path="/scratch/work/sarvasm1/AV-HuBERT-ClusterID-ASR/dump/valid.wrd_tokens_sep"

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

#convert("/scratch/work/sarvasm1/av_hubert/ussee/dump/valid.wrd","/scratch/work/sarvasm1/av_hubert/ussee/dump/valid.wrd_clean")

with open(train_path, "r", encoding='utf-8') as f:
    with open(prep_path, "w", encoding='utf-8') as fnew:
        for line in f.readlines():
            sep_line = line.replace(" ", "|").replace("\n", "")
            dump_line = " ".join(list(sep_line))
            fnew.write(dump_line+"\n")
