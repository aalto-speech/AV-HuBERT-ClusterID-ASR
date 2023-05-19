from num2words import num2words
import shutil

clean = True
splits = ["train", "valid", "test"]
data_path = "/m/teamwork/t40511_asr/c/LRS3-TED/lrs3/30h_data"
dump_path = "/scratch/work/sarvasm1/AV-HuBERT-ClusterID-ASR/dump_test"

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


for split in splits:
    train_file = data_path+"/"+split+".wrd"
    prep_file = dump_path+"/"+split+".wrd_tokens_sep"
    copy_file = dump_path+"/"+split+".wrd"

    shutil.copyfile(train_file, copy_file)

    with open(train_file, "r", encoding='utf-8') as f, open(prep_file, "w", encoding='utf-8') as fnew:
        for line in f.readlines():
            sep_line = line.replace(" ", "|").replace("\n", "")
            dump_line = " ".join(list(sep_line))
            fnew.write(dump_line+"\n")

    if clean:
        clean_file = dump_path+"/"+split+".wrd_clean"
        convert(copy_file, clean_file)

        with open(clean_file, "r", encoding='utf-8') as f, open(clean_file+"_tokens_sep", "w", encoding='utf-8') as fnew:
            for line in f.readlines():
                sep_line = line.replace(" ", "|").replace("\n", "")
                dump_line = " ".join(list(sep_line))
                fnew.write(dump_line+"\n")
