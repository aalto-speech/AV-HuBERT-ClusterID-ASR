import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle

import ast

def parse_file(path, output_folder):
    """
        Parse attention log file in openNMT format when using '--attn_debug' parameter during translatation

        Parameters:
        -----------
        path: str or list
            Path to a file containing attention value logs.
    """
    with open(path, "r", encoding='utf-8') as f:
        raw_text = f.readlines()
    line_idx = 1
    block = []
    sentence_idx = 0
    attention_dict = {}
    for line in raw_text:
        if line_idx % 3 == 0:
            # parse one attention sentence 
            print("Parsing sentence: ", sentence_idx)
            data, clusterIDs, tokens = load_sentence_data(block)
            current_sentence = "sentence_"+str(sentence_idx)
            attention_dict[current_sentence] = {"data": data, "clusters": clusterIDs, "char_tokens": tokens}
            if sentence_idx < 50:
                plot(data, clusterIDs, tokens, current_sentence, save_fig=output_folder)
            sentence_idx += 1
            line_idx += 1
            block = []
        if line == "\n":
            line_idx += 1
        else:
            block.append(line)

    with open(output_folder+"/attentions.pkl", 'wb') as handle:
            pickle.dump(attention_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_sentence_data(file_path):
    """
        Loads attention data for character-token combinations from a file or a list of lines.

        Parameters:
        -----------
        file_path: str or list
            The path to the file containing the attention data, or a list of lines
            with the same format as the file.

        Returns:
            tuple: A tuple containing the following elements:
                - att_vals (numpy.ndarray): A 2D array of shape (N, M) containing the attention values for
                    N character-token combinations and M attention heads.
                - clusterID (list): A list of cluster IDs that correspond to the character tokens.
                - chars (list): A list of characters that correspond to the character tokens.
        Raises:
                    ValueError: If the `file_path` argument is neither a string nor a list.

        Example usage:
            att_vals, clusterID, chars = load_sentence_data("attention_data.txt")
    """
    if isinstance(file_path, str):
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.readlines()
    elif isinstance(file_path, list):
        raw_text = file_path
    else:
        raise ValueError('A very specific bad thing happened.')
    # get list of clusterIDs
    clusterID = ast.literal_eval(raw_text[1].split(": ")[1])

    # attention values for each character-token combination
    att_data = raw_text[5:-1]
    chars = []
    att_vals = []
    for att in att_data:
        sample = att.strip("\n").replace("*", "")
        sample = list(filter(None, sample.split(" ")))
        if sample[0] == "|":
            chars.append("<sep>")
        else:
            chars.append(sample[0])
        att_vals.append(np.array(sample[1:], dtype=np.float32))

    return np.vstack(att_vals), clusterID, chars


def plot(data, clusterID, chars, title, save_fig="."):
    """
        Plots a heatmap of attention values cluster and character IDs for each sentence in the given data.

        Parameters:
        -----------
        data : numpy array
            The 2D numpy array containing the attention data to be plotted as a heatmap.
        clusterID : list
            A list containing the IDs of the different clusters for the data.
        chars : list
            A list containing the IDs of the different characters for the data.
        save_fig: str
            Path to the folder where plots will be saved.
        title : int or str, optional
            An optional parameter that sets the title of the plot.
        """
    fig, ax = plt.subplots(figsize=(len(clusterID)/3, len(chars)/5))
    im = ax.imshow(data, aspect='auto')

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(clusterID)), labels=clusterID)
    ax.set_yticks(np.arange(len(chars)), labels=chars)

    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
                      rotation_mode="anchor")
    ax.spines[:].set_visible(False)
    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)

    ax.grid(which="minor", color="white", linestyle='-', linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.colorbar(im, ax=ax)

    ax.set_title("")
    fig.tight_layout()
    plt.savefig(f"{save_fig}/{title}.png", dpi=150)
    plt.close()


if __name__ == '__main__':
    # Create the argument parser object
    parser = argparse.ArgumentParser()

    # Add the arguments for the log file and output folder
    parser.add_argument('--attention_log_file', metavar='LOG_FILE', type=str, help='path to the log file containing attention logs from openNMT translate')
    parser.add_argument('--output_folder', metavar='OUTPUT_FOLDER', type=str, help='path to the output folder, where attention plots will be stored')

    # Parse the arguments from the command line
    args = parser.parse_args()

    parse_file(args.attention_log_file, args.output_folder)
    #data, clusterIDs, tokens = load_sentence_data(att_path)
    #plot(data, clusterIDs, tokens)
