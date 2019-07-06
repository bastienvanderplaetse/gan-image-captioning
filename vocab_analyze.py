import argparse
import sys
import string

import matplotlib.pyplot as plt
import numpy as np

import utils.explorer_helper as exh


def check_args(argv):
    """Checks and parses the arguments of the command typed by the user
    Parameters
    ----------
    argv :
        The arguments of the command typed by the user
    Returns
    -------
    ArgumentParser
        the values of the arguments of the commande typed by the user
    """
    parser = argparse.ArgumentParser(description="Build vocabulary corpus from a set of captions.")
    parser.add_argument('INPUT', type=str, help="file containing the set of captions")
    # parser.add_argument('OUTPUT', type=str, help="name of the file in which the corpus must be saved")
    # parser.add_argument('MAX', type=int, help="maximum words to keep in corpus")

    args = parser.parse_args()

    return args

def ratio(tokens, n_captions):
    covered = set()
    for token in tokens:
        covered = covered|set(token[1]['captions'])

    return len(covered), len(covered)/n_captions

def plot_lines(threshold_l, lines, labels, filename, label_x, label_y, has_legend=True, step=2):
    fig, ax = plt.subplots()

    for index, line in enumerate(lines):
        ax.plot(threshold_l, line, label=labels[index])


    ax.set_ylabel(label_y)
    ax.set_xlabel(label_x)

    plt.xticks(np.arange(min(threshold_l)-1, max(threshold_l)+1, step))

    if has_legend:
        lgd = ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.savefig(filename, bbox_extra_artists=(lgd,), bbox_inches="tight")
    else:
        plt.savefig(filename)
    plt.close(fig)
    plt.clf()

def run(args):
    vocab = {}
    # vocab = {"<pad>": {"id": 0, "freq": float('inf')},
    #       "<bos>": {"id": 1, "freq": float('inf')},
    #       "<eos>": {"id": 2, "freq": float('inf')},
    #       "<unk>": {"id": 3, "freq": float('inf')}}
    # tokens = ["<pad>", "<bos>", "<eos>", "<unk>"]
    captions = exh.read_file(args.INPUT)

    n_captions = len(captions)

    id_token = 4

    for i, caption in enumerate(captions):
        for word in caption.split():
            if word in vocab:
                vocab[word]["freq"] += 1
                vocab[word]["captions"].add(i)
            else:
                # id_token = len(tokens)
                vocab[word] = dict()
                vocab[word]["freq"] = 1
                vocab[word]["id"] = id_token
                vocab[word]["captions"] = set([i])
                id_token += 1
                # tokens.append(word)

    top = sorted(vocab.items(), key=lambda x: x[1]['freq'], reverse=True)

    n_vocab = len(top)
    # print(top)
    # print(n_captions)
    # print(n_vocab)

    tots = []
    ratios = []
    x_ticks = []
    covered = set()

    for i in range(n_vocab):
        x_ticks.append(i+1)
        covered = covered|set(top[i][1]['captions'])
        tots.append(len(covered))
        ratios.append(len(covered)/n_captions)

    plot_lines(x_ticks, [tots], ['Number of captions covered'], 'voc_tot.png', 'Number of words', 'Number of captions covered', has_legend=False, step=100)
    plot_lines(x_ticks, [ratios], ['Ratio of captions covered'], 'voc_ratio.png', 'Number of words', 'Ratio of captions covered', has_legend=False, step=100)


    # top = top[:args.MAX+4]
    
    # tokens = [None] * (args.MAX+4)
    # vocab = dict()

    # id_token = 4
    # for word in top:
    #     id = word[1]['id']
    #     if id < 4:
    #         vocab[word[0]] = word[1]
    #         tokens[word[1]['id']] = word[0]
    #     else:
    #         vocab[word[0]] = {'id': id_token, 'freq': word[1]['freq']}
    #         tokens[id_token] = word[0]
    #         id_token += 1

    # vocab["token_list"] = tokens
    # exh.write_json(vocab, args.OUTPUT)
    

if __name__ == "__main__":
    args = check_args(sys.argv)
    run(args)