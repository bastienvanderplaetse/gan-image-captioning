import argparse
import sys
import string

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
    parser.add_argument('OUTPUT', type=str, help="name of the file in which the corpus must be saved")

    args = parser.parse_args()

    return args

def run(args):
    vocab = {"<pad>": {"id": 0},
          "<bos>": {"id": 1},
          "<eos>": {"id": 2},
          "<unk>": {"id": 3}}
    tokens = ["<pad>", "<bos>", "<eos>", "<unk>"]
    captions = exh.read_file(args.INPUT)
    for caption in captions:
        for word in caption.split():
            if word in vocab:
                vocab[word]["freq"] += 1
            else:
                id_token = len(tokens)
                vocab[word] = dict()
                vocab[word]["id"] = id_token
                vocab[word]["freq"] = 1
                tokens.append(word)

    vocab["token_list"] = tokens
    exh.write_json(vocab, args.OUTPUT)
    

if __name__ == "__main__":
    args = check_args(sys.argv)
    run(args)