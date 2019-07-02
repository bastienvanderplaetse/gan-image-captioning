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
    parser.add_argument('MAX', type=int, help="maximum words to keep in corpus")

    args = parser.parse_args()

    return args

def run(args):
    vocab = {"<pad>": {"id": 0, "freq": float('inf')},
          "<bos>": {"id": 1, "freq": float('inf')},
          "<eos>": {"id": 2, "freq": float('inf')},
          "<unk>": {"id": 3, "freq": float('inf')}}
    # tokens = ["<pad>", "<bos>", "<eos>", "<unk>"]
    captions = exh.read_file(args.INPUT)

    id_token = 4

    for caption in captions:
        for word in caption.split():
            if word in vocab:
                vocab[word]["freq"] += 1
            else:
                # id_token = len(tokens)
                vocab[word] = dict()
                vocab[word]["freq"] = 1
                vocab[word]["id"] = id_token
                id_token += 1
                # tokens.append(word)

    top = sorted(vocab.items(), key=lambda x: x[1]['freq'], reverse=True)
    top = top[:args.MAX+4]
    
    tokens = [None] * (args.MAX+4)
    vocab = dict()

    id_token = 4
    for word in top:
        id = word[1]['id']
        if id < 4:
            vocab[word[0]] = word[1]
            tokens[word[1]['id']] = word[0]
        else:
            vocab[word[0]] = {'id': id_token, 'freq': word[1]['freq']}
            tokens[id_token] = word[0]
            id_token += 1

    vocab["token_list"] = tokens
    exh.write_json(vocab, args.OUTPUT)
    

if __name__ == "__main__":
    args = check_args(sys.argv)
    run(args)