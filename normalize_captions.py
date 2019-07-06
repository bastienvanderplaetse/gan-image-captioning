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
    parser = argparse.ArgumentParser(description="Normalize the captions.")
    parser.add_argument('INPUT', type=str, help="the name of the file containing the captions to normalize")

    args = parser.parse_args()

    return args

def run(args):
    captions = exh.read_file(args.INPUT)
    norm_captions = []
    for caption in captions:
        norm_caption = "".join([w for w in caption if w not in string.punctuation])
        norm_caption = norm_caption.replace("  ", " ").lower()
        if norm_caption[-1] == " ":
            norm_caption = norm_caption[:-1]
        norm_captions.append(norm_caption)
    
    filename = args.INPUT.split('/')[-1]
    folder = args.INPUT[:len(args.INPUT)-len(filename)]
    output_file = "{0}norm_{1}".format(folder, filename)
    norm_captions = "\n".join(norm_captions)
    exh.write_text(norm_captions, output_file)
    

if __name__ == "__main__":
    args = check_args(sys.argv)
    run(args)