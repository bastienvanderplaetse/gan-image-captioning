import argparse
import nltk
import sys

import utils.explorer_helper as exh

from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from pprint import pprint

nltk.download('wordnet')

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
    parser = argparse.ArgumentParser(description="Evaluate a set of hypothesis.")
    parser.add_argument('REFERENCES', type=str, help="file containing the references")
    parser.add_argument('HYPOTHESIS', type=str, help="file containing the hypothesis")
    parser.add_argument('LINKS', type=str, help="file containing the image filenames")
    parser.add_argument('IMAGES', type=str, help="directory containing the images")

    args = parser.parse_args()

    return args

def prepare_references(references):
    refs = []
    for reference in references:
        captions = reference.split('###')
        captions = [caption.split() for caption in captions]
        refs.append(captions)
    return refs

def prepare_hypothesis(hypothesis):
    return [hyp.split() for hyp in hypothesis]

def run(args):
    references = exh.read_file(args.REFERENCES)
    references = prepare_references(references)
    
    hypothesis = exh.read_file(args.HYPOTHESIS)
    hypothesis = prepare_hypothesis(hypothesis)
    
    links = exh.read_file(args.LINKS)

    weights = [1/4] * 4

    total_bleu = corpus_bleu(references, hypothesis, weights)

    scores = [0] * len(links)

    for i in range(len(links)):
        scores[i] = dict()

        hyps = ' '.join(hypothesis[i])
        refs = [' '.join(ref) for ref in references[i]]
        scores[i]['references'] = refs
        scores[i]['hypothesis'] = hyps
        scores[i]['image'] = links[i]
        scores[i]['bleu'] = corpus_bleu([references[i]], [hypothesis[i]], weights)
        scores[i]['meteor'] = meteor_score(refs, hyps)

    scores = sorted(scores, key = lambda i: (i['bleu']))

    max_scores = scores[len(links)-5:]
    pprint(max_scores)

    

if __name__ == "__main__":
    args = check_args(sys.argv)
    run(args)