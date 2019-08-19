import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import nltk
import subprocess
import sys

import utils.explorer_helper as exh

from copy import deepcopy
from matplotlib.gridspec import GridSpec
from nltk.translate.bleu_score import corpus_bleu
# from nltk.translate.meteor_score import meteor_score
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

def prepare_references_meteor(references):
    refs = []
    references = deepcopy(references)
    for ref in references:
        if len(ref) < 7:
            for i in range(7-len(ref)):
                ref.append(ref[-1])
        for sentence in ref:
            refs.append(' '.join(sentence))

    return refs

def prepare_hypothesis(hypothesis):
    return [hyp.split() for hyp in hypothesis]

def plot_serie(scores, filename):
    for i in range(5):
        score = scores[i]
        fig = plt.figure(constrained_layout=True)

        gs = GridSpec(8, 6, figure=fig)

        ax1 = fig.add_subplot(gs[0,:])
        ax1.axis('off')
        ax1.text(0, 0, 'Ground truth\n'+'\n'.join(score['references']))

        ax2 = fig.add_subplot(gs[1:7,:])
        ax2.axis('off')
        ax2.imshow(mpimg.imread(args.IMAGES + '/' + score['image']))

        ax3 = fig.add_subplot(gs[7,:])
        ax3.axis('off')
        ax3.text(0, 0, 'Generated caption :\n' + score['hypothesis'] + '\nBLEU score : ' + str(score['bleu']))

        fig.suptitle(score['image'])
        # plt.show()
        plt.savefig(filename + str(i+1) + ".png")

def run(args):
    references = exh.read_file(args.REFERENCES)
    references = prepare_references(references)
    meteor_refs = prepare_references_meteor(references)
    exh.write_text('\n'.join(meteor_refs), args.REFERENCES+'_0')

    hypothesis = exh.read_file(args.HYPOTHESIS)
    hypothesis = prepare_hypothesis(hypothesis)

    links = exh.read_file(args.LINKS)

    weights = [1/4] * 4

    print("METEOR")
    # subprocess.call(['java', '-jar', '../meteor-1.5/meteor-1.5.jar', args.HYPOTHESIS, args.REFERENCES+'_0', '-l', 'en', '-r', '7', '-q'])
    total_bleu = corpus_bleu(references, hypothesis, weights)
    print("BLEU : {}".format(total_bleu))

    scores = [0] * len(links)

    for i in range(len(links)):
        scores[i] = dict()

        hyps = ' '.join(hypothesis[i])
        refs = [' '.join(ref) for ref in references[i]]
        scores[i]['references'] = refs
        scores[i]['hypothesis'] = hyps
        scores[i]['image'] = links[i]
        scores[i]['bleu'] = corpus_bleu([references[i]], [hypothesis[i]], weights)
        # scores[i]['meteor'] = meteor_score(refs, hyps, alpha=0.85, beta=0.2, gamma=0.6)

    scores = sorted(scores, key = lambda i: (i['bleu']))

    max_scores = scores[len(links)-5:]
    min_scores = scores[:5]

    plot_serie(max_scores, "top")
    plot_serie(min_scores, "flop")


if __name__ == "__main__":
    args = check_args(sys.argv)
    run(args)
