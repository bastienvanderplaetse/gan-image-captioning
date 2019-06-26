from nltk.translate.bleu_score import corpus_bleu

def prepare_references(references):
    return [[reference.split()] for reference in references]

def prepare_hypothesis(hypothesis):
    return [hyp.split() for hyp in hypothesis]

def bleu_score_4(references, hypothesis):
    return bleu_score(references, hypothesis, [0,0,0,1])

def bleu_score(references, hypothesis, weights):
    return corpus_bleu(references, prepare_hypothesis(hypothesis), weights)