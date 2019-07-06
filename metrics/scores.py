from nltk.translate.bleu_score import corpus_bleu

def prepare_references(references):
    refs = []
    for reference in references:
        captions = reference.split('###')
        captions = [caption.split() for caption in captions]
        refs.append(captions)
    return refs

def prepare_hypothesis(hypothesis):
    return [hyp.split() for hyp in hypothesis]

# def bleu_score_4(references, hypothesis):
#     return bleu_score(references, hypothesis, [0,0,0,1])

def bleu_score(references, hypothesis, n):
    weights = [1/n] * n
    print(weights)
    return corpus_bleu(references, prepare_hypothesis(hypothesis), weights)