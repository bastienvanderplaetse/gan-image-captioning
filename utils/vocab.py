import numpy as np

import utils.explorer_helper as exh

def words2tokens(sentence, vocab, tokenized=True):
    """Convert a sentence from words to corresponding tokens based on a vocabulary corpus
    Parameters
    ----------
    filename : str
        The sentence to convert
    vocab : dict
        The vocabulary corpus
    tokenized : boolean
        Add <bos> and <eos> iff ``True``
    Returns
    -------
    list
        the token list corresponding to the sentence
    """
    tokens = []
    for word in sentence.split():
        if word in vocab:
            tokens.append(vocab[word]["id"])
        else:
            tokens.append(vocab["<unk>"]["id"])

    if tokenized:
        tokens = [vocab["<bos>"]["id"]] + tokens + [vocab["<eos>"]["id"]]
    
    return tokens

def tokens2words(tokens, vocab):
    """Convert a token list from tokens to corresponding words based on a vocabulary corpus
    Parameters
    ----------
    filename : list
        The list of tokens
    vocab : dict
        The vocabulary corpus
    Returns
    -------
    str
        the sentence corresponding to the token list
    """
    words = []

    for token in tokens:
        if token == 2: # if <eos>
            break
        words.append(vocab['token_list'][token])

    return ' '.join(words)

def init_weights(vocab, emb_dim):
    vocab_size = len(vocab['token_list'])
    sd = 1/np.sqrt(emb_dim)
    weights = np.random.normal(0, scale=sd, size=[vocab_size, emb_dim])
    weights = weights.astype(np.float32)
    
    return weights

def glove_weights(weights, glove_file, vocab):
    f = open(glove_file, encoding="utf-8")
    glove_lines = f.readlines()
    for line in glove_lines:
        line = line.split()
        token = vocab.get(line[0], None)

        if token is not None:
            id_word = token['id']
            weights[id_word] = np.array(line[1:], dtype=np.float32)