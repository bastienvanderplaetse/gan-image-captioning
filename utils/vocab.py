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