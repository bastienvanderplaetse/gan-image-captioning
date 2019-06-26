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