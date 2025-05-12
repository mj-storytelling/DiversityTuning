from typing import List
import nltk

def ngram_diversity_score(
        data: List[str],
        num_n: int = 4, 
) -> float:
    """ Calculates corpus-level ngram diversity based on unique ngrams 
       (e.g., https://arxiv.org/pdf/2202.00666.pdf).

    Args:
        data (List[str]): List of documents. 
        num_n (int): Max ngrams to test up to. Defaults to 5. 

    Returns:
        float: ngram diveristy score.
    """
    score = 0 
    data = ' '.join(data).split(' ') # format to list of words

    for i in range(1, num_n + 1): 
        ngrams = list(nltk.ngrams(data, i))
        # num unique ngrams / all ngrams for each size n 
        score += len(set(ngrams)) / len(ngrams) 

    return round(score, 3)
