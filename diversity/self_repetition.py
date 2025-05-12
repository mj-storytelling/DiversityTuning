import numpy as np
import nltk

from tqdm import tqdm
from typing import List

def self_repetition_score(
        dataset: List[str],
        n: int = 4
) -> float:
    """Calculates a self-repetition score for a dataset based on the 
    repetition of ngrams within the corpus.

    Args:
        dataset (List[str]): A list of documents (strings) to analyze.
        n (int): Size of the ngrams to check for repetition. Defaults to 4.

    Returns:
        float: The self-repetition score, averaged over the dataset.
    """
    total_sum = 0

    for doc in tqdm(dataset, desc="Calculating self-repetition score"):
        ngrams = [' '.join(ngram) for ngram in nltk.ngrams(doc.split(), n)]
        sum_ni = 0

        for ngram in ngrams:
            # Count occurrences of ngram in other documents (excluding current doc)
            n_i = sum(1 for d in dataset if ngram in d and d != doc)

            sum_ni += (n_i + 1)

        total_sum += np.log(sum_ni)

    return total_sum / len(dataset)
