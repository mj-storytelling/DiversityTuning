
from typing import List, Tuple
import nltk 


def token_patterns(
        data: List[str],
        n: int,
        top_n: int = 10
) -> List[Tuple[str, int]]:
    """ Finds ngrams patterns in the data.

    Args:
        data (List[str]): Data to run frequency counts on.
        n (int): N-gram length.
        top_n (int, optional): Top patterns to display. Defaults to 10.

    Returns:
        List[Tuple[str, int]]: Sorted list of top n-gram patterns.
    """

    # treat data as one string 
    all_data = ' '.join(data)

    ngrams = list(nltk.ngrams(all_data.split(' '), n))
    frequency = nltk.FreqDist(ngrams)

    sorted_frequency = sorted(frequency.items(), key=lambda kv: kv[1], reverse=True)[:top_n]

    sorted_frequency = [(' '.join(x[0]), x[1]) for x in sorted_frequency]

    return sorted_frequency
