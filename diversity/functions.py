""" Functions for extracting patterns and matching text to patterns. """

import numpy as np
import itertools
from typing import List, Optional
from tqdm import tqdm
from .patterns import token_patterns, get_pos, pos_patterns
from nltk.tokenize import sent_tokenize

def extract_patterns(text: List[str], 
                     n: int = 5,
                     top_n: int = 100
) -> dict:
    """ Extracts text and part-of-speech patterns from text input. 
        Used to return a dictionary of patterns and the corresponding text match. 
    Args:
        text (List[str]): List of strings to extract patterns from.
        n (int, optional): N-gram size. Defaults to 5.
        top_n (int, optional): Number of top patterns to extract. Defaults to 100.
    Returns:
        dict: Dictionary of patterns and their corresponding text.

    Example Usage:
    >>> text = ["The quick brown fox jumps over the lazy dog.", 
                "The slow red fox walks on the hyper dog."]
    >>> extract_patterns(text, 4)

    {'DT JJ NN NN': {'The quick brown fox'},
    'JJ NN NN VBZ': {'quick brown fox jumps'},
    'NN NN VBZ IN': {'brown fox jumps over'},
    'NN VBZ IN DT': {'fox jumps over the'},
    'VBZ IN DT JJ': {'jumps over the lazy'},
    'IN DT JJ NN': {'over the lazy dog.'},
    'DT JJ JJ NN': {'The slow red fox'},
    'JJ JJ NN NNS': {'slow red fox walks'},
    'JJ NN NNS IN': {'red fox walks on'},
    'NN NNS IN DT': {'fox walks on the'},
    'NNS IN DT NN': {'walks on the hyper'},
    'IN DT NN NN': {'on the hyper dog.'}}
    """

    # sentence tokenize then search for patterns in the entire list 
    outputs = sent_tokenize(' '.join(text))

    # get the token (word)-level patterns
    patterns_token  =  token_patterns(outputs, n)

    # get the part-of-speech patterns (only include top_n patterns)
    joined_pos, tuples  =  get_pos(outputs)
    ngrams_pos  =  token_patterns(joined_pos, n, top_n)

    # for the top n-gram patterns, cycle through and get the matching text
    text_matches  = {}

    for  pattern, _  in  ngrams_pos:
        text_matches[pattern] =  pos_patterns(tuples, pattern)
    
    return text_matches


def match_patterns(text: str, 
                   patterns: dict
) -> List[tuple]:
    """ Matches text to part-of-speech patterns extracted from the `extract_patterns` function.
        Given set of patterns, used to identify which patterns appears in a single input text. 
    Args:
        text (str): Text to match patterns to.
        patterns (dict): Dictionary of patterns and their corresponding text.
    Returns:
        List[tuple]: List of tuples with the pattern and the text that matched.

    Example Usage: 
    >>> text = ["The quick brown fox jumps over the lazy dog.", 
        "The slow red fox walks on the hyper dog.",
        "The cranky blue cat scratches the calm fish." ]
    >>> patterns = extract_patterns(text, 4)
    >>> match_patterns(text[2], patterns)

    [('DT NN JJ NN', 'The cranky blue cat'),
    ('NN JJ NN VBZ', 'cranky blue cat scratches'),
    ('JJ NN VBZ DT', 'blue cat scratches the'),
    ('NN VBZ DT NN', 'cat scratches the calm'),
    ('VBZ DT NN NN', 'scratches the calm fish.')]
    """

    matches  =  []

    for pattern, text_match in patterns.items():
        for substr in text_match:
            if substr in text:
                matches.append((pattern, substr))

    return matches
