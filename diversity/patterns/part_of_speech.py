from typing import List, Tuple, Any, Set

import spacy

def get_pos(
        data: List[str]
) -> Tuple[List[str], List[Tuple[str, str]]]:
    """ Turns a sequence into parts of speech.

    Args:
        data (List[str]): Data to tranform into part of speech tags.

    Returns:
        Tuple[List[str], List[Tuple[str, str]]]: Part-of-speech tags only, tuple of (token, part-of-speech tag).
    """
    nlp = spacy.load("en_core_web_sm")

    tokenized_data = [x.split() for x in data]

    pos_tuples = []
    joined_pos = []
    joined_text = []

    for tokens in tokenized_data:

        doc = nlp(' '.join(tokens))
        
        joined_text.append(' '.join([token.text for token in doc]))
        joined_pos.append(' '.join([token.tag_ for token in doc]))
        
        pos_tuples.append([(token.text, token.tag_) for token in doc])

    return joined_pos, pos_tuples


def _find_sub_list(
        sl: List[Any],
        l: List[Any]
) -> List[Any]:
    """ Given a pattern and a list of strings, returns sublists matching the pattern. """
    
    results = []
    sll = len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            results.append((ind, ind+sll-1))

    return results


def pos_patterns(
        text: List[List[Tuple[str, str]]],
        pattern: str
) -> Set[str]: 
    """ Finds all substrings matching a part of speech pattern. 

    Args:
        text (List[List[Tuple[str, str]]]): Text containing words and part-of-speech tags.
        pattern (str): Part-of-speech tag pattern to search for.

    Returns:
        Set[str]: Returns all the string matching the pattern.
    """

    pos = []
    word = []
    
    # text is a list of lists of tuples (word, part of speech)
    for doc in text: 
        pos.append([i[1] for i in doc])
        word.append([i[0] for i in doc])
    
    pos = [' '.join(x) for x in pos]
    word = [' '.join(x) for x in word]

    all_matches = [] 

    # return positions of each tag and the corresponding tokens
    for w, p in zip(word, pos):

        test = _find_sub_list(pattern.split(), p.split())

        if test:
            for occ in test: 
                splits = w.split()[int(occ[0]):int(occ[1]+1)]
                all_matches.append(" ".join(splits))

    return set(all_matches) 
