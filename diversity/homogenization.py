import numpy as np
import itertools
from diversity import memoized
from typing import List, Optional
from tqdm import tqdm
from rouge_score import rouge_scorer
from evaluate import load


def homogenization_score(
        data: List[str],
        measure: str = 'rougel',
        use_stemmer: Optional[str] = False,
        model: Optional[str] = 'distilbert-base-uncased',
        verbose: Optional[bool] = True

) -> float:
    """ Calculates the homogenization score for a set of documents (corpus-level). 
        From https://arxiv.org/pdf/2309.05196.pdf 
     Args:
         data (List[str]): Strings to score.
         measure (str, optional): Either 'rougel', 'bertscore', or 'bleu'. Defaults to 'rougel'.
         use_stemmer(str, optional): Whether to use stemming in the ROUGE-L calculation. Defaults to False.
         model(str, optional): Model to use for BERTScore. Defaults to 'microsoft/deberta-xlarge-mnli'. 
         verbose(bool, optional): Whether to display progress bar. Defaults to True.
     Returns:
         float: Homogenization score.
     """

    if measure == 'rougel':
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=use_stemmer)
    elif measure == 'bertscore': 
        scorer = load("bertscore")
    elif measure == 'bleu':
        scorer = load("bleu")
    else: 
        raise ValueError("Scoring measure must be one of `rougel`, `bleu`, or `bertscore`.")

    # all pairs of comparisons (cartesian product) without repetition at the same index
    all_pairs = ((data[i], data[j]) 
                    for i in range(len(data))
                    for j in range(len(data))
                    if i != j)

    curr_str = data[0]
    corpus_score = 0
    doc_score = 0
    
    print('==> Scoring all pairs')
    for pair in tqdm(all_pairs, total=len(data)**2 - len(data)):
        # single document-level homogenization score, pairs are ordered
        if pair[0] == curr_str:     
            doc_score += _calculate_score(pair, scorer, measure, model)
        else:
            corpus_score += doc_score / (len(data) - 1)
            curr_str = pair[0]
            doc_score = 0 
            doc_score += _calculate_score(pair, scorer, measure, model)
    
    # case where all strings are the exact same in the list
    if corpus_score == 0: 
        corpus_score += doc_score 
    
    # returns corpus level homogenization score 
    return round(corpus_score / len(data), 3)


@memoized
def _calculate_score(pair, scorer, measure, model):
    """ Returns the score of two strings """
    
    # repeated string (excat str match) that is *not* at the same position 
    if pair[0] == pair[1]: 
        return 1
    else:
        if measure == 'rougel':
            score = scorer.score(pair[0], 
                                 pair[1])['rougeL'][-1]
        elif measure == 'bertscore': 
            score = scorer.compute(predictions=[pair[0]], 
                                   references=[pair[1]], 
                                   model_type=model)['f1'][0]
        elif measure == 'bleu': 
            score = scorer.compute(predictions=[pair[0]], 
                                   references=[pair[1]])['bleu']
        
        return score 
