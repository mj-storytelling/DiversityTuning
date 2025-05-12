import datasets
from datasets import Dataset
from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np

import random
from models import DiversityModel

# set arguments
parser = ArgumentParser()
# add the arguments
parser.add_argument("--data_path_pair", default="data/writingPrompt_post_pair_highlow_quality_2_divscore_fixed", type=str, help="Path to the dataset")
parser.add_argument("--source", default="data/writingPrompt_post", type=str, help="Path to the dataset")
# should include "semantic" or "style"
parser.add_argument("--score_process", default=["semantic", "style"],  nargs='+', type=str, help="Score processing method")
parser.add_argument("--device", default="cuda:0", type=str, help="Device to use for training")
parser.add_argument("--max_pair_num", default=2, type=int, help="Maximum number of instances per prompt")
parser.add_argument("--rho", default=25, type=int, help="Percentage of high/low quality instances")

args = parser.parse_args()

# load the dataset
ds = datasets.load_from_disk(args.source)
print(type(ds), ds.keys())

# only put necessary models below
div_models_dict = {
  "semantic": DiversityModel("jinaai/jina-embeddings-v3", device = args.device),
  "style": DiversityModel("AnnaWegmann/Style-Embedding", device = args.device)
}
div_models = []

for key in args.score_process:
  div_models.append(div_models_dict[key])

new_ds_pair = {}

for key in ds:
  print(key)
  sub_ds = ds[key]
  # print columns in the dataset
  print(sub_ds.column_names)
  
  # for pair dataset
  chosen_pair = []
  rejected_pair = []
  score_chosen_pair = []
  score_rejected_pair = []

  # for non-paired dataset
  sampled_indexes = {}


  def normalize_scores(scores):
    # between -1 and 1
    max_score = max(scores)
    min_score = min(scores)
    if max_score == min_score:
      return [0] * len(scores)
    else:
      return [(score - min_score) / (max_score - min_score) * 2 - 1 for score in scores]

  def exponent_mult(a, b, weight):
        return a ** (1-weight) * b ** weight

  def zero_to_one(a):
    return (a + 1) / 2

  def minus_one_to_one(a):
    return 2 * a - 1

  def process_scores(filtered_comments_sub):
    processed_scores = []
    # print(filtered_comments_sub)
    div_scores = []

    for model in div_models:
      div_score = model.get_diversity(filtered_comments_sub, metric='cosine')
      div_score = normalize_scores(div_score)
      div_scores.append(div_score)

    # combine the scores
    for i in range(len(filtered_comments_sub)):
      if len(div_scores) == 1:
        cur_score = div_scores[0][i]
      else:
        cur_score = exponent_mult(zero_to_one(div_scores[0][i]), zero_to_one(div_scores[1][i]), 0.5)
      cur_score = minus_one_to_one(cur_score)
      processed_scores.append(cur_score)

    min_score = min(processed_scores)
    max_score = max(processed_scores)
    print(min(processed_scores), sum(processed_scores), max(processed_scores))

    # example["score_rejected"] = ( example["score_rejected"] * len(scaled_comment_scores) - sum(scaled_comment_scores) ) / (  len(scaled_comment_scores) + sum(scaled_comment_scores) )
    
    if min_score == max_score:
      to_return= [0] * len(processed_scores)
    else:
      to_return = [ (score * len(processed_scores) - sum(processed_scores)) / (len(processed_scores) + sum(processed_scores)) for score in processed_scores]
    print(min(to_return), sum(to_return), max(to_return))
    return to_return

  effective_filtering_cases = 0
  for sub_ds_idx, example in tqdm(enumerate(sub_ds)):
    prompt = f"{example['post_title']}\n{example['post_text']}"
    filtered_comment_texts = example['posttraining_comment_text']
    filtered_comment_scores = example['posttraining_comment_score']
    if len(filtered_comment_texts) == 0:
      continue

    sample_num = max(args.max_pair_num, len(filtered_comment_scores)*(args.rho/100))
    # floor down to integer
    if sample_num > len(filtered_comment_scores)/2:
      sample_num = len(filtered_comment_scores)/2
    else:
      effective_filtering_cases += 1
    sample_num = int(sample_num)
    max_num_pair = min(args.max_pair_num, sample_num)

    div_score_all = process_scores(filtered_comment_texts)

    # get highest scores set
    high_score_indexes = np.argsort(filtered_comment_scores)[::-1][:sample_num]
    filtered_comment_scores_high = [filtered_comment_scores[i] for i in high_score_indexes]
    filtered_comment_texts_high = [filtered_comment_texts[i] for i in high_score_indexes]
    div_score_high_pre = [div_score_all[i] for i in high_score_indexes]

    # get lowest scores set
    low_score_indexes = np.argsort(filtered_comment_scores)[:sample_num]
    filtered_comment_scores_low = [filtered_comment_scores[i] for i in low_score_indexes]
    filtered_comment_texts_low = [filtered_comment_texts[i] for i in low_score_indexes]
    div_score_low_pre = [div_score_all[i] for i in low_score_indexes]

    # get the diversity scores for high
    div_scores_high = process_scores(filtered_comment_texts_high)
    # get the diversity scores for low
    div_scores_low = process_scores(filtered_comment_texts_low)

    # get max_num_pair for high and low
    filtered_comment_texts_high = [filtered_comment_texts_high[i] for i in np.argsort(div_scores_high)[::-1][:max_num_pair]]
    filtered_comment_texts_low = [filtered_comment_texts_low[i] for i in np.argsort(div_scores_low)[:max_num_pair]]
    div_score_high_pre = [div_score_high_pre[i] for i in np.argsort(div_scores_high)[::-1][:max_num_pair]]
    div_score_low_pre = [div_score_low_pre[i] for i in np.argsort(div_scores_low)[:max_num_pair]]

    # randomize
    high_indexes = list(range(len(filtered_comment_texts_high)))
    low_indexes = list(range(len(filtered_comment_texts_low)))

    random.shuffle(high_indexes)
    random.shuffle(low_indexes)

    # combine the high and low into pairs
    for i in range(max_num_pair):
      high_idx = high_indexes[i]
      low_idx = low_indexes[i]
      chosen_text = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": filtered_comment_texts_high[high_idx]}
      ]
      rejected_text = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": filtered_comment_texts_low[low_idx]}
      ]
      chosen_score = div_score_high_pre[high_idx]
      rejected_score = div_score_low_pre[low_idx]
      chosen_pair.append(chosen_text)
      rejected_pair.append(rejected_text)
      score_chosen_pair.append(chosen_score)
      score_rejected_pair.append(rejected_score)
  print('effective_filtering_cases rate', effective_filtering_cases/len(sub_ds))
  # create a new dataset
  new_subds = {
    'chosen': chosen_pair,
    'rejected': rejected_pair,
    'score_chosen': score_chosen_pair,
    'score_rejected': score_rejected_pair
  }
  print(len(chosen_pair))
  new_ds_pair[key] = Dataset.from_dict(new_subds)
  # print the length of the new dataset
  print('dataset length', len(new_ds_pair[key]))

  



new_ds_pair = datasets.DatasetDict(new_ds_pair)

new_ds_pair.save_to_disk(args.data_path_pair)