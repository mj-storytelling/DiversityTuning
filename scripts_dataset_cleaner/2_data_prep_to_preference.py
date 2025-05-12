import datasets
from datasets import Dataset
from argparse import ArgumentParser
from tqdm import tqdm

import random

from models.utils import sample_pairs_with_gap

# set arguments
parser = ArgumentParser()
# add the arguments
parser.add_argument("--data_path_pair", default="data/writingPrompt_post_pair", type=str, help="Path to the dataset")
parser.add_argument("--data_path_filtered", default="data/writingPrompt_post", type=str, help="Path to the dataset")
parser.add_argument("--source", default="data/writingPrompt_cleaned", type=str, help="Path to the dataset")
parser.add_argument("--gap", default=5, type=int, help="Gap between the scores")

args = parser.parse_args()

# load the dataset
ds = datasets.load_from_disk(args.source)
print(type(ds), ds.keys())

def filtering(example, idx, sampled_index):
  example["posttraining_comment_text"] = [comment for idx, comment in enumerate(example["filtered_comment_texts"]) if idx in sampled_index]
  example["posttraining_comment_score"] = [score for idx, score in enumerate(example["filtered_comment_scores"]) if idx in sampled_index]
  example["posttraining_transformed_score"] = [score for idx, score in enumerate(example["filtered_transformed_scores"]) if idx in sampled_index]

  return example

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



  for sub_ds_idx, example in tqdm(enumerate(sub_ds)):
    prompt = f"{example['post_title']}\n{example['post_text']}"
    filtered_comment_texts = example['filtered_comment_texts']
    filtered_comment_scores = example['filtered_comment_scores']

    if args.gap == -1:
      # all pairs
      sampled_pairs = []
      for i in range(len(filtered_comment_scores)):
        for j in range(i+1, len(filtered_comment_scores)):
          sampled_pairs.append([(filtered_comment_scores[i], i), (filtered_comment_scores[j], j)])
      # randomly sample at maximum 100
      if len(sampled_pairs) > 100:
        sampled_pairs = random.sample(sampled_pairs, 100)
      print(len(filtered_comment_scores), len(sampled_pairs))
    else:
      sampled_pairs, _ = sample_pairs_with_gap(filtered_comment_scores, min_gap=args.gap)

    sampled_indexes[f"{sub_ds_idx}-{prompt}"] = [sampled_element[1] for sampled_pair in sampled_pairs for sampled_element in sampled_pair]
    
    for sampled_pair in sampled_pairs:
      if sampled_pair[0][0] > sampled_pair[1][0]:
        chosen_idx = sampled_pair[0][1]
        rejected_idx = sampled_pair[1][1]
      else:
        chosen_idx = sampled_pair[1][1]
        rejected_idx = sampled_pair[0][1]
    
      # add it to the paired data
      chosen_text = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": filtered_comment_texts[chosen_idx]}
      ]
      rejected_text = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": filtered_comment_texts[rejected_idx]}
      ]
      chosen_pair.append(chosen_text)
      rejected_pair.append(rejected_text)
      score_chosen_pair.append(filtered_comment_scores[chosen_idx])
      score_rejected_pair.append(filtered_comment_scores[rejected_idx])
  # create a new dataset
  new_subds = {
    'chosen': chosen_pair,
    'rejected': rejected_pair,
    'score_chosen': score_chosen_pair,
    'score_rejected': score_rejected_pair,
  }
  print(len(chosen_pair))
  new_ds_pair[key] = Dataset.from_dict(new_subds)
  # print the length of the new dataset
  print('dataset length', len(new_ds_pair[key]))
  if args.data_path_filtered != "None":
    ds[key] = sub_ds.map(lambda example, idx: filtering(example, idx, sampled_indexes[f"{idx}-{example['post_title']}\n{example['post_text']}"]), with_indices=True)

  



new_ds_pair = datasets.DatasetDict(new_ds_pair)

new_ds_pair.save_to_disk(args.data_path_pair)
if args.data_path_filtered != "None":
  ds.save_to_disk(args.data_path_filtered)