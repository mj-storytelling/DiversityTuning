import datasets
from argparse import ArgumentParser
from tqdm import tqdm

from models import DiversityModel

# set arguments
parser = ArgumentParser()
# add the arguments
parser.add_argument("--data_path_pair_div", default="data/writingPrompt_post_pair_sem_sty", type=str, help="Path to the dataset")
parser.add_argument("--source_pair", default="data/writingPrompt_post_pair", type=str, help="Path to the dataset")
# should include "semantic" or "style"
parser.add_argument("--score_process", default=["semantic", "style"], nargs='+', type=str, help="Score processing method")
parser.add_argument("--device", default="cuda:0", type=str, help="Device to use for training")
parser.add_argument("--max_cutoff", default=-1, type=int, help="Max cutoff")
parser.add_argument("--min_score_non_zero", default=False, type=bool, help="Whether too keep the minimum score non zero or not.")
parser.add_argument("--sort_with_score", default=False, type=bool, help="Whether to sort with score or not.")

args = parser.parse_args()

# load the dataset
ds_paired = datasets.load_from_disk(args.source_pair)

comment_scores = {}
comments = {}

winners = {}
losers = {}

# only put necessary models below
div_models_dict = {
  "semantic": DiversityModel("jinaai/jina-embeddings-v3", device = args.device),
  "style": DiversityModel("AnnaWegmann/Style-Embedding", device = args.device)
}
div_models = []

for key in args.score_process:
  div_models.append(div_models_dict[key])

# get comment score info
for key in ds_paired:
  comment_scores[key] = {}
  comments[key] = {}
  winners[key] = {}
  losers[key] = {}
  # randomize
  if args.sort_with_score:
    print('start sorting...')
    # highest score first
    ds_paired[key] = ds_paired[key].sort("score_chosen", reverse=True)

  for example in tqdm(ds_paired[key]):
    prompt = example["chosen"][0]["content"]
    if prompt not in comment_scores[key]:
      comment_scores[key][prompt] = []
      comments[key][prompt] = []
      winners[key][prompt] = []
      losers[key][prompt] = []
    if args.max_cutoff != -1 and len(comment_scores[key][prompt]) >= args.max_cutoff:
      continue
    print(example["score_chosen"])
    comment_scores[key][prompt].append(example["score_chosen"])
    comment_scores[key][prompt].append(example["score_rejected"])
    winners[key][prompt].append(example["chosen"][1]["content"])
    losers[key][prompt].append(example["rejected"][1]["content"])
    comments[key][prompt].append(example["chosen"][1]["content"])
    comments[key][prompt].append(example["rejected"][1]["content"])
  print(key, len(ds_paired[key]))
# filter ds_paired if comments are not included in 'comments'
if args.max_cutoff != -1:
  for key in ds_paired:
    ds_paired[key] = ds_paired[key].filter(lambda example: example["chosen"][1]["content"] in comments[key][example["chosen"][0]["content"]])
    ds_paired[key] = ds_paired[key].filter(lambda example: example["rejected"][1]["content"] in comments[key][example["rejected"][0]["content"]])
    print(key, len(ds_paired[key]))
comment_embeddings = {}
iter_count = len(args.score_process)
print("iter_count", iter_count)
for iter_c in range(iter_count):
  comment_embeddings[iter_c] = {}
  for key in ds_paired:
    comment_embeddings[iter_c][key] = {}
    for prompt in tqdm(comments[key]):
      comment_embeddings[iter_c][key][prompt] = {}
      div_scores = div_models[iter_c].get_diversity(comments[key][prompt])
      for idx, comment in enumerate(comments[key][prompt]):
        comment_embeddings[iter_c][key][prompt][comment] = div_scores[idx]

def transform_scores(example, key):
  iter_count = len(args.score_process)
  
  iters_cur_diversity_scores = []
  iters_chosen_div_score = []
  iters_rejected_div_score = []
  for iter_c in range(iter_count):
    iter_cur_diversity_scores = comment_embeddings[iter_c][key][example["chosen"][0]["content"]]
    iter_chosen_div_score = iter_cur_diversity_scores[example["chosen"][1]["content"]]
    iter_rejected_div_score = iter_cur_diversity_scores[example["rejected"][1]["content"]]
    iter_cur_diversity_scores = list(iter_cur_diversity_scores.values())

    div_max = max(iter_cur_diversity_scores)
    div_min = min(iter_cur_diversity_scores)
    
    if div_max == div_min:
      # when the diversity scores are all the same
      iter_chosen_div_score = 0
      iter_rejected_div_score = 0
      iter_cur_diversity_scores = [0 for _ in iter_cur_diversity_scores]
    else:
      iter_chosen_div_score = 2 * (iter_chosen_div_score - div_min) / (div_max - div_min) - 1
      iter_rejected_div_score = 2 * (iter_rejected_div_score - div_min) / (div_max - div_min) - 1
      iter_cur_diversity_scores = [2 * (score - div_min) / (div_max - div_min) - 1 for score in iter_cur_diversity_scores]
    iters_cur_diversity_scores.append(iter_cur_diversity_scores)
    iters_chosen_div_score.append(iter_chosen_div_score)
    iters_rejected_div_score.append(iter_rejected_div_score)

  def zero_to_one(a):
    return (a + 1) / 2

  def minus_one_to_one(a):
    return 2 * a - 1

  # multiply different diversities
  cur_diversity_scores = []
  
  for idx in range(len(iters_cur_diversity_scores[0])):
    div_score = 1
    for iter_c in range(iter_count):
      div_score *= zero_to_one(iters_cur_diversity_scores[iter_c][idx]) ** (1/iter_count)
    cur_diversity_scores.append(minus_one_to_one(div_score))
  chosen_div_score = 1
  rejected_div_score = 1
  for iter_c in range(iter_count):
    chosen_div_score *= zero_to_one(iters_chosen_div_score[iter_c]) ** (1/iter_count)
    rejected_div_score *= zero_to_one(iters_rejected_div_score[iter_c]) ** (1/iter_count)
  chosen_div_score = minus_one_to_one(chosen_div_score)
  rejected_div_score = minus_one_to_one(rejected_div_score)

  example["score_chosen"] = chosen_div_score
  example["score_rejected"] = rejected_div_score
  scaled_comment_scores = cur_diversity_scores

  
  if min(scaled_comment_scores) != max(scaled_comment_scores):
    example["score_chosen"] = ( example["score_chosen"] * len(scaled_comment_scores) - sum(scaled_comment_scores) ) / (  len(scaled_comment_scores) + sum(scaled_comment_scores) )
    example["score_rejected"] = ( example["score_rejected"] * len(scaled_comment_scores) - sum(scaled_comment_scores) ) / (  len(scaled_comment_scores) + sum(scaled_comment_scores) )

  if example["score_rejected"] <-0.8 and args.min_score_non_zero:
    print("score exceeding -1?")
    example["score_rejected"] = -0.8
  return example

for key in ds_paired:
  ds_paired[key] = ds_paired[key].map(lambda example: transform_scores(example, key))

ds_paired.save_to_disk(args.data_path_pair_normal)