# This code evaluates:
# 1) reddit-based reward model scores
# 2) semantic diversity
# 3) style diversity

from models import DiversityModel
from argparse import ArgumentParser
import os
import pandas as pd
import datasets
from tqdm import tqdm
import numpy as np

parser = ArgumentParser()
# add the arguments
parser.add_argument("--val_path", default="data/eval/gen_eval1_gold.csv", type=str, help="Path to the dataset")
parser.add_argument("--device", default="cuda:0", type=str, help="Device to use")

parser.add_argument("--semanticdivmodelname", default="jinaai/jina-embeddings-v3", type=str, help="Diversity model name")
parser.add_argument("--styledivmodelname", default="AnnaWegmann/Style-Embedding", type=str, help="Diversity model name")

args = parser.parse_args()

semantic_div_model = DiversityModel(args.semanticdivmodelname, device = args.device)
style_div_model = DiversityModel(args.styledivmodelname, device = args.device)


# check if a file exists as val_path
if os.path.isfile(args.val_path):
  val_df = pd.read_csv(args.val_path)
else:
  val_df = None

ds = datasets.load_from_disk("data/writingPrompt_cleaned")
val_dataset = ds['test']
print(val_dataset)

# sample 1000
val_dataset = val_dataset.select(range(1000, 2000))

lengths = []
for i in tqdm(range(len(val_dataset))):
  # skip if i exists in the val_df
  if len(val_dataset[i]['filtered_comment_texts']) != len(val_dataset[i]['filtered_comment_scores']):
    assert False, f"Length mismatch at {i}"
  if val_df is not None and i in val_df['prompt_id'].values:
    continue
  prompt = f"{val_dataset[i]['post_title']}\n{val_dataset[i]['post_text']}"
  outputs = val_dataset[i]['filtered_comment_texts']

  if len(outputs) == 0:
    continue

  if len(outputs) > 1:
    sem_div = semantic_div_model.get_diversity(outputs, metric='cosine')
    style_div = style_div_model.get_diversity(outputs, metric='cosine')
  elif len(outputs) == 1:
    sem_div = [np.nan]
    style_div = [np.nan]
  for j in range(len(outputs)):
    if val_df is None:
      val_df = pd.DataFrame(columns=['prompt_id', 'prompt', 'response', 'data_semantic_diversity', 'data_style_diversity', 'gold_reddit_reward', 'gold_vote'])
    val_df = pd.concat([val_df, pd.DataFrame({
      'prompt_id': [i],
      'prompt': [prompt],
      'response': [outputs[j]],
      'data_semantic_diversity': [sem_div[j]],
      'data_style_diversity': [style_div[j]],
      'gold_reddit_reward': [val_dataset[i]['filtered_transformed_scores'][j]],
      'gold_vote': [val_dataset[i]['filtered_comment_scores'][j]],
    })])

  if i%10 == 0:
    val_df.to_csv(args.val_path, index=False)  
val_df.to_csv(args.val_path, index=False)