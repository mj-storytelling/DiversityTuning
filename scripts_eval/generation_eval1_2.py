# This code evaluates:
# writing score from Tuhin et al.'s dataset

# note that this code makes use of the eval file generated from generation_eval1_1.py

from argparse import ArgumentParser
import os
import pandas as pd
import numpy as np
from diversity import compression_ratio, homogenization_score, ngram_diversity_score

parser = ArgumentParser()
# add the arguments
parser.add_argument("--val_path", default="data/eval", type=str, help="Path to the dataset")
parser.add_argument("--device", default="cuda:1", type=str, help="Device to use")
parser.add_argument("--val_files", default=[], nargs='+', type=str, help="List of val files")

args = parser.parse_args()

val_files = args.val_files

for val_file in val_files:
  print(val_file)
  val_df = pd.read_csv(os.path.join(args.val_path, val_file))
  if "compression_ratio" not in val_df.columns and "homogenization_score" not in val_df.columns and "ngram_diversity_score" not in val_df.columns:
    val_df["compression_ratio"] = np.nan
    val_df["homogenization_score"] = np.nan
    val_df["ngram_diversity_score"] = np.nan
  # if nan exists in the columns, then calculate the scores
  if val_df["compression_ratio"].isnull().values.any() or val_df["homogenization_score"].isnull().values.any() or val_df["ngram_diversity_score"].isnull().values.any():
    for prompt_id in val_df['prompt_id'].unique():
      sub_df = val_df[val_df['prompt_id'] == prompt_id]
      responses = sub_df['response'].tolist()
      responses = [str(r) for r in responses]
      hs = homogenization_score(responses, 'rougel')
      cr = compression_ratio(responses, 'gzip')
      ngram_div = ngram_diversity_score(responses, 4)
      print(prompt_id, hs, cr, ngram_div, len(sub_df))

      val_df.loc[val_df['prompt_id'] == prompt_id, "compression_ratio"] = cr
      val_df.loc[val_df['prompt_id'] == prompt_id, "homogenization_score"] = hs
      val_df.loc[val_df['prompt_id'] == prompt_id, "ngram_diversity_score"] = ngram_div
      if prompt_id % 50 == 0:
        val_df.to_csv(os.path.join(args.val_path, val_file), index=False)
  val_df.to_csv(os.path.join(args.val_path, val_file), index=False)
    