# code to prepare the dataset
# not super generalizable yet...
from datasets import load_dataset
import re
from models import transform_scores
from tqdm import tqdm
from argparse import ArgumentParser

# set arguments
parser = ArgumentParser()
# add the arguments
parser.add_argument("--data_path", default="data/writingPrompt_uncleaned", type=str, help="Path to the dataset")
parser.add_argument("--source", default="euclaise/WritingPrompts_preferences", type=str, help="Source of the dataset")

args = parser.parse_args()

train_ds = load_dataset(args.source)['train']


# clean the data - remove the first [] parenthesis from the 'post_title' with regex
def clean_title(x):
  # define regex
  pattern = r'^\[.*?\]'
  # find the regex pattern
  found = re.findall(pattern, x)
  # if the pattern is found, return the text except the first pattern
  if found:
    return x[len(found[0]):].strip()
  else:
    return x

# apply the clean_title function to the 'post_title' column
train_ds = train_ds.map(lambda x: {'post_title': clean_title(x['post_title'])})
import math
max_score = -math.inf
min_score = math.inf
total_instances = 0
for i in tqdm(train_ds):
  cur_max = max(i['comment_scores'])
  cur_min = min(i['comment_scores'])
  total_instances += len(i['comment_scores'])
  if cur_max > max_score:
    max_score = cur_max
  if cur_min < min_score:
    min_score = cur_min
print(f"Max score: {max_score}, Min score: {min_score}, total instances: {total_instances}")
train_ds = train_ds.map(lambda x: {'transformed_score': transform_scores([x['comment_scores']], min_score = min_score, max_score = max_score)[0]})

# split the train dataset into training and validation
train_ds = train_ds.train_test_split(test_size=0.1, seed=42)

# save the train and validation datasets
train_ds.save_to_disk(args.data_path)


