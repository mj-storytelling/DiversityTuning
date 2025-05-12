# data preparation: for texts generated from the same prompt, measure the semantic and style diversity
import datasets
from argparse import ArgumentParser
from tqdm import tqdm
from transformers import AutoTokenizer

# set arguments
parser = ArgumentParser()
# add the arguments
parser.add_argument("--data_path", default="data/writingPrompt_cleaned", type=str, help="Path to the dataset")
parser.add_argument("--source", default="data/writingPrompt_uncleaned", type=str, help="Path to the dataset")
parser.add_argument("--model_to_use", default="meta-llama/Llama-3.1-8B-Instruct", type=str, help="Model to use for training")

args = parser.parse_args()

# load the dataset
ds = datasets.load_from_disk(args.source)

# # load the model
# semantic_div_model = DiversityModel("all-MiniLM-L6-v2", device = "cuda:0")
# style_div_model = DiversityModel("AnnaWegmann/Style-Embedding", device = "cuda:0")

tokenizer = AutoTokenizer.from_pretrained(args.model_to_use)
tokenizer.pad_token_id = tokenizer.eos_token_id

def remove_long(example):
  prompt = f"{example['post_title']}\n{example['post_text']}"
  messages = []
  for comment in example['filtered_comment_texts']:
    message = [
      {"role": "user", "content": prompt},
      {"role": "assistant", "content": comment}
    ]
    message_templated = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False)
    messages.append(message_templated)
  # tokenize the messages
  if len(messages) == 0:
    example['filtered_comment_texts'] = []
  else:
    messages = tokenizer(messages, padding=True, return_tensors="pt")
    # get the index of comment that is longer than 2048 tokens
    long_comments = messages['attention_mask'].sum(dim=1) <= 2048
    long_comments = long_comments.numpy()
    # get the text from row
    
    example['filtered_comment_texts'] = [ comment for comment, long_comment in zip(example['filtered_comment_texts'], long_comments) if long_comment]
    example['filtered_comment_scores'] = [ score for score, long_comment in zip(example['filtered_comment_scores'], long_comments) if long_comment]
    example['filtered_transformed_scores'] = [ score for score, long_comment in zip(example['filtered_transformed_scores'], long_comments) if long_comment]
  return example

total = 0
for key in ds:
  sub_ds = ds[key]
  for ridx, row in enumerate(tqdm(sub_ds['comment_texts'])):
    total += len(sub_ds[ridx]['comment_texts'])

def filtering(example):
  cur_comments = example['comment_texts']
  cur_comment_scores = example['comment_scores']
  cur_transformed_scores = example['transformed_score']
  new_cur_comments = []
  new_cur_comment_scores = []
  new_cur_transformed_scores = []
  for cur_comment, cur_comment_score, cur_transformed_score in zip(cur_comments, cur_comment_scores, cur_transformed_scores):
    if "**Welcome to the Prompt!**" in cur_comment:
      continue
    if "this submission has been removed" in cur_comment:
      continue
    if "**Off-Topic Discussion**" in cur_comment:
      continue
    

    new_cur_comments.append(cur_comment)
    new_cur_comment_scores.append(cur_comment_score)
    new_cur_transformed_scores.append(cur_transformed_score)
  example['filtered_comment_texts'] = new_cur_comments
  example['filtered_comment_scores'] = new_cur_comment_scores
  example['filtered_transformed_scores'] = new_cur_transformed_scores
  return example

for key in ds:
  ds[key] = ds[key].map(filtering)
  ds[key] = ds[key].map(remove_long)

remaining = 0

for key in ds:
  sub_ds = ds[key]
  for ridx, row in enumerate(tqdm(sub_ds['filtered_comment_texts'])):
    remaining += len(sub_ds[ridx]['filtered_comment_texts'])

# save the dataset
ds.save_to_disk(args.data_path)

print(f"Removed {total-remaining} out of {total} comments")


# print(ds)