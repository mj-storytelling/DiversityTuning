# This code evaluates:
# 1) reddit-based reward model scores
# 2) semantic diversity
# 3) style diversity

# note that the output of this file will be used to evaluate the reward based on writing-reward model

from models import GenerationModel, CreativeWritingRewardModel, DiversityModel, OpenAIModel, AnthropicModel, TogetherModel
from argparse import ArgumentParser
import os
import pandas as pd
import datasets
from tqdm import tqdm

parser = ArgumentParser()
# add the arguments
parser.add_argument("--modelname", default="meta-llama/Llama-3.1-8B", type=str, help="Model name")
parser.add_argument("--lora_model_path", default="checkpoints/Llama-3.1-8B/generation_models_DDPO_sem_sty/ddpo_00.pt", type=str, help="Path to the model")
parser.add_argument("--val_path", default="data/eval/ablation/gen_eval1_DDPO_sem_sty_.csv", type=str, help="Path to the dataset")
parser.add_argument("--device", default="cuda:0", type=str, help="Device to use")

parser.add_argument("--rewardmodelname", default="google/gemma-2-2b", type=str, help="Reward model name")
parser.add_argument("--reward_lora_model_path", default="checkpoints/reward_models/google_gemma-2-2b_10.pt", type=str, help="Path to the reward model")

parser.add_argument("--semanticdivmodelname", default="jinaai/jina-embeddings-v3", type=str, help="Diversity model name")
parser.add_argument("--styledivmodelname", default="AnnaWegmann/Style-Embedding", type=str, help="Diversity model name")

args = parser.parse_args()

if 'gpt' in args.modelname or 'o1' in args.modelname:
  generation_model = OpenAIModel(args.modelname)

elif 'claude' in args.modelname:
  generation_model = AnthropicModel(args.modelname)

elif 'deepseek' in args.modelname:
  generation_model = TogetherModel(args.modelname)

else:
  generation_model = GenerationModel(args.modelname, 
      device=args.device, 
      # use_peft=False,
      lora_model_path = args.lora_model_path, lora_r = 128, lora_alpha = 256, 
      optimized=True,)
  generation_model.model.eval()
reward_model = CreativeWritingRewardModel(args.rewardmodelname, lora_model_path = args.reward_lora_model_path, device=args.device)
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
  if val_df is not None and i in val_df['prompt_id'].values:
    continue
  prompt = f"{val_dataset[i]['post_title']}\n{val_dataset[i]['post_text']}"
  if 'gpt' in args.modelname or 'o1' in args.modelname or 'claude' in args.modelname or 'deepseek' in args.modelname:
    outputs = generation_model.generate(prompt, num_return_sequences=4)

  elif "Instruct" in args.modelname:
    print("instruct...")
    outputs = generation_model.generate(
      [prompt], max_length=2048, num_return_sequences=4, repetition_penalty=1.1, temperature=1.0, top_k=50, top_p=0.95,
      developer_prompt = "You write a creative writing based on the user-given writing prompt."
    )
  else:
    outputs = generation_model.generate([prompt], max_length=2048, num_return_sequences=4, repetition_penalty=1.1, temperature=1.0, top_k=50, top_p=0.95)
  outputs_for_rm = [f"<prompt>{prompt}</prompt><response>{output}</response>" for output in outputs]
  rewards = reward_model.get_reward(outputs_for_rm)
  sem_div = semantic_div_model.get_diversity(outputs, metric='cosine')
  style_div = style_div_model.get_diversity(outputs, metric='cosine')
  for j in range(len(outputs)):
    print(outputs[j], rewards[j].item(), f"{round(sem_div[j],2)}", f"{round(style_div[j],2)}")
    if val_df is None:
      val_df = pd.DataFrame(columns=['prompt_id', 'prompt', 'response', 'reddit_reward', 'semantic_diversity', 'style_diversity',])
    val_df = pd.concat([val_df, pd.DataFrame({
      'prompt_id': [i],
      'prompt': [prompt],
      'response': [outputs[j]],
      'reddit_reward': [rewards[j].item()],
      'semantic_diversity': [sem_div[j]],
      'style_diversity': [style_div[j]],
    })])

  if i%1 == 0:
    val_df.to_csv(args.val_path, index=False)  
val_df.to_csv(args.val_path, index=False)