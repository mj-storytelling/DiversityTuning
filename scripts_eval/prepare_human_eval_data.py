import pandas as pd
from models import AnthropicModel
import random

generation_model = AnthropicModel("claude-3-5-sonnet-20241022")
output_dir = "data/eval/human_eval_DDPO_total.csv"

skip_id_dir = "data/eval/human_eval_DDPO_vs_DPO.csv"

if skip_id_dir != None:
  skip_df = pd.read_csv(skip_id_dir)
  skip_ids = skip_df['prompt_id'].unique()
else:
  skip_ids = []

files = [
  "data/eval/Llama-3.1-8B/gen_eval1_DDPO_sem_sty_total.csv",
  "data/eval/Llama-3.1-8B/gen_eval1_DPO_fixed.csv",
  "data/eval/gen_eval1_gpt-4o.csv",
]

conditions = [
  "DDPO",
  "DPO",
  "gpt-4o",
]

dfs = []

for file in files:
  df = pd.read_csv(file)
  dfs.append(df)

output_df = None
prompt_id_list = dfs[0]['prompt_id'].unique()
random.shuffle(prompt_id_list)
for prompt_id in prompt_id_list:
  if prompt_id in skip_ids:
    continue
  
  cur_sub_dfs = [ df[df['prompt_id'] == prompt_id] for df in dfs ]
  cur_prompt = cur_sub_dfs[0].iloc[0]['prompt']
  if "[removed]" in cur_prompt:
    print(f"Prompt ID: {prompt_id} has been removed")
    continue
  
  # mix the order of conditions
  mixed_order = list(range(len(conditions)))
  random.shuffle(mixed_order)

  responses_summarized = []

  for idx in mixed_order:
    cur_sub_df = cur_sub_dfs[idx]
    responses = cur_sub_df['response'].tolist()
    cur_responses_summarized = []
    for response in responses:
      prompt = f"A story is written based on the following prompt: {cur_prompt}\n\nSummarize the plot of the story in a paragraph, as ordered within the story, in 100 words. Start with how the beginning story relates to the prompt. Do not preamble and just give me the summarization, without any appending explanation: {response}"
      summarized = generation_model.generate(response, max_tokens=512, temperature=0.0, system="You are a story summarizer. You only provide a summary, without any preamble or explanation.")
      cur_responses_summarized.append(summarized[0])

    responses_summarized.append(cur_responses_summarized)

  if output_df is None:
    output_df = pd.DataFrame(columns=[
      'prompt_id', 'prompt', 
      'response1_1', 'response1_2', 'response1_3', 'response1_4', 
      'response2_1', 'response2_2', 'response2_3', 'response2_4', 
      'response1_1_summarized', 'response1_2_summarized', 'response1_3_summarized', 'response1_4_summarized',
      'response2_1_summarized', 'response2_2_summarized', 'response2_3_summarized', 'response2_4_summarized',
      'condition1', 
      'condition2'
    ])

  # make the task text in html that is a div side-by-side
  task_text = f"""
  <h4>Prompt</h4>
  <p>{cur_prompt}</p>
  <div style="display: flex; justify-content: space-between;">
    <div style="width: 49%; padding: 10px; border: 1px solid black;">
      <h4>Set A.</h4>
      <ol>
        <li>{responses_summarized[0][0]}</li>
        <li>{responses_summarized[0][1]}</li>
        <li>{responses_summarized[0][2]}</li>
        <li>{responses_summarized[0][3]}</li>
      </ol>
    </div>
    <div style="width: 49%; padding: 10px; border: 1px solid black;">
      <h4>Set B.</h4>
      <ol>
        <li>{responses_summarized[1][0]}</li>
        <li>{responses_summarized[1][1]}</li>
        <li>{responses_summarized[1][2]}</li>
        <li>{responses_summarized[1][3]}</li>
      </ol>
    </div>
  </div>
  """

  
  output_df = pd.concat([output_df, pd.DataFrame({
    'prompt_id': [prompt_id],
    'prompt': [cur_prompt],
    'response1_1': [cur_sub_dfs[mixed_order[0]].iloc[0]['response']],
    'response1_2': [cur_sub_dfs[mixed_order[0]].iloc[1]['response']],
    'response1_3': [cur_sub_dfs[mixed_order[0]].iloc[2]['response']],
    'response1_4': [cur_sub_dfs[mixed_order[0]].iloc[3]['response']],
    'response2_1': [cur_sub_dfs[mixed_order[1]].iloc[0]['response']],
    'response2_2': [cur_sub_dfs[mixed_order[1]].iloc[1]['response']],
    'response2_3': [cur_sub_dfs[mixed_order[1]].iloc[2]['response']],
    'response2_4': [cur_sub_dfs[mixed_order[1]].iloc[3]['response']],
    'response1_1_summarized': [responses_summarized[0][0]],
    'response1_2_summarized': [responses_summarized[0][1]],
    'response1_3_summarized': [responses_summarized[0][2]],
    'response1_4_summarized': [responses_summarized[0][3]],
    'response2_1_summarized': [responses_summarized[1][0]],
    'response2_2_summarized': [responses_summarized[1][1]],
    'response2_3_summarized': [responses_summarized[1][2]],
    'response2_4_summarized': [responses_summarized[1][3]],
    'text': [task_text],
    'condition1': [conditions[mixed_order[0]]],
    'condition2': [conditions[mixed_order[1]]],
  })])
  output_df.to_csv(output_dir, index=False)
  print(f"Prompt ID: {prompt_id} - Done")
  if len(output_df) >= 100:
    break
