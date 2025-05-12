from models import CreativeWritingRewardModel, FlattenedDataset
import matplotlib.pyplot as plt
from tqdm import tqdm

from argparse import ArgumentParser
import pandas as pd
# draw linear regression
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr

parser = ArgumentParser()
# add the arguments
parser.add_argument("--modelname", default="google/gemma-2-2b", type=str, help="Model name")
parser.add_argument("--lora_model_path", default="checkpoints/reward_models/google_gemma-2-2b_10.pt", type=str, help="Path to the model")
parser.add_argument("--val_path", default="data/eval/val_results_google_gemma-2-2b.csv", type=str, help="Path to the dataset")
parser.add_argument("--device", default="cuda:0", type=str, help="Device to use")
parser.add_argument("--val_dataset", default="data/writingPrompt_flattened_val", type=str, help="Path to the dataset")

args = parser.parse_args()

# check if file exists in val_path
import os
print(os.path.exists(args.val_path), args.val_path)
if not os.path.exists(args.val_path):
  if args.lora_model_path == 'None':
    print('loading reward model without lora')
    reward_model = CreativeWritingRewardModel(args.modelname, device=args.device)
  else:
    reward_model = CreativeWritingRewardModel(args.modelname, lora_model_path = args.lora_model_path, device=args.device)
  val_dataset = reward_model.load_dataset(None, args.val_dataset)

  texts = []
  gold = []
  pred = []
  # batch of 16
  for i in tqdm(range(0, len(val_dataset), 4)):
    batch = [val_dataset[i+j] for j in range(4)]
    # print(batch)
    cur_texts = [row[0] for row in batch]
    cur_pred = reward_model.get_reward(cur_texts)
    cur_pred = [row.item() for row in cur_pred]
    # print(cur_pred)
    gold = gold + [row[1].item() for row in batch]
    pred = pred + cur_pred
    texts = texts + cur_texts
    # print(val_dataset[i][0], val_dataset[i][1].item(), cur_pred)
    # break
    if i % 96 == 0 and i >0:
      plt.scatter(gold, pred, alpha=0.002)
      gold_ = np.array(gold).reshape(-1, 1)
      pred_ = np.array(pred).reshape(-1, 1)
      reg = LinearRegression().fit(gold_, pred_)
      plt.plot(gold_, reg.predict(gold_), color='red')
      print('spearmans:', spearmanr(gold, pred))
      print('coefficient:', reg.coef_)
      print('intercept:', reg.intercept_)
      print('r2:', reg.score(gold_, pred_))
      print('mae:', mean_absolute_error(gold_, pred_))
      print("------")

      plt.savefig('scatter.png')
      plt.close()

  
  df = pd.DataFrame({'text': texts, 'gold': gold, 'pred': pred})

  df.to_csv(args.val_path, index=False)
else:
  df = pd.read_csv(args.val_path)
  gold = df['gold']
  pred = df['pred']
  texts = df['text']
  




gold = np.array(gold).reshape(-1, 1)
pred = np.array(pred).reshape(-1, 1)

# filter where gold is above -0.9
# ngold = gold[gold > -0.9]
# pred = pred[gold > -0.9]
# gold = ngold.reshape(-1, 1)
# pred = pred.reshape(-1, 1)

c = 0
for i in range(len(gold)):
  if gold[i] < -0.9:
    print(texts[i])
    print('------------------------------------\n\n\n\n')
    c=c+1
  if c > 10:
    break

reg = LinearRegression().fit(gold, pred)
print('spearmans r:', spearmanr(gold, pred))
print('coefficient:', reg.coef_)
print('intercept:', reg.intercept_)
print('r2:', reg.score(gold, pred))
print('mse:', mean_absolute_error(gold, pred))
plt.plot(gold, reg.predict(gold), color='red')


plt.scatter(gold, pred, alpha=0.02, color='red')
plt.ylim(bottom=-1.1, top=1.1)
plt.savefig('scatter_final.png')
plt.close()
plt.hist(pred, bins=100)
plt.savefig('pred_hist_final.png')



# print absolute error
abs_error = abs(gold-pred)
print('mean abs error:', np.mean(abs_error))
print('median abs error:', np.median(abs_error))
print('max abs error:', np.max(abs_error))
print('min abs error:', np.min(abs_error))
print('std abs error:', np.std(abs_error))