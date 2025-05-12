from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model_state_dict, get_peft_model, set_peft_model_state_dict, PeftConfig, PeftModel
from accelerate import Accelerator
import accelerate
import os
import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
from tqdm import tqdm
from .utils import EarlyStopping, FlattenedDataset

from collections import OrderedDict

class CreativeWritingRewardModel():
  def __init__(self, modelname, use_peft=True, lora_r=16, lora_alpha=32, lora_model_path = None, device=None, tokenizername = None, optimized=False):
    self.use_peft = use_peft
    if use_peft:
      self.peft_config = LoraConfig(
        r = lora_r,
        lora_alpha = lora_alpha,
        bias = 'none',
        task_type = "SEQ_CLS",
      )
    else:
      self.peft_config = None
    self.modelname = modelname
    if optimized:
      # bfloat16
      basemodel = AutoModelForSequenceClassification.from_pretrained(modelname, num_labels=1, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16)
    else:
      basemodel = AutoModelForSequenceClassification.from_pretrained(modelname, num_labels=1)
    self.model = get_peft_model(basemodel, self.peft_config)
    if lora_model_path:
      if ".pt" in lora_model_path:
        # load the lora model from the path
        print("here the model should be loaded")
        lora_state_dict = torch.load(lora_model_path)
        new_state_dict = {}
        for key, value in lora_state_dict.items():
          if 'lora_A.weight' in key:
            new_key = key.replace('lora_A.weight', 'lora_A.default.weight')
            new_state_dict[new_key] = value
          elif 'lora_B.weight' in key:
            new_key = key.replace('lora_B.weight', 'lora_B.default.weight')
            new_state_dict[new_key] = value
          elif key == "base_model.model.score.weight":
            # Handle the unexpected key
            new_key = "base_model.model.score.original_module.weight"
            new_state_dict[new_key] = value
            new_key = "base_model.model.score.modules_to_save.default.weight"
            new_state_dict[new_key] = value
          else:
            new_state_dict[key] = value
        self.model.load_state_dict(new_state_dict,strict=False)
      else:
        self.peft_config = PeftConfig.from_pretrained(lora_model_path)
        self.model = PeftModel.from_pretrained(basemodel, lora_model_path)
    self.model.print_trainable_parameters()
    if tokenizername:
      self.tokenizer = AutoTokenizer.from_pretrained(tokenizername)
    else:
      self.tokenizer = AutoTokenizer.from_pretrained(modelname)
    self.tokenizer.pad_token = self.tokenizer.eos_token
    self.device = device
    if device:
      self.model.to(device)

  def get_reward(self, texts):
    with torch.no_grad():
      self.model.eval()
      inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(self.device)
      outputs = self.model(**inputs)
      logits = outputs.logits
      return logits

  def flatten_data(self, dataset):
    flattened_data = []
    flattened_labels = []

    for i in range(len(dataset)):
      if i % int(len(dataset)/20)==0:
        # print two decimal percentage
        print(f"flattening {round(i/len(dataset)*100, 2)}% done...")
      if len(dataset[i]["filtered_comment_texts"]) != len(dataset[i]["filtered_transformed_scores"]):
        assert False, "The length of filtered_comment_texts and filtered_transformed_score should be the same"
      for j in range(len(dataset[i]["filtered_comment_texts"])):
        flattened_data.append(f"<prompt>{dataset[i]['post_title']}\n{dataset[i]['post_text']}</prompt><response>{dataset[i]['filtered_comment_texts'][j]}</response>")
        flattened_labels.append(dataset[i]["filtered_transformed_scores"][j])

    # turn data and labels into Dataset object
    flattened = FlattenedDataset(flattened_data, flattened_labels)

    return flattened
  
  def load_dataset(self, dataset, path):
    try: 
      flattened = FlattenedDataset.load_from_disk(path)
    except:
      if dataset:
        flattened = self.flatten_data(dataset)
        flattened.save_to_disk(path)
      else:
        print("Please provide a dataset to train the model")
        return
    return flattened

  def train(self, 
    dataset=None, 
    val_dataset=None, 
    dataset_path="data/writingPrompt_flattened_train", 
    val_dataset_path="data/writingPrompt_flattened_val", 
    epoch = 3, 
    batch_size = 4, 
    eval_batch_size = 8, 
    lr=1e-5, 
    accelerate_config=None, 
    seed = 42,
    val_steps = 5000,
    ):
    # prepare data...
    flattened_train = self.load_dataset(dataset, dataset_path)
    flattened_val = self.load_dataset(val_dataset, val_dataset_path)
    
    
    print(flattened_train, flattened_val)
    torch.manual_seed(seed)
    train_dataloader = torch.utils.data.DataLoader(flattened_train, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(flattened_val, batch_size=eval_batch_size, shuffle=True)

    

    # set optimizer, scheduler, and accelerator
    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
    self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=2, verbose=True)
    # self.criterion = torch.nn.MSELoss()
    self.criterion = torch.nn.L1Loss()
    self.early_stopper = EarlyStopping(patience=3, verbose=True)
    if accelerate_config:
      self.accelerator = Accelerator(log_with="wandb")
      self.accelerator.init_trackers(
        project_name = accelerate_config["project_name"],
        config = accelerate_config["config"],
      )

      device = self.accelerator.device
      self.model, self.optimizer, self.scheduler, train_dataloader, val_dataloader= self.accelerator.prepare(self.model, self.optimizer, self.scheduler, train_dataloader, val_dataloader)
    else:
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      self.model.to(device)


    # do fine-tuning
    step = 0
    for eidx in range(epoch):
      for i, (batch_data, batch_labels) in enumerate(tqdm(train_dataloader)):
        step = step + 1

        inputs = self.tokenizer(batch_data, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)
        outputs = self.model(**inputs)
        loss = self.criterion(outputs.logits, batch_labels.unsqueeze(1).to(device))
        if accelerate_config:
          self.accelerator.backward(loss)
        else:
          loss.backward()
        self.optimizer.step()
        # self.scheduler.step(loss)
        self.optimizer.zero_grad()

        stat = {}
        if accelerate_config:
          loss = self.accelerator.gather(loss)
          stat["train_loss"] = torch.mean(loss).cpu().item()
          # self.accelerator.log({"train_loss": torch.mean(loss).cpu().item()})
        else:
          print(f"epoch {eidx}, batch {i}, loss: {loss}")
        

        if step % val_steps == 0:
          val_loss = self.evaluate(val_dataloader, device, accelerate_config)
          stat["val_loss"] = val_loss
          self.scheduler.step(val_loss)
          self.save(int(step/val_steps))
          self.early_stopper(val_loss, int(step/val_steps))
        self.accelerator.log(stat)
      
        if self.early_stopper.early_stop and epoch > 1:
          print("Early stopping")
          return



        
        # loss.backward()

      


  def evaluate(self, dataloader, device, accelerate_config=None):
    with torch.no_grad():
      self.model.eval()
      losses = []
      for i, (batch_data, batch_labels) in enumerate(tqdm(dataloader)):
        inputs = self.tokenizer(batch_data, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)
        outputs = self.model(**inputs)
        loss = self.criterion(outputs.logits, batch_labels.unsqueeze(1).to(device))
        losses.append(loss.item())

      total_loss = None
      if accelerate_config:
        # collect losses from all devices
        losses = accelerate.utils.gather_object(losses)
        total_loss = np.mean(losses)
      else:
        total_loss = np.mean(losses)
      
      if total_loss:
        print(f"Validation loss: {total_loss}")
    self.model.train()
    return total_loss

  def save(self, epoch=0, path="./checkpoints/reward_models"):
    # if the path does not exist, create it
    if not os.path.exists(path):
      os.makedirs(path)
    if self.use_peft:
      lora_state_dict = get_peft_model_state_dict(self.accelerator.unwrap_model(self.model))
      self.accelerator.save(lora_state_dict, f"{path}/{self.modelname.replace('/', '_')}_{epoch}.pt")
    else:
      self.model.save_pretrained(f"{path}/{self.modelname.replace('/', '_')}_{epoch}")
