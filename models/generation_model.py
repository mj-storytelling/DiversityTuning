from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from accelerate import Accelerator
import accelerate
from .utils import FlattenedDataset, copy_peft_adapter
import os
import math
from trl import AutoModelForCausalLMWithValueHead

class GenerationModel():
  def __init__(self, modelname, use_peft=True, lora_r=16, lora_alpha=32, lora_model_path = None, device=None, optimized=False,
    is_ppo=False, save_path="./checkpoints/generation_models"
  ):
    self.save_path = save_path
    self.use_peft = use_peft
    if use_peft:
      self.peft_config = LoraConfig(
        r = lora_r,
        lora_alpha = lora_alpha,
        bias = 'none',
        task_type = "CAUSAL_LM",
      )
    else:
      self.peft_config = None
    self.modelname = modelname
    self.optimized = optimized
    if optimized:
      self.model = AutoModelForCausalLM.from_pretrained(modelname, torch_dtype=torch.bfloat16, 
        attn_implementation="flash_attention_2") #, peft_config=self.peft_config)
    else:
      if not is_ppo:
        self.model = AutoModelForCausalLM.from_pretrained(modelname)
    if self.use_peft:
      if not is_ppo:
        self.model = get_peft_model(self.model, self.peft_config)
      else:
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(modelname, peft_config=self.peft_config,)
    if lora_model_path:
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
        # elif key == "base_model.model.score.weight":
        #   # Handle the unexpected key
        #   new_key = "base_model.model.score.original_module.weight"
        #   new_state_dict[new_key] = value
        #   new_key = "base_model.model.score.modules_to_save.default.weight"
        #   new_state_dict[new_key] = value
        else:
          new_state_dict[key] = value

      del lora_state_dict
          
      if not is_ppo:
        self.model.load_state_dict(new_state_dict, strict=False)
      else:
        self.model.pretrained_model.load_state_dict(new_state_dict, strict=False)
        copy_peft_adapter(self.model.pretrained_model, "original")
        self.model.pretrained_model.set_adapter("default")

    if self.use_peft and not is_ppo:
      self.model.print_trainable_parameters()
    if "meta-llama/Llama-3.1-8B" in modelname:
      self.tokenizer = AutoTokenizer.from_pretrained(f"meta-llama/Llama-3.1-8B-Instruct")
      self.eos_token_ids = [128000+ i for i in range(256)]
    elif "google/gemma-2-2b" in modelname:
      self.tokenizer = AutoTokenizer.from_pretrained(f"google/gemma-2-2b-it")
      self.eos_token_ids = [self.tokenizer.eos_token_id]
      # self.eos_token_ids = [128000+ i for i in range(256)]
    elif "mistralai/Mistral-7B-v0.3" in modelname or "mistralai/Mistral-7B-Instruct-v0.3" in modelname:
      self.tokenizer = AutoTokenizer.from_pretrained(f"mistralai/Mistral-7B-Instruct-v0.3")
      self.eos_token_ids = [self.tokenizer.eos_token_id]

    self.tokenizer.pad_token = self.tokenizer.eos_token
    self.device = device
    if device:
      self.model.to(device)
    

  def generate(self, 
      prompts, 
      max_length=2000, 
      num_return_sequences=1, 
      temperature=1.0, 
      top_k=50, 
      top_p=0.0, 
      repetition_penalty=1.0, 
      do_sample=True, 
      pad_token_id=None, 
      skip_special_tokens = True,
      developer_prompt = None,
    ):
    with torch.no_grad():
      templated_prompts = []
      for prompt in prompts:
        if developer_prompt:
          message = [
            {"role": "system", "content": developer_prompt},
            {"role": "user", "content": prompt}
          ]
        else:
          message = [
            {"role": "user", "content": prompt}
          ]
        message_templated = self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        templated_prompts.append(message_templated)

      pad_token_id = self.tokenizer.eos_token_id

      tokenized_input = self.tokenizer(templated_prompts, return_tensors='pt', max_length=max_length, truncation=True, ).to(self.device)
      # print(tokenized_input, templated_prompts)
      # print(self.tokenizer.batch_decode(tokenized_input.input_ids))
      if not self.optimized:
        output = self.model.generate(tokenized_input.input_ids, 
          attention_mask=tokenized_input.attention_mask,
          max_length=max_length + tokenized_input.input_ids.size(1), 
          num_return_sequences=num_return_sequences, 
          temperature=temperature, top_k=top_k, top_p=top_p, 
          do_sample=do_sample, repetition_penalty=repetition_penalty, pad_token_id=pad_token_id,
          eos_token_id=self.eos_token_ids
        )
      else:
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
          output = self.model.generate(tokenized_input.input_ids, 
            attention_mask=tokenized_input.attention_mask,
            max_length=max_length + tokenized_input.input_ids.size(1), 
            num_return_sequences=num_return_sequences, 
            temperature=temperature, top_k=top_k, top_p=top_p, 
            do_sample=do_sample, repetition_penalty=repetition_penalty, pad_token_id=pad_token_id,
            eos_token_id=self.eos_token_ids
          )
      # remove the input prompt
      if len(prompts) == 1:
        output = output[:, tokenized_input.input_ids.size(1):]
      else:
        for i in range(len(prompts)):
          output[i] = output[i, input_ids[i].size(1):]
    return self.tokenizer.batch_decode(output, skip_special_tokens=skip_special_tokens)

  def weighted_causal_lm_loss(self, logits, labels, weights, num_items_in_batch, ignore_index=-100, **kwargs):
    # TODO implement weighted causal lm loss
    logits = logits.float()
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    vocab_size = shift_logits.size(-1)
    seq_len = shift_logits.size(1)

    # Flatten the tokens
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)

    loss = nn.functional.cross_entropy(shift_logits, shift_labels, ignore_index=ignore_index, reduction='none')
    weight_tensor = []
    for i in range(len(weights)):
      weight_tensor += [weights[i]]*seq_len
    weight_tensor = torch.tensor(weight_tensor).to(shift_logits.device)
    loss = loss * weight_tensor.view(-1)
    return loss.mean()*num_items_in_batch

  def load_dataset(self, dataset, path, score="default"):
    try:
      flattened = FlattenedDataset.load_from_disk(path)
    except:
      if dataset:
        flattened = self.flatten_data(dataset, score)
        flattened.save_to_disk(path)
      else:
        print("Please provide a dataset to train the model")
        return
    return flattened

  def flatten_data(self, dataset, score="default"):
    flattened_data = []
    flattened_labels = []

    # tokenized_lengths = []

    for i in range(len(dataset)):
      if i % int(len(dataset)/20)==0:
        # print two decimal percentage
        print(f"flattening {round(i/len(dataset)*100, 2)}% done...")
        # if i!=0:
        #   print(f"Max tokenized length: {np.max(tokenized_lengths)}")
      for j in range(len(dataset[i]["filtered_comment_texts"])):
        message = [
          {"role": "user", "content": f"{dataset[i]['post_title']}\n{dataset[i]['post_text']}"},
          {"role": "assistant", "content": dataset[i]["filtered_comment_texts"][j]}
        ]
        message_templated = self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False)
        # print(message_templated)
        # tokenized = self.tokenizer(message_templated, return_tensors="pt", padding=True, truncation=False)
        # tokenized_lengths.append(tokenized.input_ids.size(1))
        flattened_data.append(message_templated)
        if score == "diversity":
          pass
        else:
          flattened_labels.append(dataset[i]["filtered_transformed_scores"][j])


    # turn data and labels into Dataset object
    flattened = FlattenedDataset(flattened_data, flattened_labels)
    # print(f"Average tokenized length: {np.mean(tokenized_lengths)}")
    # print(f"Max tokenized length: {np.max(tokenized_lengths)}")
    return flattened


  def train_with_finetuning(self, 
      dataset, 
      val_dataset,
      diversity_weighting=False,
      epoch=3, 
      batch_size = 2,
      eval_batch_size=4, 
      lr=1e-5, 
      accelerate_config=None, 
      seed=42, 
      val_steps=5000,
      start_step = 0,
      score_type = "default"
    ):
    torch.manual_seed(seed)
    flattened_train = self.load_dataset(dataset, f"data/writingPrompt_flattened_gen_train_{score_type}", score=score_type)
    flattened_val = self.load_dataset(val_dataset, f"data/writingPrompt_flattened_gen_val_{score_type}", score=score_type)

    # return 

    train_dataloader = torch.utils.data.DataLoader(flattened_train, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(flattened_val, batch_size=eval_batch_size, shuffle=True)

    # set optimizer, scheduler, and accelerator
    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR

    def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
        """
        Cosine schedule with warmup, commonly used in language model fine-tuning.
        """
        def lr_lambda(current_step):
            # Warmup
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            # Cosine decay
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

        return LambdaLR(optimizer, lr_lambda, last_epoch)

    self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=500, num_training_steps=epoch * len(train_dataloader), num_cycles=0.25)
    # # no lr reduction
    # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=500, verbose=True)
    
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
        step=step+1
        if step - 1 < start_step:
          if step % val_steps == 0:
            print(f"Skipping step {step}...")
          continue
        # do not add bos token
        inputs = self.tokenizer(batch_data, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)
        outputs = self.model(**inputs, labels=inputs.input_ids)
        if diversity_weighting == False:
          loss = outputs.loss
        else:
          # TODO diversity weighting
          logits = outputs.logits
          # more specifically, get labels and weights...

          pass
        if accelerate_config:
          self.accelerator.backward(loss)
        else:
          loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        stat = {}
        if accelerate_config:
          loss = self.accelerator.gather(loss)
          stat["train_loss"] = torch.mean(loss).cpu().item()
        else:
          print(f"epoch {eidx}, batch {i}, loss: {np.mean(train_losses)}")


        if step % val_steps == 0:
          val_loss = 0
          val_loss = self.evaluate(val_dataloader, device, accelerate_config)
          stat["val_loss"] = val_loss
          self.save(int(step/val_steps))

          
        self.accelerator.log(stat, step = step - 1)
    val_loss = self.evaluate(val_dataloader, device, accelerate_config)
    stat = {"val_loss": val_loss}
    self.accelerator.log(stat, step = step - 1)
    self.save("epoch_final")

  
  def evaluate(self, val_dataloader, device, accelerate_config):
    with torch.no_grad():
      self.model.eval()
      losses = []
      for i, (batch_data, batch_labels) in enumerate(tqdm(val_dataloader)):
        inputs = self.tokenizer(batch_data, return_tensors="pt", padding=True, truncation=True, max_length=2048, add_special_tokens=False).to(device)
        outputs = self.model(**inputs, labels=inputs.input_ids)

        losses.append(outputs.loss.item())
      
      total_loss = None
      if accelerate_config:
        losses = accelerate.utils.gather_object(losses)
        total_loss = np.mean(losses)
      else:
        total_loss = np.mean(losses)

      if total_loss:
        print(f"Validation loss: {total_loss}")
    self.model.train()
    return sum(losses)/len(losses)

  def save(self, epoch=0):
    if not os.path.exists(self.save_path):
      os.makedirs(self.save_path)
    if self.use_peft:
      lora_state_dict = get_peft_model_state_dict(self.accelerator.unwrap_model(self.model))
      self.accelerator.save(lora_state_dict, f"{self.save_path}/{self.modelname.replace('/', '_')}_{epoch}.pt")
    else:
      self.model.save_pretrained(f"{self.save_path}/{self.modelname.replace('/', '_')}_{epoch}.pt")
