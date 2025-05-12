from datasets import load_from_disk
from trl import DPOConfig, DPOTrainer
from models import GenerationModel, copy_peft_adapter

from argparse import ArgumentParser

from accelerate import Accelerator

parser = ArgumentParser()

# for llama
parser.add_argument("--modelname", default="meta-llama/Llama-3.1-8B", type=str, help="Model to use for training")
parser.add_argument("--init_lora_model_path", default="checkpoints/Llama-3.1-8B/generation_models_SFT/meta-llama_Llama-3.1-8B_epoch_final.pt", type=str, help="Path to the model")

parser.add_argument("--output_dir", default="checkpoints/Llama-3.1-8B/generation_models_DPO", type=str, help="Path to the model")

parser.add_argument("--dataset", default="data/writingPrompt_post_pair", type=str, help="Path to the dataset")

parser.add_argument("--learning_rate", default=5e-6, type=float, help="Learning rate for the model")
parser.add_argument("--num_train_epochs", default=3, type=int, help="Number of training epochs")

parser.add_argument("--batch_size", default=2, type=int, help="Batch size for training")

args = parser.parse_args()

generation_model = GenerationModel(args.modelname, lora_r = 128,
  lora_alpha = 256, lora_model_path = args.init_lora_model_path,
)

# added for fix - copy the peft adapter
copy_peft_adapter(generation_model.model, "original")
generation_model.model.set_adapter("default")


dataset = load_from_disk(args.dataset)
train_dataset = dataset["train"]
test_dataset = dataset["test"]

accelerator = Accelerator(log_with="wandb")
accelerator.init_trackers(
  project_name = "CWTuning_DPO",
)

training_args = DPOConfig(
  output_dir=args.output_dir, 
  logging_steps=10, 
  learning_rate=args.learning_rate, 
  num_train_epochs=args.num_train_epochs, 
  per_device_train_batch_size=args.batch_size,
  per_device_eval_batch_size=args.batch_size,
  report_to = "wandb",
  bf16=True,
  bf16_full_eval=True,
  eval_strategy = "epoch",
  save_strategy = "epoch",
  beta = 0.1, 
  gradient_accumulation_steps = 1,

  # added for fix 
  ref_adapter_name = "original",
)
trainer = DPOTrainer(
  model=generation_model.model, 
  args=training_args, 
  processing_class=generation_model.tokenizer, 
  train_dataset=train_dataset,
  eval_dataset=test_dataset,
  
)

print(type(trainer.model))
trainer.train()

# save the model
trainer.save_model(args.output_dir+"/checkpoint-final")