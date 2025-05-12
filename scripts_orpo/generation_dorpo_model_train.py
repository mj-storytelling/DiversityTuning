from datasets import load_from_disk
from trl import ORPOConfig
from models import GenerationModel, DORPOTrainer

from argparse import ArgumentParser

from accelerate import Accelerator

parser = ArgumentParser()

parser.add_argument("--modelname", default="meta-llama/Llama-3.1-8B", type=str, help="Model to use for training")
parser.add_argument("--output_dir", default="checkpoints/Llama-3.1-8B/generation_models_DORPO_sem_sty", type=str, help="Path to the model")

parser.add_argument("--dataset", default="data/writingPrompt_post_pair_sem_sty", type=str, help="Path to the dataset")

parser.add_argument("--learning_rate", default=5e-6, type=float, help="Learning rate for the model")
parser.add_argument("--num_train_epochs", default=4, type=int, help="Number of training epochs")

parser.add_argument("--batch_size", default=2, type=int, help="Batch size for training")

args = parser.parse_args()

generation_model = GenerationModel(args.modelname, lora_r = 128,
  lora_alpha = 256, 
)

dataset = load_from_disk(args.dataset)
train_dataset = dataset["train"]
test_dataset = dataset["test"]

accelerator = Accelerator(log_with="wandb")
accelerator.init_trackers(
  project_name = "CWTuning_DORPO_mistral",
)

training_args = ORPOConfig(
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
  beta = 0.25, # parameter to care about
)
trainer = DORPOTrainer(
  model=generation_model.model, 
  args=training_args, 
  processing_class=generation_model.tokenizer, 
  train_dataset=train_dataset,
  eval_dataset=test_dataset,
)

trainer.train()

# save the model
trainer.save_model(args.output_dir+"/checkpoint-final")