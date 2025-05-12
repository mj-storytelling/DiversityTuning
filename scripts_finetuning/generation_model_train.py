from models import GenerationModel
import datasets

generation_model = GenerationModel(
  "meta-llama/Llama-3.1-8B", 
  lora_r = 128,
  lora_alpha = 256,
  save_path = "checkpoints/Llama-3.1-8B/generation_models_SFT",
)

ds = datasets.load_from_disk("data/writingPrompt_cleaned")

generation_model.train_with_finetuning(ds["train"], ds["test"], epoch=1, batch_size=2, 
  lr=3e-5, 
  accelerate_config={
    "project_name": "CWTuning_SFT",
    "config": {},
  }
)