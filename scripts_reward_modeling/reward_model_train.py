from models import CreativeWritingRewardModel
import datasets

# load the cleaned dataset
ds = datasets.load_from_disk("data/writingPrompt_cleaned")

reward_model = CreativeWritingRewardModel("google/gemma-2-2b")
reward_model.train(ds["train"], ds["test"], epoch=3, batch_size=4, lr=3e-5,
  accelerate_config={
    "project_name": "CWTuning_Reward_L1",
    "config": {},
  }
)
