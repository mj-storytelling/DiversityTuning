
import torch
# open a model from checkpoint

from peft import PeftModel, get_peft_model_state_dict
from transformers import AutoModelForCausalLM

base_model_name =  "meta-llama/Llama-3.1-8B"  #path/to/your/model/or/name/on/hub"
adapter_model_name = "checkpoints/Llama-3.1-8B/generation_models_DDPO_sem_sty_max4_highscore_ver2_fixed_real/checkpoint-final"

model = AutoModelForCausalLM.from_pretrained(base_model_name).to("cuda:0")
model = PeftModel.from_pretrained(model, adapter_model_name)
# print lora config
print(model.peft_config)

path = "./checkpoints/Llama-3.1-8B/generation_models_DDPO_sem_sty_max4_highscore_ver2_fixed_real"

# save the model
lora_state_dict = get_peft_model_state_dict(model)
torch.save(lora_state_dict, f"{path}/ddpo_00.pt")