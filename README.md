# Model Tuning for Creative Writing

Try run scripts like `python -m reward_model_scripts.data_prep` instead of doing `python reward_model_scripts/data_prep.py`... except they are within the shell command file or ran with accelerate. We are using module approach to run python codes.

We are using [`euclaise/WritingPrompts_preferences`](https://huggingface.co/datasets/euclaise/WritingPrompts_preferences) as the source dataset. 

## Data processing

1. By running `scripts_dataset_cleaner.0_data_prep_scoring`, we first clean the "prompts" (or posts) by removing phrases between `[]` parenthesis (these include some information irrelevant to writing prompts). Then, we transform the score so that it can be used for reward model training. 
- The input path is `--source` and the output path is `--data_path`.
2. By running `scripts_dataset_cleaner.1_data_prep_filtering`, we remove irrelevant "writings" (or comments) (e.g., moderation comments) from the dataset. We also filter out excessively long writings, as they can overload the training pipeline. 
- The input path is `--source` and the output path is `--data_path`. `--model_to_use` is the model used for deciding whether the writing is long or not. 
3. By running `scripts_dataset_cleaner.2_data_prep_to_preference`, we turn the dataset into a pairwise "preference dataset."
- The input path is `--source` and the output path is `--data_path_pair`. `--data_path_filtered` is an optional path you can use if you want to store the unpaired dataset with those instances used in the paired preference dataset. `--gap` decides how many votes gaps you would like to have between winning and losing instances.
4. By running `scripts_dataset_cleaner.3_data_prep_for_diversified_training`, we turn the dataset into a pairwise preference dataset with "deviation scores" attached. 
- The input path is `--source_pair` and the output path is `--data_path_pair_div`. `--score_process` indicates which type of diversity/deviation you will use. It should be a list including either "semantic" or "style". `--device` decides which device to use for calculating text embeddings.
- `--max_cutoff` decides the maximum number of instances per prompt, and when it is `-1`, all instances are used. `--min_score_non_zero` changes the deviation score below threshold (-0.8) to the threshold value. `--sort_with_score` sorts the instances according to quality score so that high quality winning instances can be picked first. These three parameters are used to run the ablation study.
5. By running `scripts_dataset_cleaner.4_data_prep_divpo`, we turn the dataset into pairs used for [DivPO](https://arxiv.org/abs/2501.18101).
- The input path is `--source` and the output path is `--data_path_pair`. `--score_process` indicates which type of diversity/deviation you will use. It should be a list including either "semantic" or "style". `--device` decides which device to use for calculating text embeddings.
- `--max_pair_num` decides the number of instances per prompt. `--rho` decides the percentage of high or low quality instances to be included in the training dataset.

## Reward model training

To train a reward model, you can run `scripts_reward_modeling.reward_model_train` - we recommend running it with accelerate. The shell command is in `cm_reward.sh`.
For the reward model, now we are tuning `gemma-2b` model with lora. You can evaluate this model with `scripts_reward_modeling.reward_model_perf_analysis`.

## Supervised finetuning

You can do supervised finetuning with `scripts_finetuning.generation_model_train`. You can find the shell command in `cm_generation.sh`.

## DPO and DDPO training

You can do DPO and DDPO training with `scripts_dpo.generation_dpo_model_train` and `scripts_dpo.generation_ddpo_model_train`. Note that DivPO training the DPO training code. Shell commands are `cm_dpo.sh` and `cm_ddpo.sh`, respectively.

## ORPO and DORPO training

Similarly, you can do ORPO and DORPO training with `scripts_orpo.generation_orpo_model_train` and `scripts_orpo.generation_dorpo_model_train`. Shell commands are `cm_orpo.sh` and `cm_dorpo.sh`.

## Evaluating trained models

You can run the evaluation on writing quality, semantic diversity, and style diversity with `scripts_eval.generation_eval1_1`. Once you have collected evaluation results from `scripts_eval.generation_eval1_1`, you can run `scripts_eval.generation_eval1_2` to get results on auxilliary metrics. You can also run `scripts_eval.generation_eval2` to get results on the gold data (run `scripts_eval.generation_eval1_2` afterward to get auxiliary metrics results, too).

For human evaluation, you can run `scripts_eval.prepare_human_eval_data` first to get the summarized versions of generated writings. Then, you can use this file to run annotation tasks on [Potato](https://github.com/davidjurgens/potato) (whose configuration code is under `human-eval`).

Model checkpoints coming soon..!

## Citing the work

    @misc{chung2025modifyinglargelanguagemodel,
      title={Modifying Large Language Model Post-Training for Diverse Creative Writing}, 
      author={John Joon Young Chung and Vishakh Padmakumar and Melissa Roemmele and Yuqian Sun and Max Kreminski},
      year={2025},
      eprint={2503.17126},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2503.17126}, 
}