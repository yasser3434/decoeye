# import replicate
import os
import replicate
from dotenv import load_dotenv

load_dotenv()
# REPLICATE_API_KEY = os.getenv("REPLICATE_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")
HF_REPOSITORY_ID = os.getenv("HF_REPOSITORY_ID")
TRIGGER_WORD = os.getenv("TRIGGER_WORD")

# replicate.Client(api_token=REPLICATE_API_KEY)

training = replicate.trainings.create(
  destination="yasser3434/decoeye-v1",
  version="ostris/flux-dev-lora-trainer:b6af14222e6bd9be257cbc1ea4afda3cd0503e1133083b9d1de0364d8568e6ef",
  input={
    "steps": 1000,
    "hf_token": HF_API_KEY,
    "lora_rank": 16,
    "optimizer": "adamw8bit",
    "batch_size": 1,
    "hf_repo_id": HF_REPOSITORY_ID,
    "resolution": "512,768,1024",
    "autocaption": True,
    "input_images": "https://example.com/couch_dataset.zip",
    "trigger_word": TRIGGER_WORD,
    "learning_rate": 0.0004,
    "wandb_project": "flux_train_replicate",
    "wandb_save_interval": 100,
    "caption_dropout_rate": 0.05,
    "cache_latents_to_disk": False,
    "wandb_sample_interval": 100,
    "gradient_checkpointing": False
  },
)

print("Training started! ID:", training.id)