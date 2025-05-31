import os
import replicate
import time
from dotenv import load_dotenv
from datetime import datetime, date
# from huggingface_hub import login

load_dotenv()

# token = os.getenv("HUGGINGFACE_TOKEN")
# login(token=token)

start_time = time.time()
actual_date = date.today()
current_time = datetime.now().strftime("%H-%M-%S")
exists = os.path.exists(f"generated_images/vide/{actual_date}")
if not exists:
    os.makedirs(f"generated_images/vide/{actual_date}")
    print("Daily generations directory created!")
    
image_to_image_strength = 0.125

# image = open("salon_meuble.jpg", "rb")
image = open("vide_4.jpg", "rb")

print("Generating new image...")

output = replicate.run(
    "xlabs-ai/flux-dev-controlnet:9a8db105db745f8b11ad3afe5c8bd892428b2a43ade0b67edc4e0ccd52ff2fda",
    input={
        "steps": 28,
        # "prompt": "A scandinavian decorated living room with a green couch grncch1, 4k photo, highly detailed",
        # "prompt": "A scandinavian style decorated living room, 4k photo, highly detailed",
        "prompt": '''A modern style decorated living room, 4k photo, highly detailed and realistic
                    ,respect to shape and the dimensions of the room''',
        # "lora_url": "https://huggingface.co/yasser34/decoeye-hf-v1/resolve/main/lora.safetensors",
        "control_type": "depth",
        # "control_image": "https://i.pinimg.com/736x/2c/3f/24/2c3f24f974616cf79e4371667676d103.jpg",
        "control_image": image,
        "lora_strength": 1,
        "output_format": "jpg",
        "guidance_scale": 2.5,
        "output_quality": 100,
        "negative_prompt": '''low quality, ugly, distorted, artefacts, text, 
                            abstract, glitch, deformed, mutated, disfigured''',
        "control_strength": 0.45,
        "depth_preprocessor": "DepthAnything",
        "soft_edge_preprocessor": "HED",
        "image_to_image_strength": image_to_image_strength,
        "return_preprocessed_image": False
    }
)

# with open(f'generated_images/meuble/output_{image_to_image_strength}_strength_{current_time}.jpg', 'wb') as f:
with open(f'generated_images/vide/{actual_date}/output_{image_to_image_strength}_strength_{current_time}.jpg', 'wb') as f:
    f.write(output[0].read())
# print(output)

end_time = time.time()  

print(f"Image {output} \nGenerated in {end_time - start_time :.2f} seconds")