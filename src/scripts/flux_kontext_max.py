import replicate
import time
from datetime import datetime, date
from dotenv import load_dotenv
import os

load_dotenv()

start_time = time.time()
actual_date = date.today()
current_time = datetime.now().strftime("%H-%M-%S")

# ========================= Creating directory
exists = os.path.exists(f"../../data/generated_images/vide/flux_kontext_max/{actual_date}")

if not exists:
    os.makedirs(f"data/generated_images/vide/flux_kontext_max/{actual_date}")
    print("Daily generations directory created!")

print("Generating new image...")


# ========================= Opening input image
image = open("data/input_images/vide_2.jpg", "rb")


# ========================= Model input/ output
input = {
    "prompt": "A modern style decorated living room, 4k photo, highly detailed and realistic",
    "input_image": image
}

output = replicate.run(
    "black-forest-labs/flux-kontext-max",
    input=input
)


# ========================= Save new generated image
with open(f'data/generated_images/vide/flux_kontext_max/{actual_date}/output_{current_time}.png', 'wb') as f:
    f.write(output.read())


# ========================= Calculating execution time
end_time = time.time()  
print(f"Image {output} \nGenerated in {end_time - start_time :.2f} seconds")