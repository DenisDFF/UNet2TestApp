import torch
from diffusers import DDPMPipeline
from PIL import Image
import os

model_path = './output/unet/'

pipeline = DDPMPipeline.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipeline.to(device)

prompt = "butterfly on a flower" 

with torch.no_grad():
    image = pipeline(prompt).images[0]

image.save("generated_image.png")

print("Изображение успешно сохранено как generated_image.png")