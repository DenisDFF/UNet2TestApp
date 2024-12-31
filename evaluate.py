from diffusers import DDPMPipeline
from diffusers.utils import make_image_grid
import os
from PIL import Image
import torch

def evaluate(config, epoch, pipeline):
    images = pipeline.generate_images(num_images=16)  
    
    print(f"Generated {len(images)} images during evaluation for epoch {epoch}.")