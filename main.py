from accelerate import notebook_launcher
from config import Config
from data import load_and_preprocess_data
from model import create_model
from optimizer import create_optimizer_and_scheduler
from diffusers import DDPMScheduler
import torch
from train import train_loop

# Initialize the configuration
config = Config()

# Load dataset and preprocess
train_dataset = load_and_preprocess_data(config)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True)

# Initialize model and noise scheduler
model = create_model(config)
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

# Create optimizer and scheduler
optimizer, lr_scheduler = create_optimizer_and_scheduler(model, config, train_dataloader)

# Launch the training loop
args = (config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)
notebook_launcher(train_loop, args, num_processes=1)