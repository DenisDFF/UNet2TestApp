from accelerate import Accelerator
from huggingface_hub import create_repo, upload_folder
from tqdm.auto import tqdm
from pathlib import Path
import torch
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from diffusers import DDPMPipeline, UNet2DModel, DDPMScheduler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import Image

class TrainingConfig:
    image_size = 512  # Размер изображения
    train_batch_size = 6  # Размер батча для обучения
    eval_batch_size = 6  # Размер батча для оценки
    num_epochs = 5  # Количество эпох
    gradient_accumulation_steps = 1  # Количество шагов для накопления градиентов
    learning_rate = 1e-4  # Скорость обучения
    lr_warmup_steps = 500  # Количество шагов разогрева
    save_image_epochs = 1  # Сохранять изображения каждую эпоху
    save_model_epochs = 1  # Сохранять модель каждую эпоху
    mixed_precision = "fp16"  # Использование смешанной точности
    output_dir = "./generated_model"  # Папка для сохранения модели
    push_to_hub = False  # Не загружать на Hugging Face Hub
    hub_model_id = None
    hub_private_repo = None
    overwrite_output_dir = True  # Перезаписывать папку
    seed = 0  # Начальное значение для генератора случайных чисел

config = TrainingConfig()

transform = transforms.Compose([
    transforms.Resize((config.image_size, config.image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

dataset = ImageFolder(root='./data', transform=transform)

train_loader = DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
model = UNet2DModel(
    sample_size=config.image_size,
    in_channels=3,  
    out_channels=3, 
    layers_per_block=2,
    block_out_channels=(128, 256, 512, 512),
    down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D"),
    up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
).to(device)  

optimizer = AdamW(model.parameters(), lr=config.learning_rate)

noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

def evaluate(config, epoch, model):
    sample_image, _ = next(iter(train_loader)) 
    sample_image = sample_image.to(device) 

    noise = torch.randn(sample_image.shape, device=device, requires_grad=True)
    timesteps = torch.LongTensor([50]).to(device)
    noisy_image = noise_scheduler.add_noise(sample_image, noise, timesteps)

    loss = F.mse_loss(noisy_image, sample_image)
    loss.backward()

    generated_image = noisy_image[0].cpu().detach().numpy().transpose(1, 2, 0)
    save_path = f"./generated_images/{epoch:04d}_generated_image.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    Image.fromarray(((generated_image + 1.0) * 127.5).astype("uint8")).save(save_path)
    print(f"Image saved to {save_path}")

def train_loop(config, model, noise_scheduler, optimizer, train_dataloader):
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps
    )
    if accelerator.is_main_process:
        os.makedirs(config.output_dir, exist_ok=True)

    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    global_step = 0

    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images, _ = batch  
            clean_images = clean_images.to(device)

            noise = torch.randn(clean_images.shape, device=device)
            bs = clean_images.shape[0]

            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=device)

            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
 
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        if accelerator.is_main_process:
            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(config, epoch, model)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                model_save_path = os.path.join(config.output_dir, f"model_epoch_{epoch+1}.pth")
                torch.save(model.state_dict(), model_save_path)
                print(f"Model saved to {model_save_path}")

train_loop(config, model, noise_scheduler, optimizer, train_loader)