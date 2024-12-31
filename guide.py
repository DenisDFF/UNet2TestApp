import os
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from torchvision import transforms
from accelerate import Accelerator
from huggingface_hub import create_repo, upload_folder
from tqdm.auto import tqdm
import torch.nn.functional as F
from pathlib import Path
from diffusers import DDPMPipeline

# Конфигурация
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
    dataset_name = "huggan/smithsonian_butterflies_subset"  # Имя датасета

config = TrainingConfig()

dataset = load_dataset(config.dataset_name, split="train")

preprocess = transforms.Compose(
    [
        transforms.Resize((config.image_size, config.image_size)),  
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize([0.5], [0.5]),  
    ]
)

def transform(examples):
    images = [preprocess(image.convert("RGB")) for image in examples["image"]]
    return {"images": images}

dataset.set_transform(transform)

train_dataloader = DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)

def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        if config.push_to_hub:
            repo_id = create_repo(
                repo_id=config.hub_model_id or Path(config.output_dir).name, exist_ok=True
            ).repo_id
        accelerator.init_trackers("train_example")

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0

    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch["images"]
            noise = torch.randn(clean_images.shape, device=clean_images.device)
            bs = clean_images.shape[0]

            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device,
                dtype=torch.int64
            )

            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        if accelerator.is_main_process:
            pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(config, epoch, pipeline)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                if config.push_to_hub:
                    upload_folder(
                        repo_id=repo_id,
                        folder_path=config.output_dir,
                        commit_message=f"Epoch {epoch}",
                        ignore_patterns=["step_*", "epoch_*"],
                    )
                else:
                    pipeline.save_pretrained(config.output_dir)

def evaluate(config, epoch, pipeline):
    pass

if __name__ == "__main__":
    model = None  
    noise_scheduler = None 
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    lr_scheduler = None

    train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)