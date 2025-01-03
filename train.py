from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline, DDPMScheduler
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from tqdm.auto import tqdm
import os

# Инициализация модели и текстового энкодера
def create_model(config):
    # Загрузка модели и текстового энкодера
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
    pipe.to(config.device)
    
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch16")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")

    return pipe, text_encoder, tokenizer

# Преобразование текста в эмбеддинги
def encode_text(prompts, text_encoder, tokenizer):
    inputs = tokenizer(prompts, padding=True, return_tensors="pt", truncation=True)
    text_embeddings = text_encoder(**inputs).last_hidden_state
    return text_embeddings

# Цикл тренировки с обработкой текста
def train_loop(config, model, text_encoder, tokenizer, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0

    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch["images"]
            prompts = batch["prompts"]  # предполагаем, что в батче есть промты

            # Получаем эмбеддинги текста
            text_embeddings = encode_text(prompts, text_encoder, tokenizer)

            # Генерация шума
            noise = torch.randn(clean_images.shape, device=clean_images.device)
            bs = clean_images.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device, dtype=torch.int64)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # Прогнозируем остаток шума
                noise_pred = model(noisy_images, timesteps, encoder_hidden_states=text_embeddings, return_dict=False)[0]
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

        # Сохранение модели и образцов изображений после каждой эпохи
        if accelerator.is_main_process:
            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                model.save_pretrained(config.output_dir)