import torch
from diffusers import UNet2DModel
from PIL import Image

# Указываем путь к локальному файлу с весами
model_path = './new_model/model_epoch_5.pth'

# Загрузка модели
model = UNet2DModel(
    sample_size=128,  # Например, если изображение 512x512
    in_channels=3,    # Входные каналы (например, для латентного пространства)
    out_channels=3,   # Выходные каналы
    layers_per_block=2,  # Настройки блоков
    block_out_channels=(128, 256, 512, 512),  # Количество каналов в каждом блоке
    down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D"),
    up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D")
)

# Загружаем веса модели
model.load_state_dict(torch.load(model_path))

# Переводим модель на нужное устройство
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Генерация изображения (пример)
def generate_image(model, prompt, device):
    # Здесь вы можете добавить свой код для преобразования текста в эмбеддинг и генерации изображения
    # В этом примере просто создадим случайное изображение для демонстрации
    z = torch.randn((1, 3, 512, 512)).to(device)
    timesteps = torch.randint(0, 1000, (1,), device=device)

    # Применяем модель
    with torch.no_grad():
        output = model(z, timesteps).sample

    # Преобразуем результат в изображение
    generated_image = output[0].cpu().detach().numpy().transpose(1, 2, 0)
    generated_image = ((generated_image + 1.0) * 127.5).astype("uint8")

    # Сохраняем изображение
    image = Image.fromarray(generated_image)
    image.save("generated_image.png")
    image.show()

# Пример использования
generate_image(model, "a photo of an astronaut riding a horse on mars", device)