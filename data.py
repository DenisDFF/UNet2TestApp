import os
import json
from PIL import Image
from torchvision import transforms
from datasets import Dataset

def load_and_preprocess_data(config):
    image_folder = os.path.join(config.data_dir, 'data') 
    json_file = os.path.join(config.data_dir, 'config.json') 

    # Загружаем описание из config.json
    with open(json_file, 'r') as f:
        descriptions = json.load(f)

    preprocess = transforms.Compose(
        [
            transforms.Resize((config.image_size, config.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def transform(examples):
        images = []
        texts = []

        for image_name in examples['image_names']:
            img_path = os.path.join(image_folder, image_name)
            img = Image.open(img_path).convert("RGB")
            img = preprocess(img)
            
            description = descriptions.get(image_name, "")
            images.append(img)
            texts.append(description)

        return {"images": images, "descriptions": texts}

    image_names = list(descriptions.keys())

    dataset = Dataset.from_dict({"image_names": image_names})

    dataset.set_transform(transform)

    return dataset