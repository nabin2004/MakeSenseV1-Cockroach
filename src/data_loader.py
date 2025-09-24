# src/data_loader.py
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
from transformers import AutoTokenizer

class DocDataset(Dataset):
    def __init__(self, data_path, image_paths=None, text_paths=None, annotation_paths=None, tokenizer=None, image_transform=None):
        self.image_paths = image_paths
        self.text_paths = text_paths
        self.annotation_paths = annotation_paths
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = self.image_paths[idx]
        text = self.text_paths[idx]
        annotations = self.annotation_paths[idx]

        if self.image_transform:
            image = self.image_transform(image)

        return {
            "image": image,
            "text": text,
            "annotations": annotations
        }



# dataset = DocDataset(
#     image_paths=["data/images/doc1.png", "data/images/doc2.png"],
#     text_paths=["data/texts/doc1.txt", "data/texts/doc2.txt"],
#     annotation_paths=["data/annotations/doc1.json", "data/annotations/doc2.json"],
#     tokenizer=tokenizer,
#     image_transform=image_transform
# )

# dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

