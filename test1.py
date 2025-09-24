from PIL import Image
import requests
from io import BytesIO
from src.models.SigLIP2Module import SigLIP2Module
import torch

# Initialize
device = "cuda" if torch.cuda.is_available() else "cpu"
siglip = SigLIP2Module(device=device)

# Download the image
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
response = requests.get(url)
img = Image.open(BytesIO(response.content)).convert("RGB")

# Forward pass
embeddings = siglip.forward(img)

# Check output
if isinstance(embeddings, tuple):
    print("Image embeddings:", embeddings[0].shape)
    print("Text embeddings:", embeddings[1].shape)
else:
    print("Image-only embeddings:", embeddings.shape)
