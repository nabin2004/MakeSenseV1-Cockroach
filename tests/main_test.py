from src.models.SigLIP2Module import SigLIP2Module
from PIL import Image
import time
import torch

model = SigLIP2Module()

image = Image.open("sample_doc.png")

start = time.time()
emb1 = model.forward(image, text="a sample document page", use_cache=False)
print("First run (no cache):", time.time() - start, "seconds")

start = time.time()
emb2 = model.forward(image, text="a sample document page", use_cache=True)
print("Second run (with cache):", time.time() - start, "seconds")
print("Cache size:", len(model.cache))

# cached result == freshly computed
emb3 = model.forward(image, text="a sample document page", use_cache=False)
print("Embeddings identical:", torch.allclose(emb2[0], emb3[0], atol=1e-6))