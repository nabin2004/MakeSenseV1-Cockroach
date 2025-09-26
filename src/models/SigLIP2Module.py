import torch
from transformers import SiglipProcessor, SiglipModel
from PIL import Image
import torch.nn.functional as F
import hashlib
import io

class SigLIP2Module:
    def __init__(self, model_name="google/siglip-base-patch16-224", device=None):
        """Initialize the SigLIP2Module."""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = SiglipProcessor.from_pretrained(model_name)
        self.model = SiglipModel.from_pretrained(model_name).to(self.device)
        self.cache = {}  # cache dict

    def _make_cache_key(self, image, text=None):
        """Generate a hash key for caching based on image+text."""
        buf = io.BytesIO()
        if isinstance(image, Image.Image):
            image.save(buf, format="PNG")
            img_bytes = buf.getvalue()
        # else:
        #     raise ValueError("Expected PIL.Image.Image for caching")

        # hasher = hashlib.sha256(img_bytes)
        # if text:
        #     hasher.update(text.encode("utf-8"))
        # return hasher.hexdigest()
        
        return

    def forward(self, image, text: str = None, normalize=True, use_cache=True, return_hidden=False):
        """
        image: PIL image or list of images
        text: optional text (for cross-modal embeddings)
        returns: embeddings (cached if available)
        """
        if not isinstance(image, list):
            image = [image]  # wrap single image

        cache_key = None
        if use_cache and len(image) == 1:  # cache only for single images
            cache_key = self._make_cache_key(image[0], text)
            if cache_key in self.cache:
                return self.cache[cache_key]

        if text:
            if isinstance(text, str):
                text = [text] * len(image)  # repeat same text for batch
            inputs = self.processor(text=text, images=image, return_tensors="pt", padding=True).to(self.device)
            outputs = self.model(**inputs)
            image_embeds, text_embeds = outputs.image_embeds, outputs.text_embeds
            if normalize:
                image_embeds = F.normalize(image_embeds, p=2, dim=-1)
                text_embeds = F.normalize(text_embeds, p=2, dim=-1)
            result = (image_embeds, text_embeds)
        else:
            inputs = self.processor(images=image, return_tensors="pt", padding=True).to(self.device)
            outputs = self.model.get_image_features(**inputs)
            if normalize:
                outputs = F.normalize(outputs, p=2, dim=-1)
            result = outputs

        if cache_key:
            self.cache[cache_key] = result
        return result
