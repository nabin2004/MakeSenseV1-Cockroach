from src.models.SigLIP2Module import SigLIP2Module
from src.models.llm import LLMModule
from src.models.ner import NERModule
from src.models.re_module import REModule    

import torch.nn as nn
import torch


class MakeSense(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MakeSense, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vision_enc = SigLIP2Module().to(self.device)   # Vision encoder
        self.llm = LLMModule().to(self.device)              # LLM
        self.ner = NERModule().to(self.device)              # NER
        self.re = REModule().to(self.device)                # RE
        self.fc = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()
        
    def load_adapter(self, adapter_name):
        """
        Load a task-specific adapter into the LLM.
        adapter_name: str, identifier for the adapter
        """
        if hasattr(self.llm, "load_adapter"):
            self.llm.load_adapter(adapter_name)
            print(f"[INFO] Adapter '{adapter_name}' loaded into LLM.")
        else:
            raise AttributeError("LLMModule does not support adapters.")

    def forward(self, image):
        text = self.vision_enc.forward(image)
        embedding = self.llm.forward(image, text)

        entities = self.ner.forward(text)
        relations = self.re.forward(entities, embedding)

        out = self.relu(self.fc(embedding))

        return {
            "embedding": embedding,
            "text": text,
            "entities": entities,
            "relations": relations,
            "prediction": out
        }
