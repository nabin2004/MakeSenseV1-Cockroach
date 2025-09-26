from models.SigLIP2Module import SigLIP2Module
from models.llm import LLMModule
from models.ner import NERModule
from models.re_module import REModule    

import torch
import torch.nn as nn


class MakeSense(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MakeSense, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Vision encoder
        self.vision_enc = SigLIP2Module()
        for param in self.vision_enc.model.parameters():  # freeze vision encoder
            param.requires_grad = False
                    
        # Projection layer (trainable)
        siglip_embed_dim = 768 # 768 for SigLIP2
        llm_hidden_dim = 640  # 640 For Gemma-3-270m 
        
        self.proj_layer = nn.Sequential(
            nn.Linear(siglip_embed_dim, llm_hidden_dim),
            nn.ReLU(),
            nn.Linear(llm_hidden_dim, llm_hidden_dim)
        ).to(self.device)

        # LLM (frozen)
        self.llm = LLMModule(device=self.device)
        for param in self.llm.model.parameters():
            param.requires_grad = False

        # # NER and RE (frozen)
        # self.ner = NERModule(self.llm)
        # for param in self.ner.parameters():
        #     param.requires_grad = False
        # self.re = REModule(self.llm)
        # for param in self.re.parameters():
        #     param.requires_grad = False

        # Output head
        # self.fc = nn.Linear(input_dim, output_dim).to(self.device)
        # self.relu = nn.ReLU()
        self.fc = nn.Linear(llm_hidden_dim, output_dim).to(self.device)
        self.relu = nn.ReLU()

    def load_adapter(self, adapter_name):
        if hasattr(self.llm, "load_adapter"):
            self.llm.load_adapter(adapter_name)
            print(f"[INFO] Adapter '{adapter_name}' loaded into LLM.")
        else:
            raise AttributeError("LLMModule does not support adapters.")

    def forward(self, image, text=None):
        # Step 1: Vision embeddings
        with torch.no_grad():  # no gradient through vision encoder
            vis_embeds = self.vision_enc.forward(image)

        # Step 2: Project vision embeddings
        proj_embeds = self.proj_layer(vis_embeds)  # trainable

        # Step 3: Optional text (dummy or real)
        if text is None:
            text = "Nabin Kripa Oli is going to Stanford for MSc Symbolic Systems degree."

        # Step 4: LLM embeddings (frozen)
        with torch.no_grad():
            outputs = self.llm.forward(text)
            lm_embeds = outputs.hidden_states[-1]  # [batch, seq_len, hidden_dim]

        # Combine projected vision embeddings + LM embeddings
        combined_embeds = torch.cat([proj_embeds.unsqueeze(1), lm_embeds], dim=1)  # prepend as pseudo-token

        # Step 5: Pooling over sequence dimension
        pooled = combined_embeds.mean(dim=1)  # [batch, hidden_dim]

        # Step 6: FC head
        out = self.relu(self.fc(pooled))

        return {
            "proj_embedding": proj_embeds,
            "lm_embedding": lm_embeds,
            "combined_embedding": combined_embeds,
            "pooled": pooled,
            "prediction": out
        }


if __name__ == "__main__":
    input_dim = 768
    output_dim = 10
    model = MakeSense(input_dim, output_dim)

    from transformers.image_utils import load_image
    image = load_image("https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg")

    # Optimizer only for projection layer
    optimizer = torch.optim.Adam(model.proj_layer.parameters(), lr=1e-3)

    # Dummy training step
    model.train()
    text_dummy = "Dummy text for testing."
    optimizer.zero_grad()
    outputs = model(image, text_dummy)
    loss = outputs["prediction"].sum()  
    loss.backward()  # gradients only computed for proj_layer
    optimizer.step()

    print("Forward pass completed. Only projection layer updated.")
