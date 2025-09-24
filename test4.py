import torch
import spacy
from src.models.llm import LLMModule
from src.models.re_module import REModule

# Initialize LLM
device = "cuda" if torch.cuda.is_available() else "cpu"
llm = LLMModule(model_name="google/gemma-3-270m", device=device)

# Initialize RE Module
re_module = REModule(llm_module=llm)

# Load spaCy NER model
nlp = spacy.load("en_core_web_sm")

# Dummy sentence
sentence = "Barack Obama was born in Hawaii and later moved to Chicago."

# --- Step 1: Extract entities ---
doc = nlp(sentence)
entities = [ent.text for ent in doc.ents]
print("Entities detected:", entities)

# --- Step 2: Get text embeddings from LLM ---
outputs = llm.forward(sentence)
# Take last hidden layer
text_embedding = outputs.hidden_states[-1]  # [batch, seq_len, hidden_dim]

# --- Step 3: Predict relations ---
relations = re_module.forward(entities, text_embedding)
print("\nPredicted Relations:")
for r in relations:
    print(r)
