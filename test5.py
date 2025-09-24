from src.models.llm import LLMModule
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
llm = LLMModule(model_name="google/gemma-3-270m-it", device=device)

# Example sentence
sentence = "Barack Obama was born in Hawaii."

# Manually define candidate entities
entities = ["Barack Obama", "Hawaii"]

for ent in entities:
    prompt = (
        f"You are an expert information extractor. Carefully analyze the following sentence and identify the type of a specific entity mentioned in it.\n\n"
        f"Sentence: '{sentence}'\n"
        f"Entity to classify: '{ent}'\n\n"
        "Possible entity types:\n"
        "- PERSON: Human names, individuals, or groups of people\n"
        "- PLACE: Geographic locations such as cities, countries, landmarks\n"
        "- ORGANIZATION: Companies, institutions, governmental or non-governmental organizations\n"
        "- OTHER: Any entity that does not fit the above categories\n\n"
        "Instructions:\n"
        "1. Read the sentence carefully and focus only on the specified entity.\n"
        "2. Choose the entity type that best describes it, and only one.\n"
        "3. Do not guess; if unsure, classify as OTHER.\n\n"
        "Answer:"
    )
    
    outputs = llm.forward(prompt)
    logits = outputs.logits
    last_token_logits = logits[0, -1, :]
    pred_id = torch.argmax(last_token_logits).item()
    pred_token = llm.tokenizer.decode([pred_id])
    print("======================================================================")
    print(f"Entity classification -> {ent}: {pred_token}")
    print("======================================================================")
