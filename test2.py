from src.models.llm import LLMModule
from src.models.ner import NERModule

llm = LLMModule(model_name="google/gemma-3-270m", device="cpu")
ner = NERModule(llm)
entities = ner.forward("Barack Obama was born in Hawaii.", return_entities=True)
print(entities)