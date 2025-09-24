import torch
import torch.nn as nn
from peft import PeftModel
from .llm import LLMModule

class NERModule(nn.Module):
    def __init__(self, llm_module: LLMModule, hidden_dim=640, num_labels=10, label_map=None, adapter_name=None):
        super().__init__()
        self.llm = llm_module
        if adapter_name:
            self.llm.model = PeftModel.from_pretrained(self.llm.model, adapter_name).to(self.llm.device)

        self.classifier = nn.Linear(hidden_dim, num_labels)
        self.label_map = label_map or {
            0: "O",
            1: "B-PER",
            2: "I-PER",
            3: "B-LOC",
            4: "I-LOC",
            5: "B-ORG",
            6: "I-ORG",
            7: "B-MISC",
            8: "I-MISC",
            9: "PAD"
        }

    def decode_entities(self, logits, input_ids):
        preds = logits.argmax(dim=-1)  # [batch, seq_len]
        entities = []

        for batch_idx in range(preds.size(0)):
            tokens = self.llm.tokenizer.convert_ids_to_tokens(input_ids[batch_idx])
            labels = [self.label_map[i] for i in preds[batch_idx].cpu().tolist()]

            current = None
            for token, label in zip(tokens, labels):
                if label.startswith("B-"):
                    if current: entities.append(current)
                    current = {"type": label[2:], "text": token}
                elif label.startswith("I-") and current:
                    current["text"] += " " + token
                else:
                    if current: entities.append(current); current=None
            if current: entities.append(current)

        return entities

    def forward(self, text, return_entities=False):
        inputs = self.llm.tokenizer(text, return_tensors="pt", padding=True).to(self.llm.device)
        outputs = self.llm.model(**inputs, output_hidden_states=True)

        hidden_states = outputs.hidden_states[-1]       # [batch, seq_len, hidden_dim]
        logits = self.classifier(hidden_states)         # [batch, seq_len, num_labels]

        if return_entities:
            return self.decode_entities(logits, inputs["input_ids"])
        return logits
