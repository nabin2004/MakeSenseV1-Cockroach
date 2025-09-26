import torch
import torch.nn as nn
from peft import PeftModel
from .llm import LLMModule

class REModule(nn.Module):
    def __init__(self, llm_module: LLMModule, hidden_dim=640, num_relations=10, adapter_name=None):
        """
        llm_module: your existing Gemma3 LLMModule
        hidden_dim: hidden size of LLM embeddings
        num_relations: number of relation types (including 'no_relation')
        adapter_name: optional adapter for RE
        """
        super().__init__()
        self.llm = llm_module
        if adapter_name:
            self.llm.model = PeftModel.from_pretrained(self.llm.model, adapter_name).to(self.llm.device)

        # Simple classifier for entity pair relations
        # Input: concatenated embeddings of (entity1, entity2)
        self.classifier = nn.Linear(hidden_dim * 2, num_relations)
        self.relation_map = {0: "no_relation", 1: "related_to"}  # extend with more relation labels

    def forward(self, entities, text_embedding):
        """
        entities: list of entity spans (strings or token indices)
        text_embedding: LLM embeddings, shape [seq_len, hidden_dim] or [batch, seq_len, hidden_dim]
        returns: list of (entity1, relation, entity2) tuples
        """
        relations = []


        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                e1_emb = self._get_entity_embedding(entities[i], text_embedding)
                e2_emb = self._get_entity_embedding(entities[j], text_embedding)

                pair_emb = torch.cat([e1_emb, e2_emb], dim=-1)  # [1, hidden_dim*2]
                logits = self.classifier(pair_emb)               # [1, num_relations]
                pred = logits.argmax(dim=-1).item()
                relation = self.relation_map.get(pred, "no_relation")

                relations.append((entities[i], relation, entities[j]))

        return relations

    def _get_entity_embedding(self, entity, text_embedding):
        """
        Placeholder: compute embedding for entity
        Currently just averages over all token embeddings corresponding to entity
        """
        # In practice, use entity span indices to select tokens
        # Here, we just mean-pool over sequence as a placeholder
        return text_embedding.mean(dim=1)
