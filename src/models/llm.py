import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

class LLMModule:
    def __init__(self, model_name="google/gemma-3-270m", device=None, adapter_name=None):
        """
        model_name: base LM
        adapter_name: path or HF repo of the adapter
        device: 'cuda' or 'cpu'
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)

        if adapter_name:
            self.model = PeftModel.from_pretrained(self.model, adapter_name).to(self.device)

    def forward(self, input_text):
        """
        input_text: string or list of strings
        returns: LM embeddings (last hidden states)
        """
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True).to(self.device)
        outputs = self.model(**inputs, output_hidden_states=True)
        # Usually, last_hidden_state is [batch, seq_len, hidden_dim]
        return outputs
