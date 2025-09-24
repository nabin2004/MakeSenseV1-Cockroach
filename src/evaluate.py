import torch
from torch.utils.data import DataLoader
import json
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

class Evaluator:
    def __init__(self, model, dataset, batch_size=2, device=None):
        """
        model: your MakeSense model
        dataset: torch Dataset object
        batch_size: evaluation batch size
        device: 'cuda' or 'cpu'
        """
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        self.model.to(self.device)
        self.model.eval()
        
    def evaluate(self, save_path=None):
        """
        Runs evaluation over the dataset and optionally saves results.
        Returns metrics dict and results list.
        """
        all_results = []
        ner_true, ner_pred = [], []  # for seqeval NER metrics
        
        with torch.no_grad():
            for batch in self.loader:
                images = batch["image"].to(self.device)
                texts = batch["text"]
                annotations = batch.get("annotations", [None] * len(texts))

                outputs = self.model.forward(images)

                # Save raw outputs
                for i, text in enumerate(texts):
                    all_results.append({
                        "text": text,
                        "entities": outputs["entities"][i] if isinstance(outputs["entities"], list) else outputs["entities"],
                        "relations": outputs["relations"][i] if isinstance(outputs["relations"], list) else outputs["relations"]
                    })

                # Collect for NER metrics if annotations exist
                for i, ann in enumerate(annotations):
                    if ann and "entities" in ann:
                        # seqeval expects list of labels per token
                        ner_true.append(ann["entities"])
                        ner_pred.append([e["type"] for e in outputs["entities"][i]])

        # Compute NER metrics
        ner_metrics = {}
        if ner_true:
            ner_metrics = {
                "precision": precision_score(ner_true, ner_pred),
                "recall": recall_score(ner_true, ner_pred),
                "f1": f1_score(ner_true, ner_pred),
                "report": classification_report(ner_true, ner_pred)
            }

        # Save results if path provided
        if save_path:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, ensure_ascii=False, indent=4)

        return ner_metrics, all_results
