import spacy

# Load a pre-trained English NER model
nlp = spacy.load("en_core_web_sm")  # or "en_core_web_trf" if you want transformer-based

text = "Pashupatinath is a major Hindu temple."
doc = nlp(text)

entities = [{"text": ent.text, "type": ent.label_} for ent in doc.ents]
print(entities)
