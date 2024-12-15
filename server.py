from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Charger le modèle et le tokenizer
model_name = "jassercherif/Llama-2-7b-doctorstr-finetuned"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Exemple de texte
prompt = "I have been having a lot of catching, pain and discomfort under my right rib..."

# Tokeniser le texte d'entrée
inputs = tokenizer(prompt, return_tensors="pt")

# Générer la réponse
with torch.no_grad():
    outputs = model.generate(inputs["input_ids"], max_length=200)

# Décoder et afficher la réponse
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
