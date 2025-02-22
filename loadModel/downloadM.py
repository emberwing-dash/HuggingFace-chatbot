from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

model_name = "facebook/blenderbot-400M-distill"
model_dir = "E:/SRM/university projects/side projects/hackathon/2025/4. Infusion/chatbot/blenderbot"

# Download the model and tokenizer
tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
model = BlenderbotForConditionalGeneration.from_pretrained(model_name)

# Save the model and tokenizer locally
model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)
