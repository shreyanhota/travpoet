import gradio as gr
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

def generate_text(prompt):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=50, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Create the Gradio interface
interface = gr.Interface(fn=generate_text, inputs="text", outputs="text", title="GPT-2 Text Generator")
interface.launch()
