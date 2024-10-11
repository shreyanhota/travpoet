from flask import Flask, render_template, request
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load pre-trained model and tokenizer
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

app = Flask(__name__)

def generate_sentence(prompt):
    # Create a custom prompt based on the selected location
    
    # Tokenize the prompt for GPT-2 input
    inputs = tokenizer.encode(prompt, return_tensors="pt")

    # Generate output text using GPT-2
    outputs = model.generate(inputs, max_length=15, num_return_sequences=1, 
                             no_repeat_ngram_size=2, do_sample=True, top_p=0.95, temperature=0.7)
    
    # Decode the output text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Find the first full stop to return all words up to that point
    if "." in generated_text:
        sentence = generated_text[:generated_text.index(".") + 1]  # Include the full stop
    else:
        sentence = generated_text  # In case no full stop is generated
    
    return sentence

def generate_haiku():
    line1 = generate_sentence("A relaxing trip to the beach where")
    # line1 = line1.split('.')[0].split(',')[0].strip()
    if "." in line1:
        sentence1 = line1[:line1.index(".") + 1]  # Include the full stop
    else:
        sentence1 = line1  # In case no full stop is generated
    
    line2 = generate_sentence("A vibrant trip to the bustling city where")
    #line2 = line2.split('.')[0].split(',')[0].strip()
    if "." in line2:
        sentence2 = line2[:line2.index(".") + 1]  # Include the full stop
    else:
        sentence2 = line2  # In case no full stop is generated
    
    line3 = generate_sentence("A thrilling journey through the mountains where")
    #line3 = line3.split('.')[0].split(',')[0].strip()
    if "." in line3:
        sentence3 = line3[:line3.index(".") + 1]  # Include the full stop
    else:
        sentence3 = line3  # In case no full stop is generated
    
    haiku = f"\n{line1}\n{line2}\n{line3}"
    return haiku

@app.route('/', methods=['GET', 'POST'])
def home():
    haiku = ""
    if request.method == 'POST':
        haiku = generate_haiku()  # Generate haiku when button is clicked
    return render_template('index.html', haiku=haiku)

if __name__ == '__main__':
    app.run(debug=True)
