from flask import Flask, render_template, request
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load pre-trained model and tokenizer
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

app = Flask(__name__)

def generate_line(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=60,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)

def generate_haiku():
    line1 = generate_line("A tranquil evening for me, is")
    # line1 = line1.split('.')[0].split(',')[0].strip()
    line1 = line1.split('.')[0].strip()
    
    line2 = generate_line("While I walk through the meadow at night, I")
    #line2 = line2.split('.')[0].split(',')[0].strip()
    line2 = line2.split('.')[0].strip()
    
    line3 = generate_line("And then I can see clearly that - ")
    #line3 = line3.split('.')[0].split(',')[0].strip()
    line3 = line3.split('.')[0].strip()
    
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
