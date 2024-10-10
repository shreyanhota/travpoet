from flask import Flask, render_template, request
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

app = Flask(__name__)

# Load GPT-2 model and tokenizer
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

def generate_caption(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    # Generate text
    with torch.no_grad():
        outputs = model.generate(input_ids, max_length=50, num_return_sequences=1, 
                             no_repeat_ngram_size=2, do_sample=True, top_p=0.95, temperature=0.7)
    caption = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "." in caption:
        sentence = f'"{caption[:caption.index(".") + 1]}"'  # Include the full stop
    return sentence

@app.route('/', methods=['GET', 'POST'])
def index():
    caption = ""
    if request.method == 'POST':
        tone = request.form['tone']
        theme = request.form['theme']
        location = request.form['location']
        
        

        prompt = f"A {tone} trip to the {location} {theme} is"
        caption = generate_caption(prompt)

    return render_template('index.html', caption=caption)

if __name__ == '__main__':
    app.run(debug=True)
