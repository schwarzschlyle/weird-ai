from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/generate_summary', methods=['POST'])
def generate_summary():
    text = request.form['text']

    summarizer = pipeline("summarization")
    summary = summarizer(text, max_length=150, min_length=30, do_sample=False)

    return render_template('index.html', text=text, summary=summary[0]['summary_text'])


if __name__ == '__main__':
    app.run(debug=True)
