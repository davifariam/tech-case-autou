import os
import re
from flask import Flask, render_template, request, jsonify
from openai import OpenAI
from dotenv import load_dotenv
import PyPDF2

load_dotenv()

app = Flask(__name__)
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def read_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

def analyze_email(content):
    prompt = f"""
    Analise o seguinte email.
    1. Classifique como "Produtivo" (requer ação/trabalho) ou "Improdutivo" (apenas conversa/spam).
    2. Sugira uma resposta curta e profissional.
    
    Email: {content}
    
    Retorne APENAS um JSON neste formato, sem crases ou markdown:
    {{
        "classification": "Produtivo" ou "Improdutivo",
        "reply": "Sua sugestão de resposta aqui."
    }}
    """
    
    response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.3
)
    
    import json
    try:
        content = response.choices[0].message.content
        return json.loads(content)
    except:
        return {"classification": "Erro", "reply": "Não foi possível processar."}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    content = ""
    
    if 'file' in request.files and request.files['file'].filename != '':
        file = request.files['file']
        if file.filename.endswith('.pdf'):
            content = read_pdf(file)
        elif file.filename.endswith('.txt'):
            content = file.read().decode('utf-8')
    else:
        content = request.form.get('email_text', '')

    if not content:
        return jsonify({"error": "Nenhum conteúdo fornecido"}), 400

    clean_content = preprocess_text(content)
    
    result = analyze_email(clean_content)
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)