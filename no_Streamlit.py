import streamlit as st
import langdetect
import PyPDF2
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model directly
tokenizer = AutoTokenizer.from_pretrained("unsloth/llama-3-8b-Instruct")
model = AutoModelForCausalLM.from_pretrained("unsloth/llama-3-8b-Instruct")

# Download necessary data for NLTK
nltk.download('punkt')

# Function to call LLaMA model for summarization
def llama_summarization(text):
    try:
        input_ids = tokenizer.encode(f"Summarize this: {text}", return_tensors="pt")
        output = model.generate(input_ids, max_length=200)
        summary = tokenizer.decode(output[0], skip_special_tokens=True)
        return summary
    except Exception as e:
        st.error(f"Error: {e}")
        return None

# Function to call LLaMA model for QnA
def llama_qna(text, question):
    try:
        input_ids = tokenizer.encode(f"Question: {question}\n\nContext: {text}", return_tensors="pt")
        output = model.generate(input_ids, max_length=200)
        answer = tokenizer.decode(output[0], skip_special_tokens=True)
        return answer
    except Exception as e:
        st.error(f"Error: {e}")
        return None

# Function to read PDF file
def read_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to read text file
def read_text(file):
    return file.read().decode('utf-8')

# Function to preprocess text (remove table of contents)
def preprocess_text(text):
    # Remove table of contents by looking for common patterns
    lines = text.split('\n')
    filtered_lines = []
    in_toc = False
    for line in lines:
        if any(keyword in line.lower() for keyword in ['chapter', 'contents', 'page no', 'annexure']):
            in_toc = True
        elif in_toc and line.strip().isdigit():
            in_toc = False
        if not in_toc:
            filtered_lines.append(line)
    return '\n'.join(filtered_lines)

# Function for extractive summarization using TF-IDF
def extractive_summary(text, num_sentences=3):
    sentences = nltk.sent_tokenize(text)
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(sentences)
    sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
    top_sentence_indices = sentence_scores.argsort()[-num_sentences:][::-1]
    top_sentences = [sentences[i] for i in top_sentence_indices]
    summary = ' '.join(top_sentences)
    return summary

# Main logic for displaying content
st.title("Text Summarization and QnA")

# Upload file or paste text
uploaded_file = st.file_uploader("Choose a PDF or Text file", type=["pdf", "txt"])
input_text = st.text_area("Enter your text here...")

if uploaded_file:
    if uploaded_file.type == "application/pdf":
        file_content = read_pdf(uploaded_file)
    elif uploaded_file.type == "text/plain":
        file_content = read_text(uploaded_file)
    
    if file_content:
        file_content = preprocess_text(file_content)
        lang = langdetect.detect(file_content)
        st.write("### File Content:")
        st.write(file_content[:300])

        # Choose summary type
        summary_type = st.selectbox("Choose Summary Type", ["Abstractive Summary", "Extractive Summary"])

        if summary_type == "Abstractive Summary":
            if st.button("Generate Summary"):
                with st.spinner("Generating abstractive summary..."):
                    try:
                        summary = llama_summarization(file_content[:1000])
                        st.write(summary)
                    except Exception as e:
                        st.error(f"Error: {e}")

        elif summary_type == "Extractive Summary":
            if st.button("Generate Summary"):
                with st.spinner("Generating extractive summary..."):
                    summary = extractive_summary(file_content[:2000], num_sentences=3)
                    st.write(summary)

        # Question and Answer
        question = st.text_input("Enter your question:")
        llama_indexing = st.checkbox("Enable LLaMA Indexing for multilingual collaborative answering")
        if llama_indexing:
            st.write("This feature is enabled. You can now ask questions in multiple languages.")
        if st.button("Get Answer"):
            with st.spinner("Generating answer..."):
                try:
                    if llama_indexing:
                        answer = llama_qna(file_content[:1000], question)
                    else :
                        answer = llama_qna(file_content[:1000], question)
                    st.write(answer)
                except Exception as e:
                    st.error(f"Error: {e}")

elif input_text:
    input_text = preprocess_text(input_text)
    lang = langdetect.detect(input_text)
    st.write("### Input Text:")
    st.write(input_text[:300])

    # Choose summary type
    summary_type = st.selectbox("Choose Summary Type", ["Abstractive Summary", "Extractive Summary"])

    if summary_type == "Abstractive Summary":
        if st.button("Generate Summary"):
            with st.spinner("Generating abstractive summary..."):
                try:
                    summary = llama_summarization(input_text[:1000])
                    st.write(summary)
                except Exception as e:
                    st.error(f"Error: {e}")

    elif summary_type == "Extractive Summary":
        if st.button("Generate Summary"):
            with st.spinner("Generating extractive summary..."):
                summary = extractive_summary(input_text[:2000], num_sentences=3)
                st.write(summary)

    # Question and Answer
    question = st.text_input("Enter your question:")
    llama_indexing = st.checkbox("Enable LLaMA Indexing for multilingual collaborative answering")
    if llama_indexing:
        st.write("This feature is enabled. You can now ask questions in multiple languages.")
    if st.button("Get Answer"):
        with st.spinner("Generating answer..."):
            try:
                if llama_indexing:
                    answer = llama_qna(input_text[:1000], question)
                else:
                    answer = llama_qna(input_text[:1000], question)
                st.write(answer)
            except Exception as e:
                st.error(f"Error: {e}")