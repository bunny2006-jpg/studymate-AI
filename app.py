# app.py

import streamlit as st
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
import time
from PIL import Image
import pytesseract
import shutil
from langdetect import detect, LangDetectException
import re

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="StudyMate AI 📚",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- SYSTEM CHECKS & MODEL LOADING ---
@st.cache_resource
def check_tesseract():
    """Checks if Tesseract is installed and in the system's PATH."""
    return shutil.which("tesseract") is not None

@st.cache_resource
def load_models():
    """Loads all AI models once and caches them."""
    with st.spinner("Loading AI models... This may take a moment."):
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        model_path = "ibm-granite/granite-4.0-micro"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        model.eval()
    return embedding_model, model, tokenizer

# --- BACKEND FUNCTIONS ---
def extract_text_from_pdfs(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        with fitz.open(stream=pdf.read(), filetype="pdf") as doc:
            text += "".join(page.get_text() for page in doc)
    return text

def extract_text_from_images(image_files, tesseract_installed, lang_codes):
    if not tesseract_installed:
        st.error("Tesseract-OCR is not installed or configured correctly. Image processing is disabled.")
        return ""
    lang_string = "+".join(lang_codes)
    text = ""
    for image_file in image_files:
        try:
            image = Image.open(image_file)
            text += pytesseract.image_to_string(image, lang=lang_string) + "\n"
        except Exception as e:
            st.warning(f"Could not process an image. Error: {e}")
    return text

def split_text_into_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    return text_splitter.split_text(text)

def create_faiss_index(chunks, embedding_model):
    if not chunks: return None
    with st.spinner("Creating a searchable index for your documents..."):
        embeddings = embedding_model.encode(chunks, show_progress_bar=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings, dtype='float32'))
    return index

def get_relevant_context(query, index, chunks, embedding_model, top_k=3):
    if index is None or not chunks: return []
    query_embedding = embedding_model.encode([query])
    _, indices = index.search(np.array(query_embedding, dtype='float32'), top_k)
    return [chunks[i] for i in indices[0]]

def generate_response(prompt_text, model, tokenizer):
    inputs = tokenizer(prompt_text, return_tensors="pt")
    outputs = model.generate(
        **inputs, max_new_tokens=400, temperature=0.2, pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.replace(prompt_text, "").strip()

def solve_simple_math(prompt: str):
    math_pattern = re.compile(r"^\s*([\d\.]+)\s*([+\-*/])\s*([\d\.]+)\s*$")
    match = math_pattern.match(prompt)
    if match:
        num1, operator, num2 = match.groups()
        try:
            result = 0
            if operator == '+': result = float(num1) + float(num2)
            elif operator == '-': result = float(num1) - float(num2)
            elif operator == '*': result = float(num1) * float(num2)
            elif operator == '/':
                if float(num2) == 0: return "Sorry, I can't divide by zero."
                result = float(num1) / float(num2)
            if result.is_integer(): return str(int(result))
            else: return f"{result:.2f}"
        except ValueError: return None
    return None

def handle_prompt_and_generate_response(prompt_text, llm_model, llm_tokenizer, embedding_model):
    math_result = solve_simple_math(prompt_text)
    if math_result is not None:
        return math_result, "math"

    source_of_answer = "general"
    
    # +++ MODIFIED +++ This is the new, highly-detailed "Master Tutor Prompt".
    # It instructs the AI to generate comprehensive, structured answers like your example.
    master_tutor_prompt = (
        "You are StudyMate, an expert AI research assistant and tutor. Your primary goal is to provide a comprehensive, well-structured, and easy-to-understand answer to the student's question. "
        "To do this, you will structure your response with the following sections using Markdown formatting:\n\n"
        "1.  **General Definition**: Start with a clear and concise definition of the core concept.\n"
        "2.  **Key Capabilities / What it Can Answer**: Use a bulleted list to explain the main functions or types of problems it can solve.\n"
        "3.  **How it Works**: Use another bulleted list to explain the underlying mechanism or process.\n"
        "4.  **Examples (if applicable)**: Provide a few concrete examples.\n\n"
        "Synthesize information to provide a complete picture. Be encouraging and act as a supportive study partner."
    )
    final_prompt_for_llm = f"{master_tutor_prompt}\n\n### Student's Question:\n{prompt_text}"

    # This part remains the same: check for document context first.
    if st.session_state.get("doc_processed") and st.session_state.get("faiss_index"):
        context_k = st.session_state.get('context_k', 3)
        context_chunks = get_relevant_context(prompt_text, st.session_state.faiss_index, st.session_state.chunks, embedding_model, top_k=context_k)
        
        if context_chunks:
            source_of_answer = "documents"
            context = "\n\n---\n\n".join(context_chunks)
            final_prompt_for_llm = f"Based ONLY on the context below, answer the question. If the answer is not in the context, say that you cannot find the answer in the provided documents.\n\nContext:\n{context}\n\nQuestion:\n{prompt_text}"

    # Language and ELI5 modifiers are applied after the main prompt is set.
    try:
        detected_lang_code = detect(prompt_text)
        language_map = {"hi": "Hindi", "es": "Spanish", "fr": "French", "de": "German", "en": "English"}
        response_language = language_map.get(detected_lang_code, "the user's language")
        final_prompt_for_llm += f"\n\nPlease provide the answer in {response_language}."
    except LangDetectException:
        pass
        
    if st.session_state.get('eli5_mode', False):
        final_prompt_for_llm += " Also, explain it simply, like I'm 5 years old."

    answer = generate_response(final_prompt_for_llm, llm_model, llm_tokenizer)
    return answer, source_of_answer

def stream_typing_effect(text):
    placeholder = st.empty()
    full_response = ""
    for word in text.split():
        full_response += word + " "
        time.sleep(0.01)
        placeholder.markdown(full_response + "▌")
    placeholder.markdown(full_response)
    return full_response

# --- STREAMLIT UI ---
def main():
    st.title("StudyMate AI 📚")

    tesseract_installed = check_tesseract()
    embedding_model, llm_model, llm_tokenizer = load_models()
    
    for key, val in {
        "conversation": [], "faiss_index": None, "chunks": [], 
        "doc_processed": False, "processed_image_id": None, "use_camera": False
    }.items():
        if key not in st.session_state: st.session_state[key] = val

    with st.sidebar:
        st.header("Get Started")
        with st.expander("1. Upload & Process Sources", expanded=True):
            st.session_state.ocr_languages = st.multiselect("Language(s) in images", options=['eng', 'hin', 'spa', 'fra', 'deu'], default=['eng'])
            pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True, type="pdf")
            image_docs = st.file_uploader("Upload Source Images", accept_multiple_files=True, type=["png", "jpg", "jpeg"])
            st.checkbox("📸 Use Camera to Scan Source", key="use_camera")
            camera_photo = None
            if st.session_state.use_camera:
                camera_photo = st.camera_input("Take a Photo of a Source", label_visibility="collapsed")
            if st.button("Analyze All Sources", type="primary"):
                with st.spinner("Reading and analyzing your documents..."):
                    text = ""
                    if pdf_docs: text += extract_text_from_pdfs(pdf_docs)
                    ocr_langs = st.session_state.get('ocr_languages', ['eng'])
                    if image_docs: text += extract_text_from_images(image_docs, tesseract_installed, ocr_langs)
                    if camera_photo: text += extract_text_from_images([camera_photo], tesseract_installed, ocr_langs)
                    st.session_state.chunks = split_text_into_chunks(text)
                    st.session_state.faiss_index = create_faiss_index(st.session_state.chunks, embedding_model)
                    st.session_state.doc_processed = True
                st.success("✅ Analysis Complete!")
        if st.session_state.doc_processed:
            with st.expander("2. Chat Settings", expanded=True):
                st.session_state.eli5_mode = st.checkbox("🧒 Explain Like I'm 5", value=False)
                st.session_state.context_k = st.slider("🧠 Context Depth", 1, 5, 3)
            with st.expander("3. Conversation Management"):
                if st.button("🗑️ Clear Chat History"):
                    st.session_state.conversation = []
                    st.rerun()

    if not st.session_state.doc_processed:
        st.info("👋 Welcome! Ask me any general study question, or upload documents to begin.")
    else:
        st.info("✅ Documents loaded! Ask questions about your files or any other study topic.")

    st.header("Chat Interface")
    for message in st.session_state.conversation:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    st.markdown("---")
    question_image = st.file_uploader("Upload an image of a question", type=["png", "jpg", "jpeg"])

    if st.session_state.doc_processed:
        placeholder_text = "Ask about your documents or any other topic..."
    else:
        placeholder_text = "Ask any study question..."
    prompt = st.chat_input(placeholder_text)

    question_to_process = None
    if question_image and question_image.getvalue() != st.session_state.get("processed_image_id"):
        st.session_state.processed_image_id = question_image.getvalue()
        with st.chat_message("user"):
            st.image(question_image, width=350)
        ocr_langs = st.session_state.get('ocr_languages', ['eng'])
        question_to_process = extract_text_from_images([question_image], tesseract_installed, ocr_langs).strip()
    elif prompt:
        question_to_process = prompt

    if question_to_process:
        st.session_state.conversation.append({"role": "user", "content": question_to_process})
        
        with st.chat_message("assistant"):
            with st.spinner("🧠 Thinking..."):
                answer, source = handle_prompt_and_generate_response(question_to_process, llm_model, llm_tokenizer, embedding_model)
                stream_typing_effect(answer)
                
                if source == "documents":
                    st.info("💡 Answered using your uploaded documents.", icon="📄")
                elif source == "general":
                    st.info("💡 Answered using my general knowledge.", icon="🌐")

        st.session_state.conversation.append({"role": "assistant", "content": answer})
        st.rerun()

if __name__ == '__main__':
    main()

