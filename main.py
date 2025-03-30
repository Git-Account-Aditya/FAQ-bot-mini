from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os
import streamlit as st
import shutil

# Load environment variables
load_dotenv()

# Ensure 'files' directory exists
os.makedirs('files', exist_ok=True)

# Initialize Groq LLM
groq = ChatGroq(model_name="llama3-8b-8192", api_key=os.getenv("GROQ_API_KEY"))

# Define the prompt template
template = "Here is the text: {text}. Answer the following question based on the given text: {question}."
prompt = PromptTemplate(template=template, input_variables=["text", "question"])

# Create LangChain pipeline
chain = prompt | groq

# Streamlit UI
st.title("FAQ Bot Mini")

uploaded_file = st.file_uploader("Upload a text or PDF file", type=["txt", "pdf"])
max_chars = 5000  # Token limit to prevent exceeding model constraints
text = ""

if uploaded_file is not None:
    file_extension = uploaded_file.name.split(".")[-1]
    file_save_path = os.path.join("files", uploaded_file.name)

    # Save the file
    with open(file_save_path, "wb") as f:
        shutil.copyfileobj(uploaded_file, f)

    # Load text from the file
    if file_extension == "txt":
        st.write("✅ Text file uploaded")
        loader = TextLoader(file_path=file_save_path, encoding="utf-8")
    elif file_extension == "pdf":
        st.write("✅ PDF file uploaded")
        loader = PyPDFLoader(file_path=file_save_path)

    # Extract text content
    documents = loader.load()

    text_collected = []
    total_chars = 0

    for doc in documents:
        doc_content = doc.page_content
        remaining_chars = max_chars - total_chars

        if remaining_chars <= 0:
            break  # Stop adding text if max_chars is reached

        if len(doc_content) > remaining_chars:
            text_collected.append(doc_content[:remaining_chars])  # Truncate
            total_chars += remaining_chars
        else:
            text_collected.append(doc_content)
            total_chars += len(doc_content)

    text = "\n".join(text_collected)

# Text input for user query
question = st.text_input("Ask Anything...")

if st.button("Submit"):
    if not text:
        st.warning("⚠️ Please upload a file first!")
    elif not question.strip():
        st.warning("⚠️ Please enter a question!")
    else:
        # Invoke the model
        result = chain.invoke({"text": text, "question": question})
        st.write("🤖 **AI Response:**")
        st.write(result.content)
