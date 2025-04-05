from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
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
template = "Here is the text: {text}\n\nAnswer the following question based on the above text:\n{question}"
prompt = PromptTemplate(template=template, input_variables=["text", "question"])
chain = prompt | groq

# Streamlit UI
st.title("FAQ Bot Mini")

uploaded_file = st.file_uploader("Upload a text or PDF file", type=["txt", "pdf"])
splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=400)
all_chunks = []

if uploaded_file is not None:
    file_extension = uploaded_file.name.split(".")[-1]
    file_save_path = os.path.join("files", uploaded_file.name)

    # Save uploaded file
    with open(file_save_path, "wb") as f:
        shutil.copyfileobj(uploaded_file, f)

    # Load and split file
    if file_extension == "txt":
        st.success("Text file uploaded")
        loader = TextLoader(file_path=file_save_path, encoding="utf-8")
    elif file_extension == "pdf":
        st.success("PDF file uploaded")
        loader = PyPDFLoader(file_path=file_save_path)

    documents = loader.load()
    split_docs = splitter.split_documents(documents)
    all_chunks = [doc.page_content for doc in split_docs]

# Input for user's question
question = st.text_input("Ask anything based on the uploaded document:")

if st.button("Submit"):
    if not all_chunks:
        st.warning("Please upload a file first!")
    elif not question.strip():
        st.warning("Please enter a question!")
    else:
        responses = []
        with st.spinner("Thinking..."):
            for chunk in all_chunks:
                result = chain.invoke({"text": chunk, "question": question})
                responses.append(result.content)

        final_response = "\n\n".join(responses)
        st.write("🤖 AI Response : ")
        st.write(final_response)

