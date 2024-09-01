import os
from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import UpstashVector
import tempfile
import time
import concurrent.futures

# Load API keys and configuration from environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')
UPSTASH_VECTOR_REST_URL = os.getenv("UPSTASH_VECTOR_REST_URL")
UPSTASH_VECTOR_REST_TOKEN = os.getenv("UPSTASH_VECTOR_REST_TOKEN")

# Custom CSS to style the app
st.markdown("""
<style>
    .main {
        background-color: #051622;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton > button {
        background-color: #3498db;
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        transition: background-color 0.3s;
    }
    .stButton > button:hover {
        background-color: #2980b9;
    }
    .stTextInput > div > div > input {
        background-color: #f0f0f0;
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 0.5rem;
    }
    .stExpander {
        background-color: #f8f9fa;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# App header
st.title("📚 DocuChat AI")
st.subheader("Your Intelligent Document Assistant")

# Initialize the language model
@st.cache_resource
def load_llm():
    return ChatGroq(groq_api_key=groq_api_key, model_name='gemma-7b-it')

llm = load_llm()

# Define the prompt template
prompt_template = """
Answer the question as truthfully as possible using the provided context. If the answer is not contained within the text below, say "I don't know" and suggest where the user might find more information.

Context: {context}
Question: {question}

Answer:
"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# Optimized vector store creation with Upstash Vector
@st.cache_resource
def create_vector_store(_docs):
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(_docs)
    
    texts = [doc.page_content for doc in final_documents]
    metadatas = [doc.metadata for doc in final_documents]
    
    vector_store = UpstashVector.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        url=UPSTASH_VECTOR_REST_URL,
        rest_api_key=UPSTASH_VECTOR_REST_TOKEN,
        index_name="docuchat_index"
    )
    
    return vector_store

# Function to process a single PDF file
def process_pdf(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file.getvalue())
        tmp_file_path = tmp_file.name

    loader = PyPDFLoader(tmp_file_path)
    docs = loader.load()

    os.unlink(tmp_file_path)
    return docs

# Sidebar for app controls
with st.sidebar:
    st.header("📋 Control Panel")
    uploaded_files = st.file_uploader("Upload PDF documents", type="pdf", accept_multiple_files=True)
    
    if uploaded_files:
        with st.spinner("Processing documents..."):
            with concurrent.futures.ThreadPoolExecutor() as executor:
                all_docs = list(executor.map(process_pdf, uploaded_files))
            
            all_docs = [doc for sublist in all_docs for doc in sublist]

            start_time = time.time()
            st.session_state.vector_store = create_vector_store(all_docs)
            end_time = time.time()

        st.success(f"{len(uploaded_files)} document(s) processed successfully!")
        st.info(f"Vector store creation time: {end_time - start_time:.2f} seconds")
    
    st.markdown("---")
    st.markdown("### 🔍 How to use:")
    st.markdown("1. Upload your PDF documents.")
    st.markdown("2. Enter your question in the main panel.")
    st.markdown("3. Click 'Ask' to get your answer.")
    
    st.markdown("---")
    st.markdown("### 📊 App Stats")
    if 'query_count' not in st.session_state:
        st.session_state.query_count = 0
    st.metric("Queries Answered", st.session_state.query_count)

# Main app area
st.markdown("## 🤔 Ask me anything about your documents")
prompt1 = st.text_input("Enter your question here:")

if st.button("Ask"):
    if "vector_store" not in st.session_state:
        st.error("⚠️ Please upload at least one document first using the sidebar.")
    elif not prompt1:
        st.warning("⚠️ Please enter a question.")
    else:
        with st.spinner("🧠 Thinking..."):
            retriever = st.session_state.vector_store.as_retriever()
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": PROMPT}
            )
            
            start = time.time()
            response = qa_chain({"query": prompt1})
            end = time.time()
            
            st.session_state.query_count += 1
        
        st.markdown("### 📝 Answer:")
        st.info(response['result'])
        st.markdown(f"*Response time: {end - start:.2f} seconds*")

        with st.expander("📚 View Source Documents"):
            for i, doc in enumerate(response["source_documents"]):
                st.markdown(f"**Document {i+1}:**")
                st.markdown(doc.page_content)
                st.markdown("---")

# Footer
st.markdown("---")
st.markdown("Powered by Gemma AI | © 2024 CHATWITHDOCS AI")