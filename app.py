import os
from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
import tempfile
import time

# Load API keys from environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')

# Custom CSS to style the app
st.markdown("""
<style>
    .main {
        background-color: #051622#1ba098#deb992;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton > button {
        background-color:  #3498db;
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        transition: background-color 0.3s;
    }
    .stButton > button:hover {
        background-color: black;
    }
    .stTextInput > div > div > input {
        background-color: black;
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 0.5rem;
    }
    .stExpander {
        background-color: black;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# App header
st.title("📚 DocuChat AI")
st.subheader("Your Intelligent Multi-Document Assistant")

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

# Vector store creation
@st.cache_resource
def create_vector_store(_docs):
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(_docs)
    vectors = FAISS.from_documents(final_documents, embeddings)
    return vectors

# Sidebar for app controls
with st.sidebar:
    st.header("📋 Control Panel")
    uploaded_files = st.file_uploader("Upload PDF documents", type="pdf", accept_multiple_files=True)
    
    if uploaded_files:
        with st.spinner("Processing documents..."):
            all_docs = []
            for uploaded_file in uploaded_files:
                # Save the uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name

                # Load the PDF
                loader = PyPDFLoader(tmp_file_path)
                docs = loader.load()
                all_docs.extend(docs)

                # Clean up the temporary file
                os.unlink(tmp_file_path)

            # Create vector store
            st.session_state.vectors = create_vector_store(all_docs)

        st.success(f"{len(uploaded_files)} document(s) processed successfully!")
    
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
    if "vectors" not in st.session_state:
        st.error("⚠️ Please upload at least one document first using the sidebar.")
    elif not prompt1:
        st.warning("⚠️ Please enter a question.")
    else:
        with st.spinner("🧠 Thinking..."):
            retriever = st.session_state.vectors.as_retriever()
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": PROMPT}
            )
            
            start = time.process_time()
            response = qa_chain({"query": prompt1})
            end = time.process_time()
            
            st.session_state.query_count += 1
        
        st.markdown("### 📝 Answer:")
        st.info(response['result'])
        st.markdown(f"*Response time: {end - start:.2f} seconds*")

        with st.expander("📚 View Source Documents"):
            for i, doc in enumerate(response["source_documents"]):
                st.markdown(f"**Document {i+1}:**")
                st.markdown(doc.page_content)
                st.markdown("---")

# # Footer
# st.markdown("---")
# st.markdown("Powered by Gemma AI | © 2024 CHATWITHDOCS AI")