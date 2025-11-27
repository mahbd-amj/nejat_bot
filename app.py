import streamlit as st
import os
import tempfile

# --- تلاش برای نصب خودکار در صورت خطا (Plan B) ---
try:
    from langchain_groq import ChatGroq
    from langchain_community.tools import DuckDuckGoSearchResults
    from langchain_community.document_loaders import PyPDFLoader, TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain.chains import RetrievalQA
    from langchain.agents import initialize_agent, Tool, AgentType
except ImportError as e:
    st.error(f"خطا در کتابخانه‌ها: {e}")
    st.stop()

# --- تنظیمات صفحه ---
st.set_page_config(page_title="Nejat Bot", page_icon="⛑️", layout="wide")
st.markdown("<style>.stApp {direction: rtl; text-align: right;}</style>", unsafe_allow_html=True)

st.title("⛑️ دستیار هوشمند نجات و امداد")

# --- سایدبار ---
with st.sidebar:
    st.header("تنظیمات")
    api_key = st.text_input("Groq API Key", type="password")
    uploaded_files = st.file_uploader("آپلود فایل (PDF/TXT)", accept_multiple_files=True)
    process = st.button("پردازش منابع")

# --- توابع ---
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def process_files(files):
    if not files: return None
    documents = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    
    with st.status("در حال خواندن فایل‌ها...", expanded=True):
        for file in files:
            with tempfile.NamedTemporaryFile(delete=False) as tf:
                tf.write(file.read())
                path = tf.name
            try:
                if file.name.endswith('.pdf'): loader = PyPDFLoader(path)
                else: loader = TextLoader(path)
                documents.extend(loader.load())
            except Exception as e:
                st.error(f"Error reading {file.name}: {e}")
            finally:
                os.remove(path)
        
        st.write("در حال تبدیل به پایگاه دانش...")
        chunks = text_splitter.split_documents(documents)
        db = FAISS.from_documents(chunks, get_embeddings())
        return db

if "db" not in st.session_state: st.session_state.db = None
if process and uploaded_files: st.session_state.db = process_files(uploaded_files)

# --- چت ---
if "messages" not in st.session_state: st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

prompt = st.chat_input("سوال خود را بپرسید...")

if prompt:
    if not api_key:
        st.warning("لطفاً API Key را وارد کنید.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        
        with st.chat_message("assistant"):
            try:
                llm = ChatGroq(api_key=api_key, model_name="llama3-8b-8192")
                tools = [Tool(name="Search", func=DuckDuckGoSearchResults().run, description="Search internet")]
                
                if st.session_state.db:
                    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=st.session_state.db.as_retriever())
                    tools.append(Tool(name="KnowledgeBase", func=qa.run, description="Search uploaded files"))
                
                agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, handle_parsing_errors=True)
                
                response = agent.run(f"Answer in Persian (Farsi): {prompt}")
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(str(e))
