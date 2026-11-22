import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import tempfile

st.set_page_config(page_title="本地RAG文档问答", layout="wide")
st.title("📄 基于Ollama+Langchain的本地RAG问答系统")

# 缓存RAG链
@st.cache_resource
def create_rag_chain(pdf_file_path):
    loader = PyPDFLoader(pdf_file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)
    embedding = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embedding)
    retriever = vectorstore.as_retriever()

    prompt = ChatPromptTemplate.from_template("""
根据下面上下文回答问题：
{context}

问题：{question}
""")
    llm = ChatOllama(model="qwen2:1.5b")
    rag_chain = (
        {"context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
         "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

uploaded_pdf = st.file_uploader("上传PDF文档", type="pdf")
rag_chain = None
if uploaded_pdf is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_pdf.read())
        tmp_path = tmp.name
    rag_chain = create_rag_chain(tmp_path)
    st.success("PDF加载完成，可以提问！")

question = st.text_input("请输入你的问题：")
if question and rag_chain:
    with st.spinner("正在检索文档并生成回答..."):
        ans = rag_chain.invoke(question)
    st.markdown("### 回答：")
    st.write(ans)
