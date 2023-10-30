import sys
import tiktoken
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
# from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQA
import streamlit as st
import tempfile
import os

# connect sqlite
__import__("pysqlite3")
sys.modules["sqlite3"] = sys.modeules.pop("pysqlite3")

# streamlit
st.title("KT 인공지능 서비스")
st.write("KT 인공지능 서비스는 _LLM_ 과 _LangChain_ 을 활용하여 만들어졌습니다.")

uploaded_file = st.file_uploader("PDF파일을 업로드 해주세요", type=['pdf'])


def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_tilepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_tilepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_tilepath)
    pages = loader.load_and_split()
    return pages


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


if uploaded_file is not None:
    pages = pdf_to_document(uploaded_file)

    # Split
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )
    texts = text_splitter.split_documents(pages)

    # Embedding
    embeddings_model = OpenAIEmbeddings()

    # load it into chroma
    db = Chroma.from_documents(texts, embeddings_model)

    st.header("PDF에게 질문해보세요!")
    question = st.text_input("질문을 입력하세요")
    if st.button("질문하기"):
        with st.spinner("Wait for it..."):
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
            qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())
            result = qa_chain({"query": question})
            st.write(result["result"])


# db 저장 후 임베딩 불러오기
# https://python.langchain.com/docs/integrations/vectorstores/chroma
