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
from streamlit_extras.buy_me_a_coffee import button

# connect sqlite
__import__("pysqlite3")
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

button(username="c4nd0it", floating=True, width=221)

# streamlit
st.title("KT 인공지능 서비스")
st.write("KT 인공지능 서비스는 _LLM_ 과 _LangChain_ 을 활용하여 만들어졌습니다.")
st.write("---")

uploaded_file = st.file_uploader("PDF파일을 업로드 해주세요", type=['pdf'])
st.write("---")


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


# OpenAi 키 받기
openai_key = st.text_input("Open_AI_API_KEY", type="password")

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
    embeddings_model = OpenAIEmbeddings(openai_api_key=openai_key)

    # load it into chroma
    db = Chroma.from_documents(texts, embeddings_model)

    st.header("PDF에게 질문해보세요!")
    question = st.text_input("질문을 입력하세요")
    if st.button("질문하기"):
        with st.spinner("Wait for it..."):
            llm = ChatOpenAI(model_name="gpt-4",
                             temperature=0, openai_api_key=openai_key)
            qa_chain = RetrievalQA.from_chain_type(llm,
                                                   retriever=db.as_retriever())
            result = qa_chain({"query": question})
            st.write(result["result"])


# db 저장 후 임베딩 불러오기
# https://python.langchain.com/docs/integrations/vectorstores/chroma
