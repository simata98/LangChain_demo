# from dotenv import load_dotenv
# from langchain.llms import OpenAI
# from langchain.chat_models import ChatOpenAI
from langchain.llms import CTransformers
import streamlit as st

# load_dotenv()
llm = CTransformers(
  model='llama-2-7b-chat.ggmlv3.q2_K.bin',
  model_type="llama"
)
st.title('KT 인공지능 서비스')
st.title('_LLM_ 과 _LangChain_은 :blue[cool] :sunglasses:')

content = st.text_input('시의 주제를 제시해주세요')
st.write('입력 받은 시의 주제는', content)

if st.button('시 작성 요청하기'):
    with st.spinner('시 작성 중입니다...'):
        result = llm.predict("Write a poem about" + content + ":")
        st.write(result)




