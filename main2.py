from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import streamlit as st

load_dotenv()
chat_model = ChatOpenAI()

st.title('KT 인공지능 서비스')
st.title('_LLM_ 과 _LangChain_ 은 :blue[cool] :sunglasses:')

content = st.text_input('당신의 멋진 이름을 입력해주세요!')
st.write('당신의 이름은', content)

if st.button('별명 생성하기'):
    with st.spinner('별명 생성 중입니다...'):
        result = chat_model.predict(content + "에 관련된 웃긴 별명을 지어줘. 예를 들어 '이정형'을 입력 받으면, '정형외과' 이런 식으로")
        st.write(result)
