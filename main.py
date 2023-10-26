from langchain.chat_models import ChatOpenAI
import streamlit as st

chat_model = ChatOpenAI()

st.title('KT 인공지능 서비스')
st.title('_LLM_ 과 _LangChain_ 은 :blue[cool] :sunglasses:')

content = st.text_input('시의 주제를 제시해주세요')
st.write('입력 받은 시의 주제는', content)

if st.button('별명 생성하기'):
    with st.spinner('별명 생성 중입니다...'):
        result = chat_model.predict(content + "에 관련된 별명을 지어줘")
        st.write(result)
