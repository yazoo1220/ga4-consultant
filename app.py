import streamlit as st
from langchain. chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import (
    HumanMessage,
)
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import openai
from typing import Any, Dict, List

st.header("Balencer GA4コンサル")
st.subheader("イベント設定・レポート分析")


# event setting consultation

if "starter" not in st.session_state:
    st.session_state.starter = True
                                  
def get_state(): 
     if "state" not in st.session_state: 
         st.session_state.state = {"memory": ConversationBufferMemory(memory_key="chat_history")} 
     return st.session_state.state 
state = get_state()

setting_tab, analysis_tab = st.tabs(["イベント設定", "レポート分析"])

prompt = PromptTemplate(
    input_variables=["chat_history","input"], 
    template='Based on the following chat_history, Please reply to the question in format of markdown. history: {chat_history}. question: {input}'
)

user_input = setting_tab.text_input("You: ",placeholder = "Ask me anything ...")
ask = setting_tab.button('ask',type='primary')
reset=st.button('reset')
if reset:
    st.session_state.starter = True
    
setting_tab.markdown("----")


class SimpleStreamlitCallbackHandler(BaseCallbackHandler):
    """ Copied only streaming part from StreamlitCallbackHandler """
    
    def __init__(self) -> None:
        self.tokens_area = st.empty()
        self.tokens_stream = ""
        
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        self.tokens_stream += token
        self.tokens_area.markdown(self.tokens_stream)

handler = SimpleStreamlitCallbackHandler()

if st.session_state.starter: 
    pre_start = '''
    あなたはウェブサイトコンサルです。
    このフローに則ってイベント設定の具体的な方法(どのページでどのボタンを押すかというレベル)を教えてください。

    重要！！この内容は忘れないでください。
    -必ず質問は一つずつにしてください。
    -答えが曖昧で次にどう進めば不確かな場合は質問を言い換えたり、具体的な回答例を提示するなどして聞き直してください。
    -質問攻めにしないよう、フレンドリーかつサポーティブな姿勢で聞いてください。もらった回答に関しては素晴らしいなどの共感を示してください。
    -このフローチャートについては触れないでください
    -あなたがaiであることも触れないでください

    graph TD
        A[サイトの目的は何ですか？]
        A -->|製品の販売| B[提供している主な製品は何ですか？]
        A -->|情報提供| C[ユーザーが最も頻繁に閲覧するコンテンツは何ですか？]
        A -->|ブランドの認知度向上| D[主なブランドイメージやメッセージは何ですか？]
        B --> E[ユーザーが最も頻繁に行う行動は何ですか？]
        C --> E
        D --> E
        E -->|商品の閲覧・購入| F[最も重要と考えているコンバージョンは何ですか？]
        E -->|問い合わせ・情報探求| G[最も重要と考えているコンバージョンは何ですか？]
        F --> H[成功を測定するための主な指標は何ですか？]
        G --> H
        H --> I[現在使用している主な広告戦略は何ですか？]
        I --> J[重要なユーザーセグメントは何ですか？]

    まずはあなたがどのように役に立てるのか説明したあと、サイトの目的をヒアリングすることから始めます。
    質問は必ず一つずつです。重要なので忘れないでください。
    開始。

    '''

    chat = ChatOpenAI(streaming=True, model_name='gpt-4', temperature=0.5)
    conversation = ConversationChain(
        llm=chat, 
        prompt=prompt,
        memory=state['memory']            
    )
    res = conversation.predict(input=pre_start, callbacks=[handler])
    st.session_state.starter = False

if ask:
    res_box = setting_tab.empty()
    with st.spinner('typing...'):
        report = []
        chat = ChatOpenAI(streaming=True, temperature=0.9)
        conversation = ConversationChain(
            llm=chat, 
            prompt=prompt,
            memory=state['memory']            
        )
        res = conversation.predict(input=user_input, callbacks=[handler])
        user_input = ''


# report analysis

from streamlit_chat import message
import os
import pexpect

analysis_tab.header("GA4レポート分析")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []
    
    
from langchain.agents import load_tools, initialize_agent, AgentType, Tool, tool
import pandas as pd
from langchain.agents import create_pandas_dataframe_agent
from langchain.memory import ConversationBufferMemory
from langchain import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import (
    HumanMessage,
)
from typing import Any, Dict, List

df = pd.DataFrame([])
data = analysis_tab.file_uploader(label='GA4のレポートをアップロード(csv)', type='csv')



header_num = analysis_tab.number_input(label='Header position',value=0)
index_num = analysis_tab.number_input(label='Index position',value=2)
index_list = [i for i in range(index_num)]

if data:
    df = pd.read_csv(data,header=header_num,index_col=index_list)
    analysis_tab.dataframe(df)

def get_text():
    input_text = analysis_tab.text_input("You: ", "レポートから具体的なアクションプランを提示してください", key="input")
    return input_text

def get_state(): 
     if "state" not in st.session_state: 
         st.session_state.state = {"memory": ConversationBufferMemory(memory_key="chat_history")} 
     return st.session_state.state 
state = get_state()

prompt = PromptTemplate(
    input_variables=["chat_history","input"], 
    template='You are a very helpful consultant who knows everything about GA4 (Google analytics). Based on the following chat_history, Please reply to the question in format of markdown. history: {chat_history}. question: {input}'
)

class SimpleStreamlitCallbackHandler(BaseCallbackHandler):
    """ Copied only streaming part from StreamlitCallbackHandler """
    
    def __init__(self) -> None:
        self.tokens_area = st.empty()
        self.tokens_stream = ""
        
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        self.tokens_stream += token
        self.tokens_area.markdown(self.tokens_stream)

ask_button = ""

if df.shape[0] > 0:
    agent = create_pandas_dataframe_agent(ChatOpenAI(temperature=0, model_name='gpt-4'), df, memory=state['memory'], verbose=True, return_intermediate_steps=True)
    user_input = get_text()
    ask_button = analysis_tab.button('ask')
else:
    pass


import json
import re
from collections import namedtuple
AgentAction = namedtuple('AgentAction', ['tool', 'tool_input', 'log'])

def format_action(action, result):
    action_fields = '\n'.join([f"{field}: {getattr(action, field)}"+'\n' for field in action._fields])
    return f"{action_fields}\nResult: {result}\n"

if ask_button:
#     res_box = st.empty()
    st.write("Input:", user_input)
    with st.spinner('typing...'):
        prefix = f'すべて日本語で回答してください'
        handler = SimpleStreamlitCallbackHandler()
        response = agent({"input":user_input}) #,"callbacks":handler})
        
        
        actions = response['intermediate_steps']
        actions_list = []
        for action, result in actions:
            text = f"""Tool: {action.tool}\n
               Input: {action.tool_input}\n
               Log: {action.log}\nResult: {result}\n
            """
            text = re.sub(r'`[^`]+`', '', text)
            actions_list.append(text)
            
        answer = json.dumps(response['output'],ensure_ascii=False).replace('"', '')
        with st.expander('ℹ️ 詳細を見る', expanded=False):
            st.write('\n'.join(actions_list))
            
        st.session_state.past.append(user_input)
        st.session_state.generated.append(answer)
        
if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
