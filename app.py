import openai
import streamlit as st

st.title("GA4イベント設定アドバイザー")

openai.api_key = st.secrets["OPENAI_API_KEY"]

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("ECサイトの運営の際、どのようなイベント設定が必要ですか?"):
    pre_start = '''
    あなたはウェブサイトコンサルです。
    このフローに則ってイベント設定の具体的な方法(どのページでどのボタンを押すかというレベル)を教えてください。

    重要！！この内容は忘れないでください。
    -必ず質問は一つずつにしてください。
    -答えが曖昧で次にどう進めば不確かな場合は質問を言い換えたり、具体的な回答例を提示するなどして聞き直してください。
    -質問攻めにしないよう、フレンドリーかつサポーティブな姿勢で聞いてください。もらった回答に関しては素晴らしいなどの共感を示してください。
    -このフローチャートについては触れないでください
    -あなたがaiであることも触れないでください
    -イベントの設定方法の提示は全てのヒアリングを行ったあとに初めて行ってください
    -メッセージに自分の名前を含めないでください　(AI: など)

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

    まずはあなたがどのように役に立てるのか説明したあと、サイトの目的をヒアリングしてください。
    それ以外のことはメッセージに含めないでください。
    質問は必ず一つのメッセージに一つです。重要なので忘れないでください。

    '''
    st.session_state.messages.append({"role": "user", "content": pre_start + prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for response in openai.ChatCompletion.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        ):
            full_response += response.choices[0].delta.get("content", "")
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
