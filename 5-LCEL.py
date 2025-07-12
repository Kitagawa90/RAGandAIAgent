# LCEL

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "ユーザが入力した料理のレシピを考えてください。"),
        ("human", "{dosh}"),
    ]
)

model = ChatOpenAI(model = "gpt-4o-mini", temperature = 0)

output_parser = StrOutputParser()

# ★LCELを使用しない方法
#    それぞれinvokeで呼び出す。また前のクラスの実行結果を次のクラスの引数に渡す。
prompt_value = prompt.invoke({"dosh": "カレー"})
ai_message = model.invoke(prompt_value)
output = output_parser.invoke(ai_message)

print(output)

# ★LCELを使用する方法
#    各クラスを「|」でつなげて、invokeを一度だけ呼び出す。
#    上の書き方と同等になる
chain = prompt | model | output_parser
output = chain.invoke({"dosh": "カレー"})

# ★Stream(ストリーミングで実行)
chain = prompt | model | output_parser
for chunk in chain.stream({"dosh": "カレー"}):
    print(chunk, end="", flush=True)

# ★batch(複数処理をまとめて実行)
chain = prompt | model | output_parser
outputs = chain.batch({"dish":"カレー"},{"dish":"うどん"})
print(outputs)

# ★runnableとrunnableを連結
cot_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "ユーザの質問にステップバイステップで回答してください。"),
        ("human", "{question}"),
    ]
)
cot_chain = cot_prompt | model | output_parser

summarize_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "ステップバイステップで考えた回答から結論だけ抽出してください。"),
        ("human", "{text}"),
    ]
)
summarize_chain = summarize_prompt | model | output_parser
#合体
cot_summarize_chain = cot_chain | summarize_chain
cot_summarize_chain.invoke({"question": "10 + 2 * 3"})

# ★任意の関数をrunnableに変換
from langchain_core.runnables import RunnableLambda
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        ("human", "{input}"),
    ]
)

def upper(text: str) -> str:
    return text.upper()

chain = prompt | model | output_parser | RunnableLambda(upper)

output = chain.invoke({"input": "hello"})
print(output)  # 出力: HELLO! HOW CAN I HELP YOU TODAY?

# ★Chainデコレータ
from langchain_core.runnables import chain

@chain
def lower(text: str) -> str:
    return text.lower()

chain = prompt | model | output_parser | lower

# ★自動変換
#    任意の関数を「|」で接続することで自動的にrunnnableに変換される
def upper2(text: str) -> str:
    return text.upper()
chain = prompt | model | output_parser | upper2

# ★「|」で接続したする場合、関数の引数戻り値の方を合わせる必要あり

# ★runnnableParallelで複数のrunnnableを並列につなげる
optimistic_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "あなたは楽観主義者です。ユーザの入力に対して楽観的な意見をください。"),
        ("human", "{topic}"),
    ]
)
optimistic_chain = optimistic_prompt | model | output_parser

pessimistic_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "あなたは悲観主義者です。ユーザの入力に対して悲観的な意見をください。"),
        ("human", "{topic}"),
    ]
)
pessimistic_chain = pessimistic_prompt | model | output_parser

import pprint
from langchain_core.runnables import RunnableParallel
parallel_chain = RunnableParallel(
    {
        "optimistic_option": optimistic_chain,
        "pessimistic_option": pessimistic_chain,
    }
)

output = parallel_chain.invoke({"topic": "生成AIの進化について"})
pprint.pprint(output)

# 並列で取得した意見を一つにまとめる
synthesize_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "あなたは客観的AIです。２つの意見をまとめてください。"),
        ("human", "楽観的意見:{optimistic_option}\n悲観的意見:{pessimistic_option}\nまとめ: "),
    ]
)

synthesize_chain = (
    RunnableParallel(
        {
            "optimistic_option": optimistic_chain,
            "pessimistic_option": pessimistic_chain,
        }
    )
    | synthesize_prompt
    | model
    | output_parser
)

# 以下のコードは上記と同等
synthesize_chain = (
    {
        "optimistic_option": optimistic_chain,
        "pessimistic_option": pessimistic_chain,
    }
    | synthesize_prompt
    | model
    | output_parser
)

# ★itemgetter
from operator import itemgetter
topic_getter = itemgetter("topic")
topic = topic_getter({"topic": "生成AIの進化について"})
print(topic)  # 出力: 生成AIの進化について

synthesize_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "あなたは客観的AIです。２つの意見をまとめてください。"),
        ("human", "楽観的意見:{optimistic_option}\n悲観的意見:{pessimistic_option} "),
    ]
)
synthesize_chain = (
    RunnableParallel(
        {
            "optimistic_option": optimistic_chain,
            "pessimistic_option": pessimistic_chain,
            "topic": itemgetter("topic"), # Runnableに自動変換される
        }
    )
    | synthesize_prompt
    | model
    | output_parser
)

output = synthesize_chain.invoke({"topic": "生成AIの進化について"})
print(output)

# ★RunnablePassthroughで入力を受け流す
#    Tavilyを使用するため環境変数にAPIキーを設定する必要あり「TAVILY_API_KEY」
#    pip install tavily-python

prompt = ChatPromptTemplate.from_template('''¥
以下の文脈だけを踏まえて質問に回答してください。
文脈:"""
{context}
"""

質問: {question}
''')

from langchain_community.retrievers import TavilySearchAPIRetriever
retriever = TavilySearchAPIRetriever(k=3)

from langchian_core.runnables import RunnablePassthrough
chain = (
    {"context": retriever, "question": RunnablePassthrough()} # RunnableParallelで自動変換される
    | prompt
    | model
    | output_parser
)

output = chain.invoke("今日の天気は？")
print(output)

# ★retrieverの検索結果も全体の出力に含めたい場合
#    question、context、RunnablePassthrough.assignで作成したanswerが出力される
chain = {
    "question": RunnablePassthrough(),
    "context": retriever,
} | RunnablePassthrough.assign(answer = prompt | model | StrOutputParser())
output = chain.invoke("今日の天気は？")
pprint.pprint(output)

#以下コードは上記コードと等価
chain = RunnableParallel(
    {
        "question": RunnablePassthrough(),
        "context": retriever,
    }
).assign(answer = prompt | model | StrOutputParser())

# pickでdictの一部だけをピックアップすることもできる
chain = (
    RunnableParallel(
        {
            "question": RunnablePassthrough(),
            "context": retriever,
        }
    )
    .assign(answer = prompt | model | StrOutputParser())
    .pick(["context", "answer"])  # contextとanswerだけをピックアップ
)

# ★astream_eventsで中間のイベントを取得する
async def astream_events_example():
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}  # RunnableParallelで自動変換される
        | prompt
        | model
        | StrOutputParser()
    )

    async for event in chain.astream_events("今日の天気は？",version="v2"):
        print(event,fluch=True)
    # 中間の値を取得することもできる
    async for event in chain.astream_events("今日の天気は？",version="v2"):
        event_kind = event["event"]
        
        if event_kind == "on_retriever_end":
            print("===検索結果===")
            documents = event["data"]["output"]
            for document in documents:
                print(document)
        elif event_kind == "on_parser_start":
            print("===最終出力===")
        elif event_kind == "on_parser_stream":
            chunk = event["data"]["chunk"]
            print(chunk, end="", flush=True)

# ★SQLiteでの会話履歴管理
from langchain_community.chat_message_history import SQLiteChatMessageHistory

def respond(session_id: str, human_message: str) -> str:
    chat_message_history = SQLiteChatMessageHistory(
        session_id=session_id,
        connection="sqlite:///sqlite.db"
    )
    messages = chat_message_history.get_messages()
    
    ai_message = chain.invoke(
        {
            "chat_history": messages,
            "input": human_message
        }
    )
    
    chat_message_history.add_user_message(human_message)
    chat_message_history.add_ai_message(ai_message)

# 以下のコードを実行することで会話の履歴を管理できる
from uuid import uuid4

session_id = uuid4().hex
output1 = respond(
    session_id=session_id,
    human_message="こんにちは！私はジョンといいます！"
)
print(output1)

output2 = respond(
    session_id=session_id,
    human_message="私名前が分かりますか？"
)

print(output2)

# SQLite以外にも以下で管理できる
# inMemoryMessageHistory:インメモリ
# SQLChatMessageHistory:SQLAlchemyがサポートする各種RDB
# RedisChatMessageHistory:Redis
# DynamoDBChatMessageHistory:DynamoDB
# CosmosDBChatMessageHistory:CosmosDB
# MomentoDBChatMessageHistory:MongoDB