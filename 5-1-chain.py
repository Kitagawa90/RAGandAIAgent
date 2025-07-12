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

