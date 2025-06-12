from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

messages = [
    SystemMessage("You are a helpful assistant."),
    HumanMessage("こんにちは！"),
]

for chunk in model.stream(messages):
    print(chunk.content, end="", flush=True)

# Callback機能を使ってストリーミングを実装することもできる
# LLMの開始(on_llm_start)、新しいトークンの生成(on_llm_new_token)、LLMの終了(on_llm_end)などのタイミングで任意の処理を行える
