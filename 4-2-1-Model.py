#ChatModel

from langchain_openai import OpenAI

model = OpenAI(model="gpt-4o-mini", temperature=0)
output = model.invoke("自己紹介してください。")
print(output)


## ChatModelを使った会話の例
from langchain_core.messages import AIMessage, HumanMessage,SystemMessage
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

messages = [
    SystemMessage("You are a helpful assistant."),
    HumanMessage("こんにちは！私はジョンといいます！"),
    AIMessage(content="こんにちは、ジョンさん！どのようにお手伝いできますか？"),
    HumanMessage("私の名前がわかりますか？"),
]

ai_message = model.invoke(messages)
print(ai_message.content)


#ストリーミング
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