from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["dish_name"],
    template="""以下の料理のレシピを考えてください。
    料理名: {dish_name}""",
    )

prompt_value = prompt.invoke({"dish_name":"カレー"})
print(prompt_value)


# ■ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "ユーザが入力した料理のレシピを教えてください。"),
        ("human", "{dish_name}"),
    ]
)

# ■会話履歴を含める場合はMessagesOlaceholderを使う
prompt_value = prompt.invoke({"dish_name":"カレー"})
print(prompt_value)

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
    ]
)

prompt_value = prompt.invoke(
    {
        "chat_history": [
            HumanMessage(content="こんにちは！私はジョンといいます！"),
            AIMessage("こんにちは、ジョンさん！どのようにお手伝いできますか？"),
        ],
        "input": "私の名前がわかりますか？"
    }
)
print(prompt_value)

# ■LangSmith
from langsmith import Client

# Langsmithからプロンプトをダウンロード
client = Client()
prompt= client.pull_prompt("oshima/recipe")

prompt_value = prompt.invoke({"dish_name":"カレー"})
print(prompt_value)

# ■マルチモーダルモデル
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "user",
            [
                {"type": "text", "text": "画像を説明してください。"},
                {"type": "image_url", "image_url": {"url": "{image_url}"}},
            ],
        ),
    ]
)

image_url = "https://raw.githubusercontent.com/yoshidashingo/langchain-book/main/assets/cover.jpg"

prompt_value = prompt.invoke({"image_url": image_url})

# 呼び出し
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
ai_message = model.invoke(prompt_value)
print(ai_message.content)