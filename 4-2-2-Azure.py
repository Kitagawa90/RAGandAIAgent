from langchain.chat_models import azure_openai
from langchain.schema import HumanMessage
from env import azure_openai_api_key, azure_openai_api_base

client = azure_openai(
    openai_api_base=azure_openai_api_base,
    openai_api_key=azure_openai_api_key,
)

response = client([HumanMessage(content="こんにちは、調子はどう？")])

# 結果出力
print(response[0].content)