# Chain
#   Chainを使うと処理を連鎖的につなぐことが可能
#   |でつなぐ
#   chain = prompt | model
#   ai_message = chain.invoke({"dish": "カレー"})
#   戻り値が次の処理の入力になる
#   実際使う場合はwith_structured_outputが楽
#   chain = prompt | model.with_structured_output(Recipe)
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "ユーザが入力した料理のレシピを考えてください。"),
        ("human", "{dish}"),
    ]
)

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

#★ここが大事
chain = prompt | model
ai_message = chain.invoke({"dish": "カレー"})

print(ai_message.content)

#★StrOutPutParserを連鎖に追加
chain = prompt | model | StrOutputParser()
output = chain.invoke({"dish": "カレー"})
print(output)

#★PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

class Recipe(BaseModel):
    ingredients: list[str] = Field(description="ingredients of the dish")
    steps: list[str] = Field(description="steps to make the dish")
#パーサー作成
output_parser = PydanticOutputParser(pydantic_object=Recipe)
#プロンプト作成
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "ユーザが入力した料理のレシピを考えてください。\n\n{format_instructions}"),
        ("human", "{dish}"),
    ]
)
prompt_with_format_instructions = prompt.partial(
    format_instructions=output_parser.get_format_instructions()
)
#モデル作成
model = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind(
    response_format={"type": "json_object"}
)
#chain作成
chain = prompt_with_format_instructions | model | output_parser
#chainを実行
receipe = chain.invoke({"dish": "カレー"})
print(type(receipe))
print(receipe)

#★with_structured_output
#with_structured_outputを使うと、PydanticOutputParserでのコードをより簡潔に書くことができる。
#ただし使えないモデルも存在するためそこは注意が必要。
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "ユーザが入力した料理のレシピを考えてください。"),
        ("human", "{dish}"),
    ]
)

#★ここが大事
chain = prompt | model.with_structured_output(Recipe)

receipe = chain.invoke({"dish": "カレー"})
print(type(receipe))
print(receipe)