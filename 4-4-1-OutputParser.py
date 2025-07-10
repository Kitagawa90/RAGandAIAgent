# LLMの出力をPydanticモデルに変換するためのコード例
#   Recipeクラス定義をもとに出力形式を指定する文字列が自動的に作られる
#   LLM出力を簡単にRecipeクラスのインスタンスに変換できる
from pydantic import BaseModel, Field

class Recipe(BaseModel):
    ingredients: list[str] = Field(description="ingredients of the dish")
    steps: list[str] = Field(description="steps to make the dish")

from langchain_core.output_parsers import PydanticOutputParser

## Recipeクラスを与えてPydanticOutputParserを作成
output_parser = PydanticOutputParser(pydantic_object=Recipe)

# PydanticOutputParserからプロンプトに含める出力形式の説明文を作成
format_instructions = output_parser.get_format_instructions()
print(format_instructions)

# 出力結果
# The output should be formatted as a JSON instance that conforms to the JSON schema below.

# As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}
# the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.

# Here is the output schema:
# ```
# {"properties": {"ingredients": {"description": "ingredients of the dish", "items": {"type": "string"}, "title": "Ingredients", "type": "array"}, "steps": {"description": "steps to make the dish", "items": {"type": "string"}, "title": "Steps", "type": "array"}}, "required": ["ingredients", "steps"]}
# ```

# LLMがこの形式に沿った応答を返すようにコードを記述
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "ユーザが入力した料理のレシピを考えてください。\n\n"
            "{format_instructions}",
        ),
        ("human", "{dish}"),
    ]
)

prompt_with_format_instructions = prompt.partial(
    format_instructions=format_instructions
)

prompt_value = prompt_with_format_instructions.invoke({"dish": "カレー"})
print("---role:system---")
print(prompt_value.messages[0].content)
print("---role:user---")
print(prompt_value.messages[1].content)

#上記で出力したテキストを入力としてLLMを実行

from langchain_openai import ChatOpenAI
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
ai_message = model.invoke(prompt_value)
print(ai_message.content)

# インスタンスに変換
recipe = output_parser.invoke(ai_message)
print(type(recipe))
print(recipe)
