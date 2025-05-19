# 3-3-1-プロンプトのテンプレート化

from openai import OpenAI

client = OpenAI()

prompt= '''\
以下の料理のレシピを考えてください。
料理名: """
{dish}
"""
'''

def generate_recipe(dish) -> str:
    response = client.chat.completions.create(
        model = "gpt-4o-mini",
        messages = [
            {"role": "user", "content": prompt.format(dish=dish)}
        ],
    )
    return response.choices[0].message.content

recipe = generate_recipe("カレーライス")
print(recipe)

def generate_recipe2(dish) -> str:
    response = client.chat.completions.create(
        model = "gpt-4o-mini",
        messages= [
            {"role": "system", "content": "ユーザが入力した料理のレシピを考えてください。"},
            {"role": "user", "content": f"{dish}"}
        ]
    )
    return response.choices[0].message.content

recipe2 = generate_recipe2("カレーライス")
print(recipe2)