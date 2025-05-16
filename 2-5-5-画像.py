# 参考 https://platform.openai.com/docs/guides/images-vision?api-mode=responses

import env
from openai import OpenAI

client = OpenAI()

image_url = "https://raw.githubusercontent.com/yoshidashingo/langchain-book/main/assets/cover.jpg"

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "画像を説明してください。"},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]
        },
    ],
)

# レスポンス JSON をそのままプリント
print(response.choices[0].message.content)
