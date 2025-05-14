import env
from openai import OpenAI

# stream=Trueを設定すると、ストリーミングモードでレスポンスを受け取ることができます。

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "こんにちは！私はジョンといいます！！"
        },
    ],
    stream=True
)

for chunk in response:
    content = chunk.choices[0].delta.content
    if content is not None:
        print(content, end="", flush=True)