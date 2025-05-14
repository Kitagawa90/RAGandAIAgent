import env
from openai import OpenAI

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
        }
    ]
)
print(response.to_json(indent=2))





