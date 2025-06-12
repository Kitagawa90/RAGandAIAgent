from langchain_openai import OpenAI

model = OpenAI(model="gpt-4o-mini", temperature=0)
output = model.invoke("自己紹介してください。")
print(output)

