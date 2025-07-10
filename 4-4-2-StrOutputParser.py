# StrOutputParser
# ai_message.contentでよいのではないかと思うかもしれないが、Langchain Expression Language (LCEL)の構成要素として重要な役割を果たす。

from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser

output_parser = StrOutputParser()
ai_message = AIMessage("こんにちは。私はAIアシスタントです。")
output = output_parser.invoke(ai_message)
print(type(output))
print(output)