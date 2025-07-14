# Advanced RAG

# pip install langchain-core langchain-openai langchain-community GitPython langchain-chroma tavily-python

# 検索対象としてLangChainの公式ドキュメントを使用する
# モデル実行のトレースはLangsmithで行う
# https://smith.langchain.com/

# ★GitLoaderを使用して、GitHubリポジトリからドキュメントをロードする
from langchain_community.document_loaders import GitLoader

def file_filter(file_path: str) -> bool:
    return file_path.endswith(".mdx")

loder = GitLoader(
    clone_url="https://github.com/langchain-ai/langchain",
    repo_path="./langchain",
    branch="master",
    file_filter=file_filter,
)

documents = loder.load()
print(len(documents))

# ★OpenAIのEmbeddingモデルを使用して、ドキュメントをベクトル化。Chromaにインデクシング。
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma.from_documents(documents, embeddings)

# ★簡単な実装例
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_template('''\
以下の文脈だけを踏まえて質問に回答してください。

文脈:"""
{context}
"""

質問: {question}
''')

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

retriever = db.as_retriever()

chain = {
    "question": RunnablePassthrough(),
    "context": retriever,
} | prompt | model | StrOutputParser()

# output = chain.invoke("LangChainの概要を教えて")
# print(output)

# ★HyDE

hypothetical_prompt = ChatPromptTemplate.from_template('''\
次の質問に回答する一文を書いてください。

質問:{question}
''')
# 仮説的な回答を生成する
hypothetical_chain = hypothetical_prompt | model | StrOutputParser()

hyde_rag_chain = {
    "question": RunnablePassthrough(),
    "context": hypothetical_chain | retriever, # 仮説的な回答を生成するChainの出力をRetrieverの入力として使用
} | prompt | model | StrOutputParser()

# output = hyde_rag_chain.invoke("LangChainの概要を教えて")
# print(output)

# ★複数の検索クエリの生成
from pydantic import BaseModel, Field

class QueryGenerationOutput(BaseModel):
    queries: list[str] = Field(..., description="検索クエリのリスト")

query_generation_prompt = ChatPromptTemplate.from_template('''\
質問に対してベクターデータベースから関連文書を検索するために、
3つの異なる検索クエリを生成してください。
距離ベースの類似検索の限界を克服するために、
ユーザの質問に対して複数の視点を提供することが目標です。

質問: {question}
''')

query_generation_chain = (
    query_generation_prompt
    | model.with_structured_output(QueryGenerationOutput)  # 構造化出力を使用
    | (lambda x: x.queries )  # queriesフィールドを抽出
)

multi_query_rag_chain = {
    "question": RunnablePassthrough(),
    "context": query_generation_chain | retriever.map(),  # 複数の検索クエリを使用してRetrieverを呼び出す
} | prompt | model | StrOutputParser()

output = multi_query_rag_chain.invoke("LangChainの概要を教えて")
print(output)