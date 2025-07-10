# ★Document Loader
# pip install langchain-community GitPython
from langchain_community.document_loaders import GitLoader

def file_filter(file_path: str) -> bool:
    # .mdxファイルのみを対象とする
    return file_path.endswith(".mdx")

loder = GitLoader(
    clone_url="https://github.com/langchain-ai/langchain",
    repo_path="./langchain",
    branch="master",
    file_filter=file_filter,
)

raw_docs = loder.load()
print(len(raw_docs))

# ★Document transformer
# pip install langchiain-text-splitters
from langchain_text_splitters import CharacterTextSplitter
text_splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=0,
)
docs = text_splitter.split_documents(raw_docs)
print(len(docs))

# ★Embedding model
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

query = "AWSのS3からデータを読み込むためのDocumentLoderはありますか？"

vector = embeddings.embed_query(query)
print(len(vector))
print(vector)

# ★Vector store
# pip install langchain-chroma
from langchain_chroma import Chroma
# Vector storeの初期化
db = Chroma.from_documents(docs, embeddings)

# Retriever作成
retriever = db.as_retriever()

#queryに近いドキュメントを検索
query = "AWSのS3からデータを読み込むためのDocumentLoderはありますか？"

context_docs = retriever.invoke(query)

print(f"len={len(context_docs)}")
first_doc = context_docs[0]
print(f"metadata = {first_doc.metadata}")
print(first_doc.page_content)

# ★LCEL(Chain)

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

#プロンプト作成
prompt = ChatPromptTemplate.from_template('''\
以下の文脈だけを踏まえて質問に回答してください。

文脈:"""
{context}
"""

質問: {question}
''')

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Chainの作成(入力がretrieverとprompt両方に渡されるイメージ)
# RunnablePassthrough は「入力をそのまま返すだけ」っちゅうシンプルな機能
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

output = chain.invoke(query)
print(output)