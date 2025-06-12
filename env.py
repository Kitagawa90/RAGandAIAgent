import os
from dotenv import load_dotenv

# .env の内容を読み込む
load_dotenv()

# 環境変数を取得
api_key = os.getenv("OPENAI_API_KEY")

azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_openai_api_base = os.getenv("AZURE_OPENAI_API_BASE")