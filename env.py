import os
from dotenv import load_dotenv

# .env の内容を読み込む
load_dotenv()

# 環境変数を取得
api_key = os.getenv("OPENAI_API_KEY")