from dotenv import load_dotenv
import os

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")

print(f"Here is my google key {google_api_key}")



