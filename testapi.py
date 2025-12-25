from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# تنظیمات دستی برای تست
API_KEY = os.getenv("OPENAI_API_KEY") 
BASE_URL = "https://api.avalai.ir/v1"

# مدل را موقتاً به یک مدل مطمئن تغییر می‌دهیم
MODEL = "glm-4.6" 
# یا اگر حتما GLM می‌خواهید: "glm-4-flash"

print(f"Testing connection to {BASE_URL}...")
print(f"Model: {MODEL}")

try:
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": "سلام! وضعیت سیستم چطور است؟"}
        ],
        temperature=0,
    )
    
    print("✅ Success!")
    print("Response:", response.choices[0].message.content)
    
except Exception as e:
    print("\n❌ Error Occurred:")
    print(e)