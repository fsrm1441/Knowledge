import os
from dotenv import load_dotenv

print("Testing dotenv import...")
load_dotenv()
print("Successfully loaded dotenv!")
print(f"Python version: {os.environ.get('PYTHON_VERSION', 'Not found')}")