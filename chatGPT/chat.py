import openai
from dotenv import load_dotenv

load_dotenv()

openai.api_key =

response = openai.Completion.create(
    model="gpt-3.5-turbo",
    prompt="say hello"
)
