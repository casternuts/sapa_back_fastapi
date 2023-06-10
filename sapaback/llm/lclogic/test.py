from langchain.llms import OpenAI
from sapaback.core.config import settings
llm = OpenAI(temperature=0.9,openai_api_key=settings.OPEN_API_KEY)
text = "What would be a good company name for a company that makes colorful socks?"
print(llm(text))