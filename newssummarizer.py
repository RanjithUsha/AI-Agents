from dotenv import load_dotenv

load_dotenv()

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain_tavily import TavilySearch

search = TavilySearch(max_results=3)

llm = ChatMistralAI(model_name="ministral-8b-2512")

prompt_template = ChatPromptTemplate.from_template(
    "You are a helpful news summarizer. Summarize the following search results in bullet points and then combine them"
    "into a concise paragraph.\n\n{news_articles}"
)

run_chain = prompt_template | llm | StrOutputParser()

news_result = search.invoke({"query": "latest news on AI"})

result = run_chain.invoke({"news_articles": news_result})

print(result)
