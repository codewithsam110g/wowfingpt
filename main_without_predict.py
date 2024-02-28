from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import yfinance as yf
import logging

from langchain_google_genai import GoogleGenerativeAI
from langchain.agents import Tool
from langchain.agents import initialize_agent, load_tools
from bs4 import BeautifulSoup
from langchain.tools import DuckDuckGoSearchRun
import os
import requests
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
import re

os.environ["GOOGLE_CSE_ID"] = "f2b97ee41733f4710"
os.environ["GOOGLE_API_KEY"] = "AIzaSyA1Qz28QhouL5-QbD93lSObWq47PDZ-6F4"


app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Configure logging
logging.basicConfig(
    filename="/var/log/wowfingpt/fastapi.log", level=logging.INFO)


def load_data(ticker: str, start_date: str, end_date: str):
    data = yf.download(ticker, start_date, end_date)
    data.reset_index(inplace=True)
    return data


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/whitepaper", response_class=HTMLResponse)
async def example(request: Request):
    return templates.TemplateResponse("whitepaper.html", {"request": request})


@app.get("/chat", response_class=HTMLResponse)
async def chat(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})


@app.get("/chat/{prompt}")
async def chat_data(prompt: str):
    response = await analyze(prompt)
    return response


def google_query(search_term):
    if "news" not in search_term:
        search_term = search_term+" stock news"
    url = f"https://www.google.com/search?q={search_term}&cr=countryIN"
    url = re.sub(r"\s", "+", url)
    return url


def get_recent_stock_news(company_name):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36'}

    g_query = google_query(company_name)
    res = requests.get(g_query, headers=headers).text
    soup = BeautifulSoup(res, "html.parser")
    news = []
    for n in soup.find_all("div", "n0jPhd ynAwRc tNxQIb nDgy9d"):
        news.append(n.text)
    for n in soup.find_all("div", "IJl0Z"):
        news.append(n.text)

    if len(news) > 6:
        news = news[:4]
    else:
        news = news
    news_string = ""
    for i, n in enumerate(news):
        news_string += f"{i}. {n}\n"
    top5_news = "Recent News:\n\n"+news_string

    return top5_news


async def analyze(query):

    search = DuckDuckGoSearchRun()

    ls = [

        Tool(
            name="DuckDuckGo Search",
            func=search.run,
            description="Use only for NSE/BSE stock ticker or recent stock-related news."
        ),
        Tool(
            name="get recent news",
            func=get_recent_stock_news,
            description="Use to fetch recent news about stocks.you should input the stock ticker name to it"
        )
    ]

    llm = GoogleGenerativeAI(model="gemini-pro")
    sys_msg = """As a trading expert and intelligence member of the market, your task is to analyze stocks/companies based on user queries. You are to provide insights using:

Fundamental Analysis: Evaluate the company's financial health, market position, competitive advantage, revenue, profit margins, and other key financial ratios. Consider recent earnings reports, management effectiveness, industry conditions, and future growth prospects.
Sentimental Analysis: Assess market sentiment towards the stock/company by analyzing news articles, investor opinions, social media trends, and overall media coverage. Determine whether the sentiment is positive, negative, or neutral.
Technical Analysis: Examine stock price movements, trading volumes, historical trends, chart patterns, and technical indicators such as moving averages, RSI (Relative Strength Index), MACD (Moving Average Convergence Divergence), and others to predict future price movements.
After conducting the requested analysis, provide a concise conclusion on whether to invest in the stock/company. This recommendation should only be given if the user requests investment advice or a comprehensive analysis involving all three aspects mentioned above.

For each query:

If the user asks specifically for fundamental analysis, provide insights based solely on the company's financial and market fundamentals.
If the user seeks sentimental analysis, focus on the prevailing sentiment and its potential impact on the stock.
If the request is for technical analysis, offer an assessment based on price trends and technical indicators.
Always ensure the analysis is up-to-date, relying on the most recent data and market trends.

Your goal is to deliver precise, concise, and actionable insights to assist users in making informed investment decisions.

Disclaimer:

Include a disclaimer noting that the recommendation is based on the current market analysis and is subject to change. Remind users to perform their own research before making trading decisions.

Your analysis output format:

User Input: [Insert user query here]

Analysis Type: [Based on user query - Determine type: Technical, Sentimental, Fundamental, Combined]

Response:

If the user asks a general question: Provide a descriptive answer explaining relevant concepts without specific recommendations.

Example: "The stock market is influenced by various factors like company performance, economic conditions, and investor sentiment. Understanding these factors can help you make informed investment decisions."

If the user asks for a specific analysis type: Deliver a concise response with the following structure:

Analysis Type: [Technical | Sentimental | Fundamental | Combined]

Key Findings: Briefly summarize the core insights from the chosen analysis type.

Recommendation: [Invest/Hold/Avoid] with brief justification based on the analysis.

Disclaimer: This analysis is based on current market conditions and is subject to change. Do your own research before investing.

Example:

User Input: Should I invest in Apple (AAPL)?**

Response:

Analysis Type: Combined

Key Findings: AAPL faces a downward trend with weak technical signals, but boasts strong financials and a leading market position. Market sentiment is slightly negative due to a broader tech sector correction.

Recommendation: Hold AAPL. Consider buying on a trend reversal or if long-term potential outweighs short-term risks.

Disclaimer: This is based on current analysis and is subject to change. Do your own research before investing.

Benefits:

Adaptability: Provides descriptive answers for general questions and concise analysis for specific requests.

Clarity: Separates key findings, recommendations, and disclaimers for easy understanding.

Conciseness: Delivers key insights in 3-5 lines for specific analysis requests.

Remember:

Avoid using external tools repeatedly for the same analysis type (technical, sentimental, fundamental) once the data is retrieved.

Ensure the analysis is based on the most recent data and market trends."""
    tools = load_tools(["google-search"], llm=llm)+ls
    conversational_memory = ConversationBufferWindowMemory(
        memory_key='chat_history',
        k=5,
        return_messages=True)
    agent = initialize_agent(tools, llm=llm, agent='chat-conversational-react-description', verbose=True,
                             handle_parsing_errors=True, max_iterations=20, early_stopping_method='generate', memory=conversational_memory)
    new_prompt = agent.agent.create_prompt(
        system_message=sys_msg,
        tools=tools)
    agent.agent.llm_chain.prompt = new_prompt

    # Use the agent to process the query
    response = await agent.arun(query)
    return response
