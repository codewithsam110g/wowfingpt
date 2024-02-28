from fastapi import FastAPI, Request
from concurrent.futures import ThreadPoolExecutor
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from datetime import datetime
from typing import List
import yfinance as yf
import pmdarima as pm
import functools
import logging
import asyncio
import httpx

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Configure logging
logging.basicConfig(
    filename="/var/log/wowfingpt/fastapi.log", level=logging.INFO)


def load_data(ticker: str, start_date: str, end_date: str):
    data = yf.download(ticker, start_date, end_date)
    data.reset_index(inplace=True)
    return data


executor = ThreadPoolExecutor(max_workers=4)


async def run_in_threadpool(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    # Use functools.partial to pass args and kwargs
    func_partial = functools.partial(func, *args, **kwargs)
    return await loop.run_in_executor(executor, func_partial)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/whitepaper", response_class=HTMLResponse)
async def example(request: Request):
    return templates.TemplateResponse("whitepaper.html", {"request": request})


@app.get("/chat", response_class=HTMLResponse)
async def chat(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})


@app.get("/prediction", response_class=HTMLResponse)
async def prediction(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/predict")
async def predict(symbol: str, days: int):
    end_date = datetime.today().strftime("%Y-%m-%d")
    start_date = '2018-01-01'

    data = await run_in_threadpool(load_data, symbol, start_date, end_date)
    auto_arima_close = await run_in_threadpool(pm.auto_arima, data['Close'], stepwise=False, seasonal=False)
    auto_arima_open = await run_in_threadpool(pm.auto_arima, data['Open'], stepwise=False, seasonal=False)
    auto_arima_low = await run_in_threadpool(pm.auto_arima, data['Low'], stepwise=False, seasonal=False)
    # Similar for auto_arima_open and auto_arima_low

    forecast_close = auto_arima_close.predict(n_periods=days)
    forecast_open = auto_arima_open.predict(n_periods=days)
    forecast_low = auto_arima_low.predict(n_periods=days)
    # Similar for forecast_open and forecast_low

    return {'close': forecast_close.tolist(), 'open': forecast_open.tolist(), 'low': forecast_low.tolist()}


async def get_response_from_api(prompt: str) -> str:
    async with httpx.AsyncClient() as client:
        response = await client.get(f"https://shahir321123.pythonanywhere.com/analyze?query={prompt}")
        response.raise_for_status()
        return response.json().get('response', 'Oops! Something went wrong.')


@app.get("/chat/{prompt}")
async def chat_data(prompt: str):
    response = await get_response_from_api(prompt)
    return response
