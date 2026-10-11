"""Shared tools and LLM setup for CLI and Streamlit agents."""

from __future__ import annotations

import os

import requests
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_mistralai import ChatMistralAI
from langchain_tavily import TavilySearch

load_dotenv()

CITY_ALIASES = {
    "bangalore": "Bengaluru,IN",
    "bengaluru": "Bengaluru,IN",
    "hyderabad": "Hyderabad,IN",
    "chennai": "Chennai,IN",
}


@tool
def get_city_weather(city: str) -> str:
    """Get current weather for a city in India."""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return "Error: set OPENWEATHER_API_KEY in your .env file."

    query = CITY_ALIASES.get(city.strip().lower(), f"{city.strip()},IN")
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": query, "appid": api_key, "units": "metric"}

    try:
        response = requests.get(url, params=params, timeout=15)
    except requests.RequestException as exc:
        return f"Network error calling OpenWeather: {exc}"

    if response.status_code != 200:
        return f"Weather API error {response.status_code}: {response.text[:200]}"

    data = response.json()
    temp = data["main"]["temp"]
    desc = data["weather"][0]["description"]
    return f"Weather in {query}: {desc}, {temp}°C"


@tool
def get_city_news(city: str, topic: str = "latest news") -> str:
    """Get recent news for an Indian city."""
    if not os.getenv("TAVILY_API_KEY"):
        return "Error: set TAVILY_API_KEY in your .env file."

    query = f"{CITY_ALIASES.get(city.strip().lower(), city.strip())} {topic}"
    try:
        raw = TavilySearch(max_results=3).invoke({"query": query})
    except Exception as exc:
        return f"Tavily search error: {exc}"

    results = raw.get("results", raw) if isinstance(raw, dict) else raw
    if not results:
        return "No news found."

    first = results[0]
    title = first.get("title", "")
    snippet = (first.get("content") or "")[:200]
    return f"{title}: {snippet}"


TOOLS = [get_city_weather, get_city_news]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


def build_llm() -> ChatMistralAI:
    return ChatMistralAI(model_name="ministral-8b-2512")


def api_key_status() -> dict[str, bool]:
    return {
        "Mistral": bool(os.getenv("MISTRAL_API_KEY")),
        "OpenWeather": bool(os.getenv("OPENWEATHER_API_KEY")),
        "Tavily": bool(os.getenv("TAVILY_API_KEY")),
    }
