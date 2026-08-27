from dotenv import load_dotenv
load_dotenv()
from langchain.tools import tool
from bs4 import BeautifulSoup
import requests
from tavily import TavilyClient
import os
from rich import print

tavily = TavilyClient(api_key = os.getenv("TAVILY_API_KEY"))


@tool
def web_search(query: str) -> str:
    """
    Search the web for recent and reliable information on a topic.
    Returns titles, URLs, and snippets.
    """

    results = tavily.search(
        query=query,
        max_results=5
    )

    out = []

    for r in results["results"]:
        out.append(
            f"title: {r['title']}\n"
            f"URL: {r['url']}\n"
            f"snippet: {r['content'][:300]}\n"
        )

    return "\n---\n".join(out)

@tool
def scrape_url(url:str)->str:
    """Scrape and return the clean text content from given url for deeeper reading"""
    try:
        resp = requests.get(url, timeout=8 , headers={"User-agent": "Mozilla/5.0"})
        soup = BeautifulSoup(resp.text,"html.parser")
        for tag in soup(["script","style","nav","footer"]):
            tag.decompose()
        return soup.get_text(separator="",strip=True)[:3000]
    except Exception as e:
        return f"Could not scrape url : {str(e)}"
