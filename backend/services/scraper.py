import httpx
from bs4 import BeautifulSoup
from urllib.parse import quote

BASE_URL = "https://indiankanoon.org"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def scrape_indian_kanoon(query: str, max_results: int = 5) -> list:
    url = f"{BASE_URL}/search/?formInput={quote(query)}"
    results = []

    try:
        with httpx.Client(headers=HEADERS, timeout=12, follow_redirects=True) as client:
            response = client.get(url)
            response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        result_divs = soup.find_all("div", class_="result_title")

        for div in result_divs[:max_results]:
            a_tag = div.find("a")

            if not a_tag:
                continue

            title = a_tag.text.strip()
            href = a_tag.get("href", "")
            link = BASE_URL + href if href.startswith("/") else href

            results.append({
                "title": title,
                "link": link,
                "snippet": "",
                "source": "Indian Kanoon"
            })

    except Exception as e:
        print(f"[Scraper] Error fetching Indian Kanoon: {e}")

    return results
    print("SCRAPED KANOON:", results)