import argparse
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_PATH = PROJECT_DIR / "raw_ikman_lands.csv"
CHECKPOINT_PATH = PROJECT_DIR / "data" / "raw_ikman_lands_checkpoint.csv"


def scrape_ikman_lands(max_pages=50, delay_seconds=2):
    base_url = "https://ikman.lk/en/ads/sri-lanka/land-for-sale"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    data = []

    for page in range(1, max_pages + 1):
        url = f"{base_url}?page={page}"
        response = requests.get(url, headers=headers, timeout=15)

        if response.status_code != 200:
            print(f"Stopping at page {page}; status code {response.status_code}.")
            break

        soup = BeautifulSoup(response.text, "html.parser")
        listings = soup.select("li.gtm-normal-ad")
        if not listings:
            if page == 1:
                raise RuntimeError(
                    f"No listings found on page 1; site markup may have changed ({response.url})."
                )
            print(f"No more listings found on page {page}.")
            break

        for item in listings:
            title_element = item.find("h2")
            price_element = item.find("div", class_="price--3SnqI")
            location_element = item.find("div", class_="description--2-ez3")
            link_element = item.find("a", href=True)

            listing_url = None
            if link_element:
                href = link_element["href"]
                listing_url = href if href.startswith("http") else f"https://ikman.lk{href}"

            data.append(
                {
                    "title": title_element.get_text(strip=True) if title_element else None,
                    "raw_price": price_element.get_text(" ", strip=True) if price_element else None,
                    "location": location_element.get_text(" ", strip=True) if location_element else None,
                    "listing_url": listing_url,
                    "source_page": page,
                }
            )

        print(f"Scraped page {page}; total records: {len(data)}")

        if page % 10 == 0:
            CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(data).to_csv(CHECKPOINT_PATH, index=False)

        if page < max_pages:
            time.sleep(delay_seconds)

    if not data:
        raise RuntimeError("No land listings were scraped; output was not written.")

    return pd.DataFrame(data)


def main():
    parser = argparse.ArgumentParser(description="Scrape ikman.lk land listings.")
    parser.add_argument("--pages", type=int, default=50, help="Maximum pages to scrape.")
    parser.add_argument("--delay", type=float, default=2, help="Delay between page requests.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="CSV output path.",
    )
    args = parser.parse_args()

    df = scrape_ikman_lands(max_pages=args.pages, delay_seconds=args.delay)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} listings to {args.output}.")


if __name__ == "__main__":
    main()
