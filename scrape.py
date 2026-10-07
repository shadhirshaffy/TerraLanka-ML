import argparse
import os
import re
import time
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_PATH = PROJECT_DIR / "raw_ikman_lands.csv"
CHECKPOINT_PATH = PROJECT_DIR / "data" / "raw_ikman_lands_checkpoint.csv"
DEFAULT_LPW_OUTPUT_PATH = PROJECT_DIR / "data" / "raw_lankapropertyweb_colombo_lands.csv"
DEFAULT_PRIMELANDS_OUTPUT_PATH = PROJECT_DIR / "data" / "raw_primelands_colombo_lands.csv"


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


def _normalize_spaces(value):
    if not isinstance(value, str):
        return None
    return re.sub(r"\s+", " ", value).strip() or None


def _extract_lpw_record(card, page_number):
    text = card.get("text") or ""
    lines = [
        _normalize_spaces(line)
        for line in re.split(r"[\r\n]+", text)
        if _normalize_spaces(line)
    ]
    title = card.get("title") or next(
        (
            line
            for line in lines
            if not re.search(r"^(rs\.?|lkr|view|more|contact|whatsapp|advertiser)", line, re.I)
        ),
        None,
    )
    raw_price = next(
        (
            line
            for line in lines
            if re.search(r"\b(rs\.?|lkr)\b|million|lakh|per\s+perch", line, re.I)
        ),
        None,
    )
    extent = next(
        (
            line
            for line in lines
            if re.search(r"\b\d+(?:\.\d+)?\s*(?:perch|perches|acre|acres)\b", line, re.I)
        ),
        None,
    )
    location = next(
        (
            line
            for line in lines
            if re.search(r"\bcolombo\b|\bdehiwala\b|\bnawala\b|\bnugegoda\b|\bmalabe\b", line, re.I)
        ),
        None,
    )
    if title and extent and extent not in title:
        title = f"{title} {extent}"
    if location and not re.search(r"\bcolombo\b", location, re.I):
        location = f"Colombo, {location}"

    return {
        "title": title,
        "raw_price": raw_price,
        "location": location or "Colombo",
        "extent": extent,
        "listing_url": card.get("url"),
        "source_page": page_number,
        "source": "lankapropertyweb",
        "raw_text": _normalize_spaces(text),
    }


def _find_installed_chromium():
    candidates = [
        Path(os.environ.get("PROGRAMFILES", "")) / "Google/Chrome/Application/chrome.exe",
        Path(os.environ.get("PROGRAMFILES(X86)", "")) / "Google/Chrome/Application/chrome.exe",
        Path(os.environ.get("PROGRAMFILES", "")) / "Microsoft/Edge/Application/msedge.exe",
        Path(os.environ.get("PROGRAMFILES(X86)", "")) / "Microsoft/Edge/Application/msedge.exe",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return None


def scrape_lankapropertyweb_lands(max_pages=10, delay_seconds=2, headless=False):
    """Scrape LankaPropertyWeb Colombo land listings with a real browser.

    LankaPropertyWeb is protected by Cloudflare, so requests/BeautifulSoup receives
    the challenge page. Playwright lets a user complete the check in the opened
    browser once, then the scraper reads listing cards from the rendered page.
    """
    try:
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError(
            "Playwright is required for LankaPropertyWeb scraping. Run:\n"
            "  python -m pip install playwright\n"
            "  python -m playwright install chromium"
        ) from exc

    base_url = "https://www.lankapropertyweb.com/land/colombo/"
    data = []
    seen_urls = set()

    with sync_playwright() as playwright:
        launch_options = {"headless": headless}
        installed_browser = _find_installed_chromium()
        if installed_browser:
            launch_options["executable_path"] = installed_browser
        browser = playwright.chromium.launch(**launch_options)
        page = browser.new_page(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        )

        for page_number in range(1, max_pages + 1):
            url = base_url if page_number == 1 else f"{base_url}?page={page_number}"
            print(f"Opening {url}")
            page.goto(url, wait_until="domcontentloaded", timeout=90_000)

            if "Just a moment" in page.title():
                print("Cloudflare check detected. Complete it in the browser window.")
                page.wait_for_load_state("networkidle", timeout=180_000)

            try:
                page.wait_for_selector("a[href]", timeout=60_000)
            except PlaywrightTimeoutError:
                print(f"No links found on page {page_number}; stopping.")
                break

            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            page.wait_for_timeout(1500)
            cards = page.evaluate(
                """
                () => {
                  const anchors = [...document.querySelectorAll('a[href]')]
                    .filter(a => /land|property/i.test(a.href) && a.innerText.trim().length > 3);
                  const byUrl = new Map();
                  for (const anchor of anchors) {
                    const card = anchor.closest('article, li, .property-card, .listing, .search-result, .result, div');
                    const node = card || anchor;
                    const text = node.innerText || anchor.innerText || '';
                    if (text.length < 30 || !/(Rs\\.?|LKR|million|lakh|perch|acre)/i.test(text)) continue;
                    byUrl.set(anchor.href, {
                      url: anchor.href,
                      title: anchor.innerText.trim(),
                      text
                    });
                  }
                  return [...byUrl.values()];
                }
                """
            )

            new_records = 0
            for card in cards:
                absolute_url = urljoin(base_url, card.get("url") or "")
                if not absolute_url or absolute_url in seen_urls:
                    continue
                card["url"] = absolute_url
                record = _extract_lpw_record(card, page_number)
                if record["title"] and record["raw_price"]:
                    data.append(record)
                    seen_urls.add(absolute_url)
                    new_records += 1

            print(f"Scraped page {page_number}; new records: {new_records}; total records: {len(data)}")
            if new_records == 0 and page_number > 1:
                break
            if page_number < max_pages:
                time.sleep(delay_seconds)

        browser.close()

    if not data:
        raise RuntimeError("No LankaPropertyWeb listings were scraped; output was not written.")
    return pd.DataFrame(data)


def _first_text(node, selector):
    element = node.select_one(selector)
    return element.get_text(" ", strip=True) if element else None


def _prime_lands_detail(listing_url, headers):
    response = requests.get(listing_url, headers=headers, timeout=20)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    description = None
    meta_description = soup.select_one('meta[name="description"]')
    if meta_description and meta_description.get("content"):
        description = _normalize_spaces(meta_description["content"])

    page_text = _normalize_spaces(soup.get_text(" ", strip=True))
    extent = None
    if page_text:
        match = re.search(
            r"\b\d+(?:\.\d+)?\s*(?:perch|perches|acre|acres)\b(?:\s+onwards)?",
            page_text,
            re.I,
        )
        if match:
            extent = match.group(0)

    return description, extent


def scrape_primelands_lands(max_pages=5, delay_seconds=2, fetch_details=True):
    base_url = "https://www.primelands.lk/land/en"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    data = []
    seen_urls = set()

    for page_number in range(1, max_pages + 1):
        params = {
            "category": "all",
            "district": "1",
            "city": "all",
            "min_value": "LKR",
            "max_value": "LKR",
            "keyword": "colombo",
        }
        if page_number > 1:
            params["page"] = page_number

        response = requests.get(base_url, params=params, headers=headers, timeout=20)
        if response.status_code != 200:
            print(f"Stopping at page {page_number}; status code {response.status_code}.")
            break

        soup = BeautifulSoup(response.text, "html.parser")
        listings = soup.select(".land_listing a.text_link[href]")
        if not listings:
            if page_number == 1:
                raise RuntimeError(
                    f"No Prime Lands listings found on page 1; site markup may have changed ({response.url})."
                )
            print(f"No more Prime Lands listings found on page {page_number}.")
            break

        new_records = 0
        for item in listings:
            listing_url = urljoin(base_url, item["href"])
            if listing_url in seen_urls:
                continue

            title = _first_text(item, "h2.card-title")
            location = _first_text(item, "p")
            raw_price = _first_text(item, "h3")
            price_note = _first_text(item, "p.fst-italic small")
            description = None
            extent = None

            if raw_price and price_note:
                raw_price = f"{raw_price} {price_note}"

            if fetch_details:
                try:
                    description, extent = _prime_lands_detail(listing_url, headers)
                except requests.RequestException as exc:
                    print(f"Could not fetch details for {listing_url}: {exc}")

            title_parts = [part for part in [title, extent] if part]
            data.append(
                {
                    "title": " ".join(title_parts) if title_parts else title,
                    "raw_price": _normalize_spaces(raw_price),
                    "location": _normalize_spaces(f"Colombo, {location}" if location else "Colombo"),
                    "extent": extent,
                    "description": description,
                    "listing_url": listing_url,
                    "source_page": page_number,
                    "source": "primelands",
                }
            )
            seen_urls.add(listing_url)
            new_records += 1

            if fetch_details:
                time.sleep(0.5)

        print(f"Scraped Prime Lands page {page_number}; new records: {new_records}; total records: {len(data)}")
        if page_number < max_pages:
            time.sleep(delay_seconds)

    if not data:
        raise RuntimeError("No Prime Lands listings were scraped; output was not written.")
    return pd.DataFrame(data)


def main():
    parser = argparse.ArgumentParser(description="Scrape Sri Lankan land listings.")
    parser.add_argument(
        "--source",
        choices=["ikman", "lankapropertyweb", "primelands"],
        default="ikman",
        help="Website to scrape.",
    )
    parser.add_argument("--pages", type=int, default=50, help="Maximum pages to scrape.")
    parser.add_argument("--delay", type=float, default=2, help="Delay between page requests.")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run browser invisibly for sources that use Playwright.",
    )
    parser.add_argument(
        "--no-details",
        action="store_true",
        help="Skip detail-page requests for sources that support richer listing details.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV output path.",
    )
    args = parser.parse_args()

    if args.source == "ikman":
        output_path = args.output or DEFAULT_OUTPUT_PATH
        df = scrape_ikman_lands(max_pages=args.pages, delay_seconds=args.delay)
    elif args.source == "lankapropertyweb":
        output_path = args.output or DEFAULT_LPW_OUTPUT_PATH
        df = scrape_lankapropertyweb_lands(
            max_pages=args.pages,
            delay_seconds=args.delay,
            headless=args.headless,
        )
    else:
        output_path = args.output or DEFAULT_PRIMELANDS_OUTPUT_PATH
        df = scrape_primelands_lands(
            max_pages=args.pages,
            delay_seconds=args.delay,
            fetch_details=not args.no_details,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} listings to {output_path}.")


if __name__ == "__main__":
    main()
