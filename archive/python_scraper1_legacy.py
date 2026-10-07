
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from pathlib import Path

def scrape_ikman_lands(pages=3):
    base_url = "https://ikman.lk/en/ads/colombo/land-for-sale"
    data = []

    for page in range(1, pages + 1):
        url = f"{base_url}?page={page}"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        listings = soup.select('li.gtm-normal-ad')
        if not listings:
            raise RuntimeError(
                f"No land listings found on page {page}; "
                f"the site markup may have changed ({response.url})."
            )

        for item in listings:
            title_element = item.find('h2')
            price_element = item.find(class_='price--3SnqI')
            location_element = item.find(class_='description--2-ez3')

            data.append({
                'title': title_element.get_text(strip=True) if title_element else None,
                'raw_price': price_element.get_text(" ", strip=True) if price_element else None,
                'location': location_element.get_text(" ", strip=True) if location_element else None
            })

        print(f"Scraped {len(listings)} listings from page {page}.")
        if page < pages:
            time.sleep(2)

    if not data:
        raise RuntimeError("No land listings were scraped; CSV was not written.")
    df = pd.DataFrame(data)
    return df

df = scrape_ikman_lands(pages=5)
output_path = Path(__file__).with_name('raw_ikman_lands.csv')
df.to_csv(output_path, index=False)
print(f"Saved {len(df)} listings to {output_path}.")