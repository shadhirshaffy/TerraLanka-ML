import os
import time
import requests
from bs4 import BeautifulSoup
import pandas as pd
from pathlib import Path

def scrape_all_ikman_lands(max_pages=50):
    base_url = "https://ikman.lk/en/ads/sri-lanka/land-for-sale"
    data = []
    
    print("Initializing bulk data import from ikman.lk...")

    for page in range(1, max_pages + 1):
        url = f"{base_url}?page={page}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            print(f"Reached end or failed at page {page} (Status code: {response.status_code})")
            break

        soup = BeautifulSoup(response.text, "html.parser")
        listings = soup.select("li.gtm-normal-ad")

        if not listings:
            if page == 1:
                raise RuntimeError(
                    f"No land listings found on page 1; the site markup may have changed ({response.url})."
                )
            print(f"No more listings found on page {page}. Ending crawl.")
            break

        for item in listings:
            title_element = item.find("h2")
            price_element = item.find("div", class_="price--3SnqI")
            location_element = item.find("div", class_="description--2-ez3")

            data.append({
                "title": title_element.get_text(strip=True) if title_element else None,
                "raw_price": price_element.get_text(" ", strip=True) if price_element else None,
                "location": location_element.get_text(" ", strip=True) if location_element else None
            })

        print(f"Successfully scraped page {page} (Total records so far: {len(data)})")
        
        if page % 10 == 0:
            temp_df = pd.DataFrame(data)
            os.makedirs("data", exist_ok=True)
            temp_df.to_csv("data/raw_ikman_lands_checkpoint.csv", index=False)

        if page < max_pages:
            time.sleep(2)

    if not data:
        raise RuntimeError("No land listings were scraped; the existing CSV was not changed.")
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    
    # Adjust max_pages based on how deep you want to go (e.g., 50 pages * ~25 ads = ~1,250+ records)
    df_lands = scrape_all_ikman_lands(max_pages=50)
    
    output_path = Path(__file__).with_name("raw_ikman_lands.csv")
    write_header = not output_path.exists() or output_path.stat().st_size == 0
    df_lands.to_csv(output_path, mode="a", header=write_header, index=False)
    print(f"\nBulk Import Complete! Appended {len(df_lands)} records to {output_path}.")