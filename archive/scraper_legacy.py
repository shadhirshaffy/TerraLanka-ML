import os
import time
import requests
from bs4 import BeautifulSoup
import pandas as pd

def scrape_ikman_lands(pages=3):
    base_url = "https://ikman.lk/en/ads/sri-lanka/land"
    data = []

    for page in range(1, pages + 1):
        url = f"{base_url}?page={page}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            print(f"Failed to fetch page {page} (Status code: {response.status_code})")
            continue

        soup = BeautifulSoup(response.text, "html.parser")
        listings = soup.find_all("li", class_="normal-item--1SFgg")

        for item in listings:
            try:
                title = item.find("span", class_="title--3s devotional").text.strip()
            except AttributeError:
                title = None

            try:
                price_str = item.find("div", class_="price--3SnqI").text.strip()
            except AttributeError:
                price_str = None

            try:
                location = item.find("div", class_="description--2-ez3").text.strip()
            except AttributeError:
                location = None

            data.append({
                "title": title,
                "raw_price": price_str,
                "location": location
            })

        print(f"Scraped page {page} successfully.")
        time.sleep(2)

    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    print("Starting data collection from ikman.lk...")
    df_lands = scrape_ikman_lands(pages=3)
    output_path = "data/raw_ikman_lands.csv"
    df_lands.to_csv(output_path, index=False)
    print(f"Successfully saved {len(df_lands)} records to {output_path}!")
    print(df_lands.head())

