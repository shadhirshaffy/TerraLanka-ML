import os
import re
import pandas as pd
import numpy as np

def clean_land_data(input_path="raw_ikman_lands.csv", output_path="data/clean_lands.csv"):
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found. Run your scraper first!")
        return

    print("Loading raw dataset...")
    df = pd.read_csv(input_path)
    
    # Drop completely empty rows or duplicates
    df = df.dropna(how='all')
    df = df.drop_duplicates()
    print(f"Initial records loaded: {len(df)}")

    # 1. Clean Price Field (Extract numeric value in LKR)
    def parse_price(price_str):
        if pd.isna(price_str):
            return np.nan
        # Remove commas and non-digit characters except decimals
        digits = re.sub(r'[^\d]', '', str(price_str))
        return float(digits) if digits else np.nan

    df['total_price_lkr'] = df['raw_price'].apply(parse_price)

    # 2. Extract Land Extent and Normalize to Perches
    def parse_extent(title_str):
        if pd.isna(title_str):
            return np.nan, np.nan
        
        text = str(title_str).lower()
        
        # Check for acres
        acre_match = re.search(r'(\d+(\.\d+)?)\s*acr', text)
        if acre_match:
            acres = float(acre_match.group(1))
            return acres * 160, 'perches' # 1 acre = 160 perches
            
        # Check for perches
        perch_match = re.search(r'(\d+(\.\d+)?)\s*per', text)
        if perch_match:
            perches = float(perch_match.group(1))
            return perches, 'perches'
            
        return np.nan, np.nan

    # Apply extraction over title/descriptions
    extent_results = df['title'].apply(parse_extent)
    df['land_extent_perches'] = [res[0] for res in extent_results]
    
    # 3. Calculate Target Variable: Price Per Perch
    # If total price and extent are present, compute price per perch
    df['price_per_perch_lkr'] = df['total_price_lkr'] / df['land_extent_perches']

    # 4. Clean Location Data
    # The site generally stores the district/city prefix first, followed by 'Land For Sale'.
    df['city'] = df['location'].apply(lambda x: str(x).split(',')[0].strip() if pd.notna(x) else 'Unknown')
    df['district'] = df['city']

    # Drop rows missing crucial target or size values
    df_clean = df.dropna(subset=['price_per_perch_lkr', 'land_extent_perches']).copy()
    
    # Filter out unreasonable outliers (e.g., negative prices or extreme typos)
    df_clean = df_clean[(df_clean['price_per_perch_lkr'] > 10000) & (df_clean['price_per_perch_lkr'] < 500000000)]

    # Ensure output directory exists and save
    os.makedirs('data', exist_ok=True)
    df_clean.to_csv(output_path, index=False)
    print(f"Preprocessing complete! Cleaned data saved to {output_path} with {len(df_clean)} valid records.")
    print(df_clean[['city', 'land_extent_perches', 'total_price_lkr', 'price_per_perch_lkr']].head())

if __name__ == "__main__":
    clean_land_data()