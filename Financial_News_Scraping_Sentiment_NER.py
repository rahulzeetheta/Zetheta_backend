# %%
pages = 1 # Increase this number. Start with 5-10, adjust as needed. It might take some time.

# %%
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
import numpy as np
import os

# --- Configuration ---
BASE_URL = "https://www.moneycontrol.com"
NEWS_URL = "https://www.moneycontrol.com/news/business/"
NUM_PAGES_TO_SCRAPE = pages # Increase this number. Start with 5-10, adjust as needed.
CSV_FILE = "moneycontrol_financial_news.csv" # Define the CSV file name

# --- Data Storage List ---
all_articles_data = []

# --- Function to fetch page content ---
def get_page_content(url, retries=3):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36'
    }
    for i in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
            return response.content
        except requests.exceptions.RequestException as e:
            print(f"Request failed for {url} (Attempt {i+1}/{retries}): {e}")
            time.sleep(2 * (i + 1)) # Exponential back-off
    print(f"Failed to retrieve content for {url} after {retries} attempts.")
    return None

# --- Function to parse article links from a news listing page ---
def get_article_links(page_content):
    soup = BeautifulSoup(page_content, 'html.parser')
    article_links = []

    news_list_container = soup.find('ul', id='cagetory')

    if not news_list_container:
        print("ERROR: Could not find the main news listing container <ul id='cagetory'>.")
        print("      The website structure might have changed again or this ID is not present on this specific URL.")
        return []

    print("DEBUG: Found news list container <ul id='cagetory'>.")

    list_items = news_list_container.find_all('li')
    print(f"DEBUG: Found {len(list_items)} list items within the container.")

    for item in list_items:
        h2_tag = item.find('h2')
        if h2_tag:
            link_tag = h2_tag.find('a')
            if link_tag and link_tag.get('href'):
                full_link = link_tag['href']
                if not full_link.startswith(BASE_URL):
                    full_link = BASE_URL + full_link
                article_links.append(full_link)

    unique_links = list(set(article_links))
    print(f"DEBUG: Extracted {len(unique_links)} unique article links.")
    return unique_links

# --- Function to extract details from an individual article page (FINAL DATE EXTRACTION & PARSING) ---
def get_article_details(article_url):
    print(f"Scraping article: {article_url}")
    article_content = get_page_content(article_url)
    if not article_content:
        return None

    soup = BeautifulSoup(article_content, 'html.parser')

    title = ""
    publish_date = ""
    article_body = ""

    # 1. Extract Title: Remains the same
    title_tag = soup.find('h1', class_='artTitle') or soup.find('h1', class_='article_title')
    if title_tag:
        title = title_tag.get_text(strip=True)

    # 2. Extract Publish Date: UPDATED REGEX PRIORITY
    date_tag = soup.find('div', class_='article_schedule')
    if date_tag:
        date_text_raw = date_tag.get_text(strip=True)
        print(f"DEBUG: Raw date text found: '{date_text_raw}' for {article_url}")

        # NEW Pattern 1 (Highest Priority): "Month DD, YYYY/ HH:MM IST" (e.g., June 02, 2025/ 10:18 IST)
        match1 = re.search(r'(\w+\s+\d{1,2},\s+\d{4}/\s*\d{1,2}:\d{2}\s+IST)', date_text_raw, re.IGNORECASE)
        if match1:
            publish_date = match1.group(1).strip()
            print(f"DEBUG: Date extracted with NEW Pattern 1: '{publish_date}'")
        else:
            # Original Pattern 1 (now lower priority): "Month DD, YYYY HH:MM IST" (e.g., JUNE 02, 2025 14:14 IST)
            match2 = re.search(r'(\w+\s+\d{1,2},\s+\d{4}\s+\d{1,2}:\d{2}\s+IST)', date_text_raw, re.IGNORECASE)
            if match2:
                publish_date = match2.group(1).strip()
                print(f"DEBUG: Date extracted with Original Pattern 1: '{publish_date}'")
            else:
                # Pattern 2: "FIRST PUBLISHED: Jun 02, 2025 09:30 AM IST"
                match3 = re.search(r'PUBLISHED:\s*(\w+\s+\d{1,2},\s+\d{4}\s+\d{1,2}:\d{2}\s+(?:AM|PM)\s+IST)', date_text_raw, re.IGNORECASE)
                if match3:
                    publish_date = match3.group(1).strip()
                    print(f"DEBUG: Date extracted with Pattern 2: '{publish_date}'")
                else:
                    # Pattern 3: "Month DD, YYYY" (e.g., Jun 02, 2025) - Only if time not found
                    match4 = re.search(r'(\w+\s+\d{1,2},\s+\d{4})', date_text_raw, re.IGNORECASE)
                    if match4:
                        publish_date = match4.group(0).strip() # Using group(0) to get entire match
                        print(f"DEBUG: Date extracted with Pattern 3 (date only): '{publish_date}'")
                    else:
                        # Pattern 4: "Updated: Month DD, YYYY HH:MM AM/PM IST"
                        match5 = re.search(r'Updated:\s*(\w+\s+\d{1,2},\s+\d{4}\s+\d{1,2}:\d{2}\s+(?:AM|PM)\s+IST)', date_text_raw, re.IGNORECASE)
                        if match5:
                            publish_date = match5.group(1).strip()
                            print(f"DEBUG: Date extracted with Pattern 4: '{publish_date}'")
                        else:
                            publish_date = date_text_raw # As last resort, use raw text if no pattern matches
                            print(f"DEBUG: No specific date pattern matched, using raw text: '{publish_date}'")
    else:
        print(f"DEBUG: Date element with class 'article_schedule' not found for {article_url}")

    # 3. Extract Article Content (Body): Remains the same, but robust
    content_container = soup.find('div', id='mc_content')
    if not content_container:
        content_container = soup.find('div', class_='content_wrapper') or \
                             soup.find('div', class_='article_body') or \
                             soup.find('div', class_='article_content')

    if not content_container:
        print(f"Warning: Could not find main content container for {article_url}. Skipping body extraction.")

    if content_container:
        paragraphs = content_container.find_all('p')
        article_body = '\n'.join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True) and len(p.get_text(strip=True)) > 30])
        if "Disclaimer:" in article_body:
            article_body = article_body.split("Disclaimer:")[0].strip()
        if "Source:" in article_body:
            article_body = article_body.split("Source:")[0].strip()
        if "Read more:" in article_body:
            article_body = article_body.split("Read more:")[0].strip()

    if not title or not article_body:
        print(f"Warning: Missing title or body for {article_url}. Title: '{title[:50]}...', Body length: {len(article_body)}")
        return None

    return {
        'url': article_url,
        'title': title,
        'publish_date': publish_date, # This will now contain the full date-time string if matched
        'content': article_body
    }

# --- Main Scraping Logic (Remains the same) ---
print("Starting Moneycontrol Web Scraper...")

for page_num in range(1, NUM_PAGES_TO_SCRAPE + 1):
    page_url = f"{NEWS_URL}page-{page_num}/" if page_num > 1 else NEWS_URL
    print(f"\nScraping listing page: {page_url}")

    page_content = get_page_content(page_url)
    if not page_content:
        continue

    article_links_on_page = get_article_links(page_content)

    if not article_links_on_page:
        print(f"No article links found on {page_url}. This might mean no more pages or the selectors are incorrect.")
        break

    print(f"Attempting to scrape {len(article_links_on_page)} unique articles from this page.")
    articles_scraped_on_page = 0
    for link in article_links_on_page:
        article_data = get_article_details(link)
        if article_data:
            all_articles_data.append(article_data)
            articles_scraped_on_page += 1
        time.sleep(1)

    print(f"Successfully scraped {articles_scraped_on_page} articles from {page_url}")
    time.sleep(2)

print(f"\nScraping finished. Total articles collected: {len(all_articles_data)}")

# --- Store Data in Pandas DataFrame and CSV (UPDATED DATE CONVERSION AND DATA HANDLING) ---
if all_articles_data:
    df = pd.DataFrame(all_articles_data)

    # Convert 'publish_date' to datetime objects for better handling
    def convert_date(date_str):
        if not isinstance(date_str, str) or not date_str: # Handle empty strings or non-strings
            return pd.NaT

        # Try format: "YYYY-MM-DD HH:MM:SS" (This is what pandas saves and loads as)
        try:
            return pd.to_datetime(date_str, format='%Y-%m-%d %H:%M:%S')
        except ValueError:
            pass

        # Try format: "Month DD, YYYY/ HH:MM IST" (e.g., June 02, 2025/ 10:18 IST)
        try:
            return pd.to_datetime(date_str, format='%B %d, %Y/ %H:%M IST')
        except ValueError:
            pass

        # Try format: "Month DD, YYYY HH:MM IST" (e.g., JUNE 02, 2025 14:14 IST)
        try:
            return pd.to_datetime(date_str, format='%B %d, %Y %H:%M IST')
        except ValueError:
            pass

        # Try format: "Month DD, YYYY HH:MM AM/PM IST" (e.g., Jun 02, 2025 09:30 AM IST)
        try:
            return pd.to_datetime(date_str, format='%b %d, %Y %I:%M %p IST')
        except ValueError:
            pass

        # Try format: "Month DD, YYYY" (e.g., Jun 02, 2025) - Last resort for just date
        try:
            return pd.to_datetime(date_str, format='%b %d, %Y')
        except ValueError:
            pass

        # Try format: "Month DD, YYYY HH:MM AM/PM" (e.g., November 29, 2023 09:27 AM)
        try:
            return pd.to_datetime(date_str, format='%B %d, %Y %I:%M %p')
        except ValueError:
            pass

        print(f"Warning: Could not parse date string '{date_str}'") # Debug unparsed strings
        return pd.NaT # If no format matches

    df['publish_date'] = df['publish_date'].apply(convert_date)

    # Load existing data, if any
    if os.path.exists(CSV_FILE):
        existing_df = pd.read_csv(CSV_FILE)
        # Apply convert_date to existing data *before* any merge operations
        # This is where the new format string is essential for previously saved dates
        existing_df['publish_date'] = existing_df['publish_date'].apply(convert_date)

        # Identify new articles by URL that are not in existing_df OR have NaT dates that need re-scraping
        # First, get URLs of valid articles from existing_df
        existing_valid_urls = set(existing_df[existing_df['publish_date'].notna()]['url'])

        # Now, filter new articles that are truly new or whose dates were NaT and have been fixed in the current run
        # This line was slightly adjusted to be more robust
        new_articles_df = df[~df['url'].isin(existing_valid_urls)].copy()

        # If an article was re-scraped and now has a valid date, we need to update the existing_df
        # Create a temporary DataFrame of articles from the current run that are already in existing_df
        # and have a valid date
        articles_to_update = df[df['url'].isin(existing_df['url']) & df['publish_date'].notna()]

        # Merge/update existing_df with the newly scraped (and valid) data
        # This ensures that previously NaT dates get updated
        existing_df = existing_df.set_index('url')
        articles_to_update_indexed = articles_to_update.set_index('url')
        existing_df.update(articles_to_update_indexed)
        existing_df = existing_df.reset_index()


        # Concatenate truly new articles with the (potentially updated) existing data
        final_df = pd.concat([existing_df, new_articles_df], ignore_index=True)
        final_df.drop_duplicates(subset=['url'], keep='first', inplace=True) # Ensure no URL duplicates after concat
    else:
        final_df = df.copy()  # Copy to avoid modifying original df

    # Sort by publish_date, newest first
    final_df.sort_values(by='publish_date', ascending=False, inplace=True)

    csv_filename = 'moneycontrol_financial_news.csv'
    final_df.to_csv(csv_filename, index=False, encoding='utf-8')
    print(f"\nData successfully saved to {csv_filename}")
    print(final_df.head())
    print(f"\nDataFrame shape: {final_df.shape}")
    print(f"\nNumber of NaT dates after final parsing: {final_df['publish_date'].isnull().sum()}")  # Check NaT count
    print("✅ Success: Content saved to 'moneycontrol_financial_news.csv'")
else:
    print("No articles were scraped.")

# %%
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# %%
model_path = r"finbert-finetuned-3"

# %%
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
model.eval()

# %%
df = final_df
headlines = df['title'].astype(str).tolist()  # ensure all entries are strings
content = df['content'].astype(str).tolist()
df.head()

# %%
all_sentiments = []
all_probs = []

sentiment_labels = ['Positive', 'Neutral', 'Negative']

for text in headlines:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        sentiment = sentiment_labels[pred_idx]
        
        all_sentiments.append(sentiment)
        all_probs.append(probs.tolist())

df['sentiment'] = all_sentiments
df['positive_score'] = [p[0] for p in all_probs]
df['neutral_score'] = [p[1] for p in all_probs]
df['negative_score'] = [p[2] for p in all_probs]

df.to_csv("headlines_with_sentiment.csv", index=False)
print("✅ Success: Headlines with sentiments are saved to 'headlines_with_sentiment.csv'")

# %%
import numpy
import scipy
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification

# %%
model_name = "ai4bharat/IndicNER"

# %%
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# %%
import pandas as pd
from transformers import pipeline
from rapidfuzz import process, fuzz

contents = df['content'].astype(str).tolist()

# Load company-sector mapping
company_df = pd.read_csv("India_PLC_Sector_Industry_2024_MAY.csv")
company_sector_map = dict(zip(company_df['Description'].str.upper(), company_df['Sector']))
company_names = list(company_sector_map.keys())

# Load NER model
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy = "simple")

# Keywords
market_keywords = [
    "stock market", "sensex", "nifty", "indices", "index", "equities", "benchmark",
    "markets", "share market", "stock exchange", "bse", "nse", "dow jones", "nasdaq"
]
commodity_keywords = [
    "gold", "silver", "oil", "crude", "natural gas", "metals", "commodity", "commodities",
    "bullion", "energy prices", "base metals", "fuel", "agriculture", "wheat", "sugar"
]

# Function to extract companies and assign tag
def extract_info(text):
    try:
        ner_results = ner_pipeline(text)
    except:
        ner_results = []

    matched_companies = set()
    for ent in ner_results:
        if ent['entity_group'] == 'ORG':
            entity_name = ent['word'].upper().strip()
            match, score, _ = process.extractOne(entity_name, company_names, scorer=fuzz.ratio)
            if score >= 80:
                matched_companies.add(match)

    # Tagging logic
    lowered = text.lower()
    if matched_companies:
        tag = "Company-specific"
    elif any(k in lowered for k in market_keywords):
        tag = "Market-wide"
    elif any(k in lowered for k in commodity_keywords):
        tag = "Commodity"
    else:
        tag = "Uncategorized"

    return list(matched_companies), tag

# Apply to all content
df[['companies', 'tag']] = df['content'].astype(str).apply(lambda x: pd.Series(extract_info(x)))

# Save result
df.to_csv("articles_with_companies_and_tags.csv", index=False)
print("✅ Success: companies and tags saved to 'articles_with_companies_and_tags.csv'")


# %%



