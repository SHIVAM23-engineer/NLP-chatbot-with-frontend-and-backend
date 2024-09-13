# content_scraper.py
import requests
from bs4 import BeautifulSoup

# Function to scrape content from a single webpage
def scrape_page_content(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an HTTPError for bad responses
        soup = BeautifulSoup(response.content, 'html.parser')

        title = soup.title.string if soup.title else "No Title"
        content = soup.get_text(separator='\n', strip=True)
        
        return {
            'url': url,
            'title': title,
            'content': content
        }
    except requests.exceptions.RequestException as e:
        print(f"Failed to scrape {url}: {e}")
        return None
