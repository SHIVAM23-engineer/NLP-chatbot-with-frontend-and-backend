# main.py
from content_scraper import scrape_page_content
from utils.save_json import save_to_json

# Main function to scrape data from a single URL and save it as JSON
def scrape_and_save_to_json(url, output_file):
    # Scrape content from the specified URL
    page_data = scrape_page_content(url)
    
    # If scraping was successful, save the data
    if page_data:
        save_to_json(page_data, output_file)

# Example usage
if __name__ == "__main__":
    url = "https://en.wikipedia.org/wiki/Kaiser_Permanente"  # Replace with the URL you want to scrape
    output_file = "scraped_data.json"
    scrape_and_save_to_json(url, output_file)
