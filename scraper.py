from bs4 import BeautifulSoup
import requests

url = "https://finviz.com/news.ashx"

# Header to mimic a chrome browser
headers = {
    'User-Agent': 'Chrome/92.0.4515.159 Safari/537.36'
}

# The request
response = requests.get(url, headers=headers)

print(response)

soup = BeautifulSoup(response.text, features="html.parser")

# print(soup.prettify)

news_cells = soup.find_all('td', class_ = 'news_link-cell')

for cell in news_cells:
    print(cell.get_text(strip=True))  # strip=True removes extra whitespace




