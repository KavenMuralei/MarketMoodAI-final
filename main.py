

from bs4 import BeautifulSoup
import requests

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax

MODEL = f"mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

url = "https://finviz.com/news.ashx"

# Header to mimic a chrome browser
headers = {
    'User-Agent': 'Chrome/92.0.4515.159 Safari/537.36'
}

# The request
response = requests.get(url, headers=headers)

# Execute program if status is 200
if response.status_code != 200:
    print(response)

else:
    soup = BeautifulSoup(response.text, features="html.parser")

    # print(soup.prettify)

    news_cells = soup.find_all('td', class_ = 'news_link-cell')

    for cell in news_cells:
        text = cell.get_text(strip=True)
        print(text)  # strip=True removes extra whitespace

        encoded_text = tokenizer(text, return_tensors='pt')
        output = model(**encoded_text)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        scores_dict = {
            'neg' : scores[0],
            'neu' : scores[1],
            'pos' : scores[2]
        }
        if(scores_dict['pos'] >= scores_dict['neg'] and scores_dict['pos'] >= scores_dict['neu']):
            print("Positive: "+ str(scores_dict['pos']))
        elif (scores_dict['neu'] >= scores_dict['neg'] and scores_dict['neu'] >= scores_dict['pos']):
            print("Neutral: "+ str(scores_dict['neu']))
        elif (scores_dict['neg'] >= scores_dict['neu'] and scores_dict['neg'] >= scores_dict['pos']):
            print("Negative: "+ str(scores_dict['neg']))

        print(f"\n")




