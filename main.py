
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, TFAutoModelForCausalLM
from datetime import datetime, timedelta
import requests
import tensorflow as tf

app = FastAPI()

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForCausalLM.from_pretrained(model_name)


class NewsRequest(BaseModel):
    keyword: str


@app.post("/analyze_news")
def analyze_news(request: NewsRequest):
    keyword = request.keyword

    days_back = 7
    today = datetime.today()
    date_end = today.strftime("%Y-%m-%d")
    date_start = (today - timedelta(days=days_back)).strftime("%Y-%m-%d")

    url = (
        f"https://eventregistry.org/api/v1/article/getArticles"
        f"?apiKey=a02a1316-1972-4a6c-9ea4-fd031a36281f"
        f"&resultType=articles&articlesPage=1&articlesCount=1"
        f"&articlesSortBy=date&articlesSortByAsc=false"
        f"&articleBodyLen=-1&dataType=news"
        f"&keyword={keyword}&lang=eng"
        f"&dateStart={date_start}&dateEnd={date_end}"
        f"&keywordLoc=body&keywordOper=and"
    )

    r = requests.get(url, headers={'Accept': 'application/json'})
    news_data = r.json()
    articles = news_data.get('articles', {}).get('results', [])

    if not articles:
        raise HTTPException(status_code=404, detail="No articles found")

    article = articles[0]
    title = article.get('title')
    source = article.get('source', {}).get('title')
    sentiment_score = article.get('sentiment')
    article_body = article.get('body', '')
    article_url = article.get('url')

    if sentiment_score is not None:
        if sentiment_score <= -3:
            sentiment_label = "Extremely Negative"
        elif -3 < sentiment_score <= 0:
            sentiment_label = "Mild Negative"
        elif sentiment_score == 1:
            sentiment_label = "Neutral"
        elif 0 <= sentiment_score < 1:
            sentiment_label = "Mild Positive"
        elif sentiment_score >= 3:
            sentiment_label = "Strongly Positive"
        else:
            sentiment_label = "Unclassified Sentiment"
    else:
        sentiment_label = "Sentiment data not available"

    prompt = (
        f"Read the following news article and perform the following tasks:\n"
        f"1. Summarize the main point of the article.\n"
        f"2. Provide any relevant background information.\n"
        f"3. Predict possible implications.\n\n"
        f"News Article:\n{article_body}\n\nResponse:"
    )

    input_ids = tokenizer(prompt, return_tensors="tf").input_ids
    output_ids = model.generate(
        input_ids,
        max_new_tokens=200,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.9,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3
    )

    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return {
        "title": title,
        "source": source,
        "url": article_url,
        "sentiment_score": sentiment_score,
        "sentiment_label": sentiment_label,
        "summary": summary.strip()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=5000, reload=True)
