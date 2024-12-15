import requests
from bs4 import BeautifulSoup
import time
import random
import json
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Základní URL produktu
url = 'https://www.amazon.com/Under-Armour-Outerwear-Packaged-Legging/dp/B07K3GVPXM/ref=sr_1_2?brr=1&content-id=amzn1.sym.77efd490-c71e-4614-b1fe-9c82f75964fe&dib=eyJ2IjoiMSJ9.vNQPDaJh0Id5aLUv7GK-m9cLg1JD-A2a1o02D2jhVN3dEubWZCG9LP8QxXiB6vWswETGegyieWICZgS6IDOvObIz4FVi7u_DpavGFmj0HOI01_Djx7-z4M-athKe0JfNZbRbyCFNiIy69PhTo8skH1vOUTCdAIFPOBDHtcTzTY--diL6mKGUHpXXDukmFxbj9jkQd8ZYT7DBCIEGC82ATb3EPYozrkbuQk8snARVINBFmaDh4zSJ1v-jrr5jbSZe8ZTwIUTuColssZGfg_ZBou94tbkCmK2v1aMlFKrPGDJCpUix7nzdWjNubdizzDswJsvAcwmlxyq7zAHrfUjHEg48B14h-fAsqRXu-_xWC6up7iYkXahkXpFDvFw-WabPEdaIexVANomIuU0B79WCxL_GJGeOYKpU-pn0rSvksCEI40YSR8f91sRPQf1CifuS.2GdrlpyhHD6T46gZIzi50uq-XSTxhpAhMY6v-drrj88&dib_tag=se&pd_rd_r=5e377116-624c-4aba-807b-f98071971a66&pd_rd_w=80BE5&pd_rd_wg=aWoej&pf_rd_p=77efd490-c71e-4614-b1fe-9c82f75964fe&pf_rd_r=TQQZNXK50GQ69H9CP3GN&qid=1734274260&rd=1&refinements=p_123%3A6832&rnid=85457740011&s=apparel&sr=1-2&th=1'

# Hlavičky pro emulaci prohlížeče
headers = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
}

# Funkce pro získání recenzí z jedné stránky
def get_reviews_from_page(page_url):
    response = requests.get(page_url, headers=headers)
    if response.status_code != 200:
        print(f"Chyba při načítání stránky: {response.status_code}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    reviews = []

    # Hledáme text recenzí v odpovídajících elementech
    review_elements = soup.find_all("span", {"data-hook": "review-body"})
    for element in review_elements:
        review_text = element.get_text(strip=True)
        reviews.append(review_text)

    return reviews

# Iterace přes stránky recenzí
def get_all_reviews(base_url, max_pages=5):
    all_reviews = []
    for page in range(1, max_pages + 1):
        print(f"Načítám stránku {page}...")
        page_url = f"{base_url}?pageNumber={page}"
        reviews = get_reviews_from_page(page_url)
        if not reviews:
            break
        all_reviews.extend(reviews)
        time.sleep(random.uniform(1, 3))  # Náhodné zpoždění

    return all_reviews

# Analýza sentimentu recenzí
def analyze_sentiments(reviews):
    analyzer = SentimentIntensityAnalyzer()
    analyzed_reviews = []

    for review in reviews:
        sentiment_score = analyzer.polarity_scores(review)
        compound = sentiment_score["compound"]

        if compound > 0.05:
            sentiment_label = "Positive"
        elif compound < -0.05:
            sentiment_label = "Negative"
        else:
            sentiment_label = "Neutral"

        analyzed_reviews.append({
            "review": review,
            "sentiment": sentiment_label,
            "score": compound
        })

    return analyzed_reviews

# Hlavní funkce
def main():
    print("Začínám stahování recenzí...")
    reviews = get_all_reviews(url, max_pages=10)
    print(f"Načteno {len(reviews)} recenzí.")

    print("Analyzuji sentimenty...")
    analyzed_reviews = analyze_sentiments(reviews)

    # Uložení výsledků do souboru JSON
    with open("reviews_sentiment.json", "w", encoding="utf-8") as f:
        json.dump(analyzed_reviews, f, ensure_ascii=False, indent=4)

    print("Recenze a sentimenty byly uloženy do 'reviews_sentiment.json'.")

if __name__ == "__main__":
    main()





