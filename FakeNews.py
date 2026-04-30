import webview
from newspaper import Article
from ddgs import DDGS
from sentence_transformers import SentenceTransformer, util
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import spacy

model = SentenceTransformer("all-MiniLM-L6-v2")
nlp = spacy.load("en_core_web_sm")

TRUSTED_SOURCES = [
    "bbc.com", "reuters.com", "apnews.com", "thehindu.com",
    "nytimes.com", "theguardian.com", "cnn.com", "forbes.com",
    "ndtv.com", "hindustantimes.com", "indianexpress.com"
]

SATIRE_SOURCES = [
    "theonion.com",
    "babylonbee.com",
    "clickhole.com"
]




def get_article_data(url):
    article = Article(url)
    article.download()
    article.parse()
    return article.title, article.text


def get_keywords(title):
    return " ".join(title.split()[:6])


def search_similar(title):
    query = get_keywords(title)
    results = []

    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=20):
                results.append(r["href"])
    except Exception as e:
        print("Search error:", e)

    return results


def get_page_text(url):
    try:
        article = Article(url)
        article.download()
        article.parse()

        text = article.text.strip()

        if len(text) < 300:
            return ""

        return text

    except:
        return ""


def check_similarity(main_text, urls):
    texts = [main_text]
    valid_urls = []

    for url in urls:
        page_text = get_page_text(url)

        if page_text.strip():
            texts.append(page_text)
            valid_urls.append(url)

        if len(valid_urls) == 10:
            break

    if len(texts) <= 1:
        return [], []

    embeddings = model.encode(texts, convert_to_tensor=True)
    similarities = util.cos_sim(embeddings[0], embeddings[1:])[0]

    return similarities.tolist(), valid_urls


def compare_titles(main_title, urls):
    titles = [main_title]
    valid_title_urls = []

    for url in urls:
        try:
            article = Article(url)
            article.download()
            article.parse()

            if article.title.strip():
                titles.append(article.title)
                valid_title_urls.append(url)

        except:
            continue

    if len(titles) <= 1:
        return [], []

    embeddings = model.encode(titles, convert_to_tensor=True)
    similarities = util.cos_sim(embeddings[0], embeddings[1:])[0]

    return similarities.tolist(), valid_title_urls


def extract_entities(text):
    doc = nlp(text)
    return set(ent.text.lower() for ent in doc.ents)



class API:
    def analyze(self, url):
        try:
            title, text = get_article_data(url)

            if not text.strip():
                return "❌ Could not extract article text."

            original_domain = urlparse(url).netloc

            if any(s in original_domain for s in SATIRE_SOURCES):
                return (
                    f"📰 {title}\n\n"
                    f"🧠 This is a known satire source.\n"
                    f"📊 Verdict: Satire ⚠️\n"
                    f"No further analysis performed."
                )

            similar_urls = search_similar(title)

            if not similar_urls:
                return "⚠️ No similar articles found."

            similarities, valid_urls = check_similarity(text, similar_urls)

            if len(similarities) == 0:
                return "⚠️ Could not compare articles."

            title_similarities, _ = compare_titles(title, valid_urls)

            num_results = min(
                len(valid_urls),
                len(similarities),
                len(title_similarities)
            )

            if num_results == 0:
                return "⚠️ Not enough valid comparisons."

            main_entities = extract_entities(title + " " + text[:1000])

            BODY_THRESHOLD = 0.6
            TITLE_THRESHOLD = 0.6
            ENTITY_THRESHOLD = 1

            high_matches = 0
            trusted_count = 0
            entity_overlaps = []

            for i in range(num_results):
                link = valid_urls[i]
                domain = urlparse(link).netloc

                if domain != original_domain:
                    if any(ts in domain for ts in TRUSTED_SOURCES):
                        trusted_count += 1

                page_text = get_page_text(link)
                article_entities = extract_entities(page_text[:1000])

                overlap = len(main_entities & article_entities)
                entity_overlaps.append(overlap)

                score = 0

                if similarities[i] > BODY_THRESHOLD:
                    score += 1

                if title_similarities[i] > TITLE_THRESHOLD:
                    score += 1

                if overlap >= ENTITY_THRESHOLD:
                    score += 2

                if score >= 3:
                    high_matches += 1

            same_source_match = any(
                urlparse(link).netloc == original_domain
                for link in valid_urls[:num_results]
            )

            title_matches = sum(
                sim > TITLE_THRESHOLD
                for sim in title_similarities[:num_results]
            )

            if high_matches >= 3 and trusted_count >= 2:
                verdict = "Reliable News ✅"

            elif high_matches >= 3:
                verdict = "Probably REAL ✅"

            elif same_source_match and trusted_count >= 1:
                verdict = "Source Verified ℹ️"

            elif trusted_count >= 2:
                verdict = "Likely REAL ✅"

            elif high_matches == 0 and trusted_count == 0:
                verdict = "Suspicious ⚠️"

            elif high_matches <= 2 and trusted_count == 0:
                verdict = "Uncertain 🤔"
            else:
                verdict = "Uncertain 🤔"

            output = (
                f"📰 {title}\n\n"
                f"📊 Verdict: {verdict}\n"
                f"🔗 Matches: {high_matches}/{num_results}\n"
                f"📰 Title Matches: {title_matches}/{num_results}\n"
                f"🏛 Trusted Sources: {trusted_count}\n\n"
                f"Similarity Scores:\n"
            )

            for i in range(num_results):
                output += (
                    f"{i+1}. "
                    f"Body: {similarities[i]:.2f} | "
                    f"Title: {title_similarities[i]:.2f} | "
                    f"Entities: {entity_overlaps[i]}\n"
                )

            output += "\n🌐 Sources:\n"
            for i in range(num_results):
                output += f"{i+1}. {urlparse(valid_urls[i]).netloc}\n"

            output += "\n🔗 Matching Article Links:\n"
            for i in range(num_results):
                output += f"{i+1}. {valid_urls[i]}\n"

            return output

        except Exception as e:
            return f"❌ Error: {e}"

    def quit(self):
        window.destroy()



api = API()

window = webview.create_window(
    "Fake News Detector",
    "index.html",
    js_api=api,
    width=950,
    height=750
)

webview.start()