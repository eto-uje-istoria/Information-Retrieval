import os
import re
import html
import math
import logging
from collections import Counter, defaultdict
from bs4 import BeautifulSoup

# ────────────────────────────────────────────────
# Пути к директориям
PAGES_DIR = 'pages/'
TOKENS_DIR = 'data/tokens/'
LEMMAS_DIR = 'data/lemmas/'
OUTPUT_TOKENS = 'tf-idf/tokens/'
OUTPUT_LEMMAS = 'tf-idf/lemmas/'


# ────────────────────────────────────────────────
# Настройка цветного логгера
class ColorFormatter(logging.Formatter):
    COLORS = {
        'INFO': '\033[92m',
        'WARNING': '\033[93m',
        'ERROR': '\033[91m',
        'RESET': '\033[0m'
    }

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        return f"{color}[{record.levelname}] {record.getMessage()}{self.COLORS['RESET']}"


logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
logging.getLogger().handlers[0].setFormatter(ColorFormatter())


# ────────────────────────────────────────────────
# Вспомогательные функции
def extract_text(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    for tag in soup(['script', 'style', 'meta', 'link', 'noscript']):
        tag.decompose()
    return soup.get_text(separator=' ', strip=True)


def clean_text(text):
    text = html.unescape(text)
    text = re.sub(r'[^а-яё\s-]', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()


def compute_tf(word_counts, total_words):
    return {word: count / total_words for word, count in word_counts.items()}


def compute_idf(doc_freq, total_docs):
    return {word: math.log(total_docs / (freq + 1)) for word, freq in doc_freq.items()}


# ────────────────────────────────────────────────
# Обработка одного документа
def process_document(doc_id, html_filename, token_idf, lemma_idf):
    html_path = os.path.join(PAGES_DIR, html_filename)

    try:
        with open(html_path, 'r', encoding='utf-8', errors='ignore') as f:
            raw_html = f.read()
    except FileNotFoundError:
        logging.warning(f"Файл не найден: {html_path}")
        return

    cleaned_text = clean_text(extract_text(raw_html))
    words = []
    for word in cleaned_text.split():
        words.extend(word.split('-'))

    # Токены
    tokens_path = os.path.join(TOKENS_DIR, f"{doc_id}-tokens.txt")
    with open(tokens_path, 'r', encoding='utf-8') as f:
        valid_tokens = [line.strip() for line in f if line.strip()]

    token_counts = Counter(word for word in words if word in valid_tokens)
    total_token_words = sum(token_counts.values()) or 1
    tf_tokens = compute_tf(token_counts, total_token_words)

    # Леммы
    lemmas_path = os.path.join(LEMMAS_DIR, f"{doc_id}-lemmas.txt")
    lemma_forms = defaultdict(list)
    with open(lemmas_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                lemma = parts[0]
                forms = parts[1:] if len(parts) > 1 else [lemma]
                lemma_forms[lemma] = forms

    lemma_counts = Counter()
    for lemma, forms in lemma_forms.items():
        lemma_counts[lemma] += sum(words.count(form) for form in forms)

    total_lemma_words = sum(lemma_counts.values()) or 1
    tf_lemmas = compute_tf(lemma_counts, total_lemma_words)

    os.makedirs(OUTPUT_TOKENS, exist_ok=True)
    os.makedirs(OUTPUT_LEMMAS, exist_ok=True)

    with open(os.path.join(OUTPUT_TOKENS, f"{doc_id}-tfidf.txt"), 'w', encoding='utf-8') as f:
        for token in valid_tokens:
            tf = tf_tokens.get(token, 0.0)
            idf = token_idf.get(token, 0.0)
            f.write(f"{token} {idf:.6f} {tf * idf:.6f}\n")

    with open(os.path.join(OUTPUT_LEMMAS, f"{doc_id}-tfidf.txt"), 'w', encoding='utf-8') as f:
        for lemma in lemma_forms:
            tf = tf_lemmas.get(lemma, 0.0)
            idf = lemma_idf.get(lemma, 0.0)
            f.write(f"{lemma} {idf:.6f} {tf * idf:.6f}\n")

    logging.info(f"TF-IDF по токенам и лемма рассчитан для документа с ID: {doc_id}")


# ────────────────────────────────────────────────
# Основной поток выполнения
def main():
    os.makedirs(OUTPUT_TOKENS, exist_ok=True)
    os.makedirs(OUTPUT_LEMMAS, exist_ok=True)

    # Определяем doc_ids из имён файлов токенов
    token_files = sorted(f for f in os.listdir(TOKENS_DIR) if f.endswith('-tokens.txt'))
    doc_ids = [f.split('-')[0] for f in token_files]
    total_docs = len(doc_ids)

    if total_docs == 0:
        logging.error("Нет токенов в data/tokens/")
        return

    token_doc_freq = Counter()
    lemma_doc_freq = Counter()

    for doc_id in doc_ids:
        tokens_path = os.path.join(TOKENS_DIR, f"{doc_id}-tokens.txt")
        with open(tokens_path, 'r', encoding='utf-8') as f:
            tokens = [line.strip() for line in f if line.strip()]
            token_doc_freq.update(set(tokens))

        lemmas_path = os.path.join(LEMMAS_DIR, f"{doc_id}-lemmas.txt")
        with open(lemmas_path, 'r', encoding='utf-8') as f:
            for line in f:
                lemma = line.strip().split()[0]
                lemma_doc_freq[lemma] += 1

    token_idf = compute_idf(token_doc_freq, total_docs)
    lemma_idf = compute_idf(lemma_doc_freq, total_docs)

    # Ищем HTML-файлы по doc_id
    html_files = {f.split('_')[0]: f for f in os.listdir(PAGES_DIR) if f.endswith('.html')}

    for doc_id in doc_ids:
        html_filename = html_files.get(doc_id)
        if html_filename:
            process_document(doc_id, html_filename, token_idf, lemma_idf)
        else:
            logging.warning(f"Не найден HTML-файл для doc_id={doc_id}")

    logging.info("TF-IDF обработка завершена.")


if __name__ == "__main__":
    main()
