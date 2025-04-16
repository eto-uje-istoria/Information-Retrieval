import os
import re
import html
import logging
from collections import defaultdict

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import nltk
import pymorphy3

# ────────────────────────────────────────────────
# Настройки
SOURCE_DIR = 'pages'
TOKENS_DIR = 'data/tokens'
LEMMAS_DIR = 'data/lemmas'

VALID_POS = {'NOUN', 'ADJF', 'ADJS', 'VERB', 'INFN', 'PRTF', 'PRTS', 'GRND'}
CUSTOM_STOPWORDS = {'т.д.', 'др.', 'т.п.'}
MIN_WORD_LENGTH = 3


# ────────────────────────────────────────────────
# Цветной логгер
class ColorFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[94m',
        'INFO': '\033[92m',
        'WARNING': '\033[93m',
        'ERROR': '\033[91m',
        'CRITICAL': '\033[1;91m'
    }
    RESET = '\033[0m'

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        message = super().format(record)
        return f"{color}{message}{self.RESET}"


def setup_logging():
    handler = logging.StreamHandler()
    formatter = ColorFormatter("[%(levelname)s] %(message)s")
    handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = [handler]


# ────────────────────────────────────────────────
# Предзагрузка
nltk.download("stopwords", quiet=True)
russian_stopwords = set(stopwords.words("russian"))
russian_stopwords.update(CUSTOM_STOPWORDS)

morph = pymorphy3.MorphAnalyzer()


# ────────────────────────────────────────────────
def ensure_directories():
    os.makedirs(TOKENS_DIR, exist_ok=True)
    os.makedirs(LEMMAS_DIR, exist_ok=True)


def extract_text(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    for tag in soup(['script', 'style', 'meta', 'link', 'noscript']):
        tag.decompose()
    return soup.get_text(separator=' ', strip=True)


def clean_text(text):
    text = html.unescape(text)
    text = re.sub(r'[^а-яА-ЯёЁ\s-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()


def tokenize(text):
    return re.findall(r'\b(?:[а-яё]+-)*[а-яё]+\b', text)


def process_tokens(words):
    valid_tokens = set()
    for word in words:
        parts = word.split('-')
        for part in parts:
            if len(part) < MIN_WORD_LENGTH:
                continue
            parsed = morph.parse(part)
            if any(p.score > 0.3 and p.tag.POS in VALID_POS and len(p.normal_form) > 2 for p in parsed):
                if part not in russian_stopwords:
                    valid_tokens.add(part)
    return sorted(valid_tokens)


def get_lemmas(tokens):
    lemmas = defaultdict(set)
    for token in tokens:
        lemma = morph.parse(token)[0].normal_form
        lemmas[lemma].add(token)
    return lemmas


def save_tokens(doc_id, tokens):
    path = os.path.join(TOKENS_DIR, f'{doc_id}-tokens.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(tokens))


def save_lemmas(doc_id, lemmas):
    path = os.path.join(LEMMAS_DIR, f'{doc_id}-lemmas.txt')
    with open(path, 'w', encoding='utf-8') as f:
        for lemma in sorted(lemmas):
            forms = sorted(lemmas[lemma])
            if lemma in forms:
                forms.remove(lemma)
            forms.insert(0, lemma)
            f.write(f"{' '.join(forms)}\n")


def process_file(filename, doc_ids):
    if not filename.endswith('.html'):
        return False

    try:
        doc_id = int(filename.split('_')[0])
    except (ValueError, IndexError):
        logging.warning(f"Пропущен файл с некорректным ID: {filename}")
        return False

    if doc_id in doc_ids:
        logging.warning(f"Обнаружен дубликат ID {doc_id} в файле {filename}")
        return False

    file_path = os.path.join(SOURCE_DIR, filename)
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        html_content = f.read()

    text = extract_text(html_content)
    cleaned = clean_text(text)
    words = tokenize(cleaned)
    tokens = process_tokens(words)
    lemmas = get_lemmas(tokens)

    save_tokens(doc_id, tokens)
    save_lemmas(doc_id, lemmas)

    logging.info(f"Обработан документ {doc_id} ({filename})")
    doc_ids.add(doc_id)
    return True


def main():
    setup_logging()
    ensure_directories()
    doc_ids = set()
    files = sorted(os.listdir(SOURCE_DIR))
    count = 0
    for filename in files:
        if process_file(filename, doc_ids):
            count += 1

    logging.info(f"\nОбработка завершена. Всего успешно обработано: {count} файлов.")


if __name__ == "__main__":
    main()
