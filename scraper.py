import os
import logging
import scrapy
from bs4 import BeautifulSoup
from langdetect import detect
from scrapy.crawler import CrawlerProcess
from urllib.parse import urljoin, urlparse

# ────────────────────────────────────────────────
# Настройки
PAGES_COUNT = 100
SAVE_DIR = 'pages'
INDEX_FILE = 'index.txt'
FULL_DUMP_FILE = 'dump.txt'

TEXT_FILE_EXTENSIONS = ['.html', '.htm', '', '/']
MIN_PAGE_SIZE_BYTES = 100 * 1024


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
# Паук
class ScientificSpider(scrapy.Spider):
    name = "scientific_spider"

    custom_settings = {
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'LOG_LEVEL': 'ERROR'
    }

    start_urls = [
        'https://elementy.ru/',
        'https://nauka.tass.ru/',
        'https://nplus1.ru/',
        'https://postnauka.ru/',
        'https://www.nkj.ru/',
        'https://indicator.ru/',
        'https://chrdk.ru/',
        'https://scientificrussia.ru/',
        'https://kot.sh/',
        'https://22century.ru/'
    ]

    restricted_domains = [
        't.me', 'instagram.com', 'vk.com', 'm.vk.com', 'ok.ru', 'youtube.com',
        'www.youtube.com', 'www.tiktok.com', 'viber.com', 'music.apple.com', 'rutube.ru',
        'www.linkedin.com', 'linkedin.com', 'apps.apple.com', 'www.apple.com', 'github.com',
        'account.ncbi.nlm.nih.gov', 'kudago.com', 'www.zoom.com'
    ]

    restricted_urls = [
        'https://zen.yandex.ru/tolkosprosit',
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.page_counter = 0
        self.visited_urls = set()

    def parse(self, response):
        if self.page_counter >= PAGES_COUNT:
            return

        url = response.url

        if not self.is_valid_url(url) or url in self.visited_urls:
            return

        self.visited_urls.add(url)

        if not self.is_text_page(url):
            return

        if not self.is_valid_scheme(url):
            logging.warning(f"Пропущено {url}: недопустимая схема")
            return

        if not response.headers.get('Content-Type', b'').startswith(b'text'):
            logging.warning(f"Пропущено {url}: бинарный контент")
            return

        if not self.is_correct_language(response.text, expected_lang="ru"):
            logging.warning(f"Пропущено {url}: не русский язык")
            return

        if len(response.body) < MIN_PAGE_SIZE_BYTES:
            logging.warning(f"Пропущено {url}: размер меньше 100 КБ")
            return

        self.page_counter += 1
        self.save_page(response)

        if self.page_counter >= PAGES_COUNT:
            logging.info(f"Достигнут лимит в {PAGES_COUNT} страниц. Остановка.")
            raise scrapy.exceptions.CloseSpider('Reached page limit')

        next_pages = set(response.css('a::attr(href)').getall() + response.xpath("//a/@href").getall())
        next_pages = set(urljoin(url, link) for link in next_pages)

        for next_page in next_pages:
            if (
                    next_page
                    and next_page not in self.visited_urls
                    and self.is_valid_url(next_page)
                    and self.is_valid_scheme(next_page)
            ):
                yield response.follow(next_page, callback=self.parse)

    def save_page(self, response):
        try:
            url_path = urlparse(response.url).path.replace('/', '_')[:40]
            filename = f'{self.page_counter}{url_path}.html'
            filepath = os.path.join(SAVE_DIR, filename)

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(response.text)

            with open(INDEX_FILE, 'a', encoding='utf-8') as index_file:
                index_file.write(f'{self.page_counter},{response.url}\n')

            with open(FULL_DUMP_FILE, 'a', encoding='utf-8') as dump_file:
                dump_file.write(f'FILE {self.page_counter}: {response.url}\n')
                dump_file.write(response.text + '\n' + '=' * 80 + '\n')

            logging.info(f"Сохранено: {filename} ({self.page_counter})")

        except Exception as e:
            logging.error(f"Ошибка при сохранении страницы: {e}")

    def is_valid_url(self, url):
        try:
            domain = urlparse(url).netloc
            if url in self.restricted_urls or any(rd in domain for rd in self.restricted_domains):
                return False
        except Exception:
            return False
        return True

    def is_valid_scheme(self, url):
        parsed = urlparse(url)
        return parsed.scheme in ("http", "https")

    def is_text_page(self, url):
        return any(url.endswith(ext) for ext in TEXT_FILE_EXTENSIONS)

    def is_correct_language(self, html, expected_lang="ru"):
        try:
            text = self.extract_text(html)
            lang = detect(text)
            return lang == expected_lang
        except Exception:
            return False

    def extract_text(self, html):
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text(separator=" ", strip=True)


def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


def main():
    setup_logging()
    create_directory(SAVE_DIR)
    process = CrawlerProcess()
    process.crawl(ScientificSpider)
    process.start()


if __name__ == "__main__":
    main()
