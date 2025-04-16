import os
import re
import numpy as np
from collections import defaultdict, Counter
import pymorphy3

# ────────────────────────────────────────────────
# Стоп-слова
custom_stopwords = {'т.д.', 'др.', 'т.п.'}


# ────────────────────────────────────────────────
# Класс поисковой системы на основе TF-IDF
class VectorSearchEngine:
    def __init__(self, index_dir: str, mode: str):
        self.mode = mode
        self.index_dir = os.path.join(index_dir, mode)
        self.documents = self._load_document_names()
        self.term_idf = {}
        self.term_index = {}
        self.doc_vectors = {}
        self.morph = pymorphy3.MorphAnalyzer()

        self._load_index()

    def _load_document_names(self):
        docs = {}
        for filename in os.listdir('pages'):
            if filename.endswith('.html'):
                doc_id = filename.split('_')[0]
                docs[doc_id] = filename
        return docs

    def _load_index(self):
        terms = set()

        for filename in os.listdir(self.index_dir):
            with open(os.path.join(self.index_dir, filename), 'r', encoding='utf-8') as f:
                for line in f:
                    term, idf, _ = line.strip().split()
                    self.term_idf[term] = float(idf)
                    terms.add(term)

        self.term_index = {term: idx for idx, term in enumerate(sorted(terms))}

        for filename in os.listdir(self.index_dir):
            doc_id = filename.split('-')[0]
            vector = defaultdict(float)
            with open(os.path.join(self.index_dir, filename), 'r', encoding='utf-8') as f:
                for line in f:
                    term, _, tfidf = line.strip().split()
                    idx = self.term_index[term]
                    vector[idx] = float(tfidf)
            self.doc_vectors[doc_id] = vector

    def _process_term(self, word: str) -> str:
        if self.mode == 'lemmas':
            return self.morph.parse(word)[0].normal_form
        return word

    def _tokenize_query(self, query: str):
        query = query.lower()
        words = re.findall(r'\b[а-яё-]+\b', query)
        tokens = []

        for word in words:
            for part in word.split('-'):
                term = self._process_term(part)
                if len(term) > 2 and term not in custom_stopwords:
                    tokens.append(term)
        return tokens

    def _vectorize_query(self, query_terms):
        term_counts = Counter(query_terms)
        total_terms = len(query_terms)
        query_vector = defaultdict(float)

        for term, count in term_counts.items():
            if term in self.term_index:
                tf = count / total_terms
                idf = self.term_idf.get(term, 0.0)
                query_vector[self.term_index[term]] = tf * idf
        return query_vector

    def search(self, query: str, top_n=10):
        query_terms = self._tokenize_query(query)
        if not query_terms:
            return []

        query_vector = self._vectorize_query(query_terms)
        query_norm = np.linalg.norm(list(query_vector.values()))
        if query_norm < 1e-9:
            return []

        scores = {}
        for doc_id, doc_vector in self.doc_vectors.items():
            doc_norm = np.linalg.norm(list(doc_vector.values()))
            if doc_norm < 1e-9:
                continue

            common = set(doc_vector.keys()) & set(query_vector.keys())
            if not common:
                continue

            dot_product = sum(doc_vector[i] * query_vector[i] for i in common)
            cosine = dot_product / (doc_norm * query_norm)

            if cosine > 1e-6:
                scores[doc_id] = cosine

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return [(self.documents[doc_id], score) for doc_id, score in sorted_scores if doc_id in self.documents]


# ────────────────────────────────────────────────
# Точка входа
def main():
    mode = input("Выберите режим поиска (tokens/lemmas): ").strip().lower()
    while mode not in ('tokens', 'lemmas'):
        print("Некорректный режим! Допустимые значения: tokens, lemmas")
        mode = input("Выберите режим поиска (tokens/lemmas): ").strip().lower()

    search_engine = VectorSearchEngine('tf-idf', mode=mode)

    print("\nПоисковая система готова к работе!")
    print("Введите поисковый запрос (или 'exit' для выхода):")

    while True:
        query = input("\n> ").strip()
        if query.lower() == 'exit':
            break

        results = search_engine.search(query)

        if not results:
            print("Ничего не найдено.")
        else:
            print(f"\nНайдено документов: {len(results)}")
            for idx, (filename, score) in enumerate(results, 1):
                print(f"{idx}. [Сходство: {score:.4f}] {filename}")


if __name__ == '__main__':
    main()
