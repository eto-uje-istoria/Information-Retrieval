import os
import json
from collections import defaultdict

LEMMAS_DIR = "data/lemmas"
INDEX_PATH = "index.txt"
OUTPUT_PATH = "inverted_index.json"


def load_index(index_path):
    doc_ids = {}
    with open(index_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(',', 1)
            if len(parts) == 2:
                doc_id, url = parts
                doc_ids[int(doc_id)] = url
    return doc_ids


def build_inverted_index(lemmas_dir):
    inverted_index = defaultdict(set)

    for filename in sorted(os.listdir(lemmas_dir)):
        if not filename.endswith('-lemmas.txt'):
            continue

        try:
            doc_id = int(filename.split('-')[0])
        except ValueError:
            continue

        file_path = os.path.join(lemmas_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                lemma = line.strip().split()[0]
                if lemma:
                    inverted_index[lemma].add(doc_id)

    return {k: sorted(list(v)) for k, v in inverted_index.items()}


def save_inverted_index(inverted_index, default_index, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('{\n')
        f.write('  "inverted_index": {\n')

        lemmas = sorted(inverted_index.items())
        for i, (lemma, doc_ids_list) in enumerate(lemmas):
            ids_str = ', '.join(str(doc_id) for doc_id in sorted(doc_ids_list))
            comma = ',' if i < len(lemmas) - 1 else ''
            f.write(f'    "{lemma}": [{ids_str}]{comma}\n')

        f.write('  },\n')
        f.write('  "documents": ')
        json.dump(default_index, f, ensure_ascii=False, indent=2)
        f.write('\n}')


def main():
    default_index = load_index(INDEX_PATH)
    inverted_index = build_inverted_index(LEMMAS_DIR)
    save_inverted_index(inverted_index, default_index, OUTPUT_PATH)
    print("Инвертированный индекс успешно построен.")


if __name__ == "__main__":
    main()
