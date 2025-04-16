import json
import re


def load_inverted_index(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['inverted_index'], {int(k): v for k, v in data['documents'].items()}


def tokenize_query(query):
    query = query.upper().replace('AND', '&&').replace('OR', '||').replace('NOT', '!')
    return re.findall(r'\(|\)|\w+|\&\&|\|\||!', query)


def to_rpn(tokens):
    precedence = {'!': 3, '&&': 2, '||': 1}
    output = []
    stack = []

    for token in tokens:
        if token == '(':
            stack.append(token)
        elif token == ')':
            while stack and stack[-1] != '(':
                output.append(stack.pop())
            if stack:
                stack.pop()  # remove '('
        elif token in precedence:
            while stack and stack[-1] != '(' and precedence.get(stack[-1], 0) >= precedence[token]:
                output.append(stack.pop())
            stack.append(token)
        else:
            output.append(token.lower())

    while stack:
        output.append(stack.pop())

    return output


def evaluate_rpn(rpn_tokens, inverted_index, all_doc_ids):
    stack = []

    for token in rpn_tokens:
        if token == '&&':
            b = stack.pop()
            a = stack.pop()
            stack.append(a & b)
        elif token == '||':
            b = stack.pop()
            a = stack.pop()
            stack.append(a | b)
        elif token == '!':
            a = stack.pop()
            stack.append(all_doc_ids - a)
        else:
            stack.append(set(inverted_index.get(token, [])))

    return sorted(stack.pop()) if stack else []


def evaluate_query(query, inverted_index, all_doc_ids):
    if not query.strip():
        return []

    try:
        tokens = tokenize_query(query)
        rpn = to_rpn(tokens)
        return evaluate_rpn(rpn, inverted_index, all_doc_ids)
    except Exception as e:
        print(f"[Ошибка] Некорректный запрос: {e}")
        return []


def main_search_loop(inverted_index, default_index):
    all_doc_ids = set(default_index.keys())

    while True:
        try:
            query = input("\nВведите булев запрос (q для выхода): ").strip()
            if query.lower() in ('q', 'quit', 'exit'):
                print("Выход.")
                break

            result = evaluate_query(query, inverted_index, all_doc_ids)

            print(f"\nНайдено документов: {len(result)}")
            for doc_id in result:
                print(f"[{doc_id}] {default_index[doc_id]}")

        except KeyboardInterrupt:
            print("\nВыход по Ctrl+C.")
            break
        except Exception as e:
            print(f"[Ошибка] {e}")


if __name__ == "__main__":
    inverted_index, default_index = load_inverted_index('inverted_index.json')
    main_search_loop(inverted_index, default_index)
