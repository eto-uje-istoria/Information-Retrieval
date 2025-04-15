# 🔍 Information-Retrieval — Search Engine from Scratch

## 📌 Описание проекта

Реализация информационно-поисковой системы: от сбора веб-страниц и обработки текстов до построения индексов, векторного поиска и веб-интерфейса.  
Проект структурирован по этапам — каждый этап представлен в своей ветке.

---

## 📂 Структура проекта

| Ветка | Название | Описание |
|-------|----------|----------|
| [task1](https://github.com/eto-uje-istoria/Information-Retrieval/tree/task1) | Скрапер | Скачивание и сохранение HTML-страниц |
| [task2](https://github.com/eto-uje-istoria/Information-Retrieval/tree/task2) | Лемматизация | Очистка текста, токенизация, лемматизация |
| [task3](https://github.com/eto-uje-istoria/Information-Retrieval/tree/task3) | Индекс | Построение инвертированного индекса и булев поиск |
| [task4](https://github.com/eto-uje-istoria/Information-Retrieval/tree/task4) | TF-IDF | Расчёт tf-idf весов по леммам и терминам |
| [task5](https://github.com/eto-uje-istoria/Information-Retrieval/tree/task5) | Векторный поиск | Поиск документов по векторной модели |
| [demo](https://github.com/eto-uje-istoria/Information-Retrieval/tree/demo) | Веб-интерфейс | UI для ввода запроса и отображения топ-результатов |

---

## 🧰 Технологии

- Python 3.10+
- Flask
- BeautifulSoup
- pymorphy2, NLTK
- NumPy, Scikit-learn
- HTML, CSS, JavaScript

---

## 🚀 Быстрый старт (для demo)

```bash
cd demo
pip install -r requirements.txt
python app.py
```

Перейдите на `http://localhost:5000`, чтобы протестировать поиск.

---

## 👤 Авторы

- Галиахметов Нияз, 11-101
- Бартновский Андрей, 11-102
