from flask import Flask, render_template, request
from vector_search import VectorSearchEngine

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    query = ""
    results = []
    mode = "tokens"

    if request.method == "POST":
        query = request.form.get("query", "").strip()
        mode = "lemmas" if request.form.get("mode") == "lemmas" else "tokens"

        if query:
            search_engine = VectorSearchEngine(index_dir='tf-idf', mode=mode)
            results = search_engine.search(query)

    return render_template("search.html", query=query, results=results, mode=mode)


if __name__ == "__main__":
    app.run(debug=True)
