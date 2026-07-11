import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from flask import Flask, render_template, request
from predict import Predictor
from enrich import search_title, _cache

app = Flask(__name__)

P = Predictor(
    os.path.join(
        os.path.dirname(__file__),
        "models",
        "rating_predictor.joblib"
    )
)


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    error = None

    if request.method == "POST":
        title = request.form.get("movie", "").strip()

        if not title:
            error = "Type a title first."
        else:
            try:
                md = search_title(title)

                if md is None:
                    error = f"Couldn't find \u201c{title}\u201d. Check the spelling, or try the full title."
                else:
                    # movie_details drives the top info card + the "Movie Details" section
                    movie_details = {
                        "poster": md.get("poster"),
                        "genres": md.get("genres", []),
                        "directors": md.get("directors", []),
                        "runtime": md.get("runtime"),
                        "imdb_rating": md.get("imdb_rating_display", "N/A"),
                        "plot": md.get("overview", ""),
                        "year": md.get("year"),
                    }

                    predict_md = dict(md)
                    predict_md["imdb_rating"] = predict_md.pop("tmdb_vote") or 7.0
                    predict_md["num_votes"] = max(predict_md.pop("tmdb_votes", 1000) * 40, 1000)

                    result = P.predict_metadata(predict_md)
                    result["details"] = movie_details

                    # attach a poster to each similar movie too (shared cache = fast)
                    cache = _cache()
                    for sim in result["similar_movies_from_your_history"]:
                        sim_md = search_title(sim["title"], sim["year"], cache=cache)
                        sim["poster"] = sim_md.get("poster") if sim_md else None

            except Exception:
                error = "Something went wrong looking that title up. Try again in a moment."

    return render_template(
        "index.html",
        result=result,
        error=error,
    )


if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug_mode)
