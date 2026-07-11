"""
enrich.py — Phase 2: metadata enrichment via OMDb (+ optional TMDb helpers).

NOTE: This module requires internet access to www.omdbapi.com and needs a
free OMDb API key (https://www.omdbapi.com/apikey.aspx). It cannot run
inside the Claude sandbox; run it on your own machine:

    export OMDB_API_KEY=your_key_here
    python src/enrich.py "Interstellar"   # fetch one title (poster, genre,
                                           # director, runtime, plot, etc.)

All responses are cached in cache/tmdb_cache.json so repeat lookups are
instant and don't re-hit the API.
"""
import requests
import os, json, time, sys, urllib.request, urllib.parse
OMDB_KEY = os.getenv("OMDB_API_KEY")
TMDB_KEY = os.getenv("TMDB_API_KEY")   # was missing entirely — fetch_by_imdb_id/poster_url need this

BASE = "https://api.themoviedb.org/3"
IMG_BASE = "https://image.tmdb.org/t/p/w342"
CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "cache", "tmdb_cache.json")


def _cache():
    if os.path.exists(CACHE_PATH):
        return json.load(open(CACHE_PATH))
    return {}


def _save_cache(c):
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    json.dump(c, open(CACHE_PATH, "w"))


def _get(url):
    response = requests.get(
        url,
        headers={"User-Agent": "rating-predictor"},
        timeout=15
    )

    response.raise_for_status()

    return response.json()


def fetch_by_imdb_id(imdb_id, cache=None):
    """Find TMDb entry from IMDb tt-id, then pull full details + credits + keywords."""
    cache = cache if cache is not None else _cache()
    if imdb_id in cache:
        return cache[imdb_id]
    if not TMDB_KEY:
        raise RuntimeError("Set TMDB_API_KEY environment variable first.")
    find = _get(f"{BASE}/find/{imdb_id}?api_key={TMDB_KEY}&external_source=imdb_id")
    hits = find.get("movie_results", [])
    if not hits:
        cache[imdb_id] = None
        _save_cache(cache)
        return None
    tmdb_id = hits[0]["id"]
    detail = _get(f"{BASE}/movie/{tmdb_id}?api_key={TMDB_KEY}"
                  f"&append_to_response=credits,keywords")
    rec = {
        "tmdb_id": tmdb_id,
        "overview": detail.get("overview", ""),
        "tagline": detail.get("tagline", ""),
        "original_language": detail.get("original_language", ""),
        "production_countries": [c["name"] for c in detail.get("production_countries", [])],
        "budget": detail.get("budget", 0),
        "revenue": detail.get("revenue", 0),
        "collection": (detail.get("belongs_to_collection") or {}).get("name", ""),
        "tmdb_popularity": detail.get("popularity", 0.0),
        "cast_top5": [c["name"] for c in detail.get("credits", {}).get("cast", [])[:5]],
        "writers": [c["name"] for c in detail.get("credits", {}).get("crew", [])
                    if c.get("job") in ("Writer", "Screenplay")][:4],
        "keywords": [k["name"] for k in detail.get("keywords", {}).get("keywords", [])][:15],
    }
    cache[imdb_id] = rec
    _save_cache(cache)
    time.sleep(0.25)  # be polite to the API
    return rec


def poster_url(title, year=None, cache=None):
    """Look up a TMDb poster by title (+year). Returns a full image URL or None
    (no key set, no match, or network error) so callers can fall back to a
    placeholder card instead of breaking."""
    cache = cache if cache is not None else _cache()
    ck = f"poster::{title.strip().lower()}::{year or ''}"
    if ck in cache:
        return cache[ck]
    if not TMDB_KEY:
        return None
    try:
        q = f"api_key={TMDB_KEY}&query={urllib.parse.quote(title)}"
        if year:
            q += f"&year={year}"
        data = _get(f"{BASE}/search/movie?{q}")
        results = data.get("results", [])
        path = results[0].get("poster_path") if results else None
        url = f"{IMG_BASE}{path}" if path else None
    except Exception:
        url = None
    cache[ck] = url
    _save_cache(cache)
    return url


def search_title(title, year=None, cache=None):
    cache = cache if cache is not None else _cache()
    ck = f"omdb::{title.strip().lower()}::{year or ''}"
    if ck in cache:
        return cache[ck]

    if not OMDB_KEY:
        raise RuntimeError("Set OMDB_API_KEY environment variable first.")

    url = f"https://www.omdbapi.com/?apikey={OMDB_KEY}&t={urllib.parse.quote(title)}"

    if year:
        url += f"&y={year}"

    data = requests.get(url, timeout=15).json()

    if data.get("Response") == "False":
        cache[ck] = None
        _save_cache(cache)
        return None

    directors = []
    if data.get("Director") and data["Director"] != "N/A":
        directors = [d.strip() for d in data["Director"].split(",")]

    genres = []
    if data.get("Genre") and data["Genre"] != "N/A":
        genres = [g.strip().replace("Science Fiction", "Sci-Fi")
                  for g in data["Genre"].split(",")]

    try:
        imdb_rating = float(data.get("imdbRating", 7.0))
    except:
        imdb_rating = 7.0

    try:
        num_votes = int(data.get("imdbVotes", "1000").replace(",", ""))
    except:
        num_votes = 1000

    try:
        runtime = int(data.get("Runtime", "120 min").split()[0])
    except:
        runtime = 120

    poster = data.get("Poster")
    if not poster or poster == "N/A":
        poster = None

    rec = {
        "title": data.get("Title"),
        "year": int(data.get("Year", "2024")[:4]),
        "runtime": runtime,
        "genres": genres,
        "directors": directors,
        "tmdb_vote": imdb_rating,
        "tmdb_votes": num_votes,
        "overview": data.get("Plot", ""),
        "poster": poster,
        "imdb_rating_display": data.get("imdbRating", "N/A"),
    }
    cache[ck] = rec
    _save_cache(cache)
    return rec


def enrich_csv(csv_in, csv_out):
    import pandas as pd
    df = pd.read_csv(csv_in)
    df = df[df["Title Type"] == "Movie"].copy()
    cache = _cache()
    recs = []
    for i, cid in enumerate(df["Const"]):
        rec = fetch_by_imdb_id(cid, cache) or {}
        recs.append(rec)
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(df)} fetched")
    extra = pd.DataFrame(recs, index=df.index)
    out = pd.concat([df, extra], axis=1)
    os.makedirs(os.path.dirname(csv_out), exist_ok=True)
    out.to_csv(csv_out, index=False)
    print(f"Enriched CSV written to {csv_out}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(json.dumps(search_title(" ".join(sys.argv[1:])), indent=2))
    else:
        enrich_csv("data/imdb_data.csv", "data/enriched.csv")
