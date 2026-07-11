"""
app.py — Phase 7: Streamlit prediction interface.

Run locally:
    export TMDB_API_KEY=your_key
    streamlit run app.py

Deploy: push this folder to GitHub -> share.streamlit.io -> pick repo -> add
TMDB_API_KEY as a secret. Done.
"""
import os, sys, json
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from predict import Predictor

st.set_page_config(page_title="Will I Like It?", page_icon="🎬", layout="centered")
st.title("🎬 Will I Like It?")
st.caption("Personalized rating prediction trained on 414 of your IMDb ratings. "
           "OOF MAE ≈ 0.94 — treat every prediction as ±1 point.")

@st.cache_resource
def load_model():
    return Predictor(os.path.join(os.path.dirname(__file__), "models", "rating_predictor.joblib"))

P = load_model()

if "history" not in st.session_state:
    st.session_state.history = []

tab_search, tab_manual = st.tabs(["Search by title (TMDb)", "Manual metadata"])

with tab_search:
    title = st.text_input("Movie title", placeholder="e.g. The Prestige")
    if st.button("Predict", key="b1") and title:
        try:
            from enrich import search_title
            md = search_title(title)
            if md is None:
                st.error("Title not found on TMDb.")
            else:
                md["imdb_rating"] = md.pop("tmdb_vote") or 7.0
                md["num_votes"] = max(md.pop("tmdb_votes", 1000) * 40, 1000)  # rough TMDb->IMDb scale
                res = P.predict_metadata(md)
                st.session_state.history.insert(0, res)
        except RuntimeError as e:
            st.warning(f"{e} — falling back to manual mode (use the second tab).")

with tab_manual:
    c1, c2 = st.columns(2)
    with c1:
        m_title = st.text_input("Title", key="mt")
        m_imdb = st.number_input("IMDb rating", 1.0, 10.0, 7.5, 0.1)
        m_votes = st.number_input("Num votes", 100, 5_000_000, 100_000, step=1000)
    with c2:
        m_year = st.number_input("Year", 1930, 2030, 2024)
        m_runtime = st.number_input("Runtime (mins)", 40, 300, 130)
    m_genres = st.text_input("Genres (comma separated)", "Thriller, Mystery")
    m_dirs = st.text_input("Directors (comma separated)", "")
    if st.button("Predict", key="b2") and m_title:
        md = dict(title=m_title, imdb_rating=m_imdb, num_votes=int(m_votes),
                  runtime=m_runtime, year=int(m_year),
                  genres=[g.strip() for g in m_genres.split(",") if g.strip()],
                  directors=[d.strip() for d in m_dirs.split(",") if d.strip()])
        st.session_state.history.insert(0, P.predict_metadata(md))

# ----- render results -----
for res in st.session_state.history[:5]:
    with st.container(border=True):
        st.subheader(f"{res['title']} — {res['recommendation']}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted rating", f"{res['predicted_rating']}/10")
        c2.metric("P(you like it)", f"{res['probability_like']*100:.0f}%")
        c3.metric("Confidence", f"{res['confidence']*100:.0f}%")
        if res["reasons_positive"]:
            st.markdown("**Pushing the prediction up:**")
            for r in res["reasons_positive"]:
                st.markdown(f"- {r}")
        if res["reasons_negative"]:
            st.markdown("**Pulling it down:**")
            for r in res["reasons_negative"]:
                st.markdown(f"- {r}")
        st.markdown("**Closest movies in your own history:**")
        for s in res["similar_movies_from_your_history"]:
            st.markdown(f"- {s['title']} ({s['year']}) — you gave it {s['your_rating']}/10 "
                        f"(similarity {s['similarity']})")
