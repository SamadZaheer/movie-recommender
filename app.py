import streamlit as st
import numpy as np
import pickle
import requests

st.set_page_config(
    page_title="Movie Recommender | Samad Zaheer",
    page_icon="🎬",
    layout="wide"
)

st.markdown("""
<style>
#MainMenu, footer, header {visibility: hidden;}
[data-testid="stMarkdownContainer"] p { margin-bottom: 0 !important; margin-top: 0 !important; }
[data-testid="stMarkdownContainer"] img { display: block; margin-bottom: 0 !important; }
.movie-title {
    font-size: 13px;
    font-weight: 600;
    color: #e6f1ff;
    margin-top: 4px;
    line-height: 1.4;
}
.movie-meta { font-size: 11px; color: #8892b0; margin-top: 1px; }
.no-poster {
    background: #112240;
    border: 1px solid #1e3a5f;
    border-radius: 6px;
    height: 250px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 48px;
}
.overview-text { font-size: 14px; color: #a8b2d8; font-style: italic; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading movie database...")
def load_data():
    cosine_sim = np.load('cosine_sim2.npz')['matrix']
    with open('movie_data.pkl', 'rb') as f:
        movie_data = pickle.load(f)
    with open('indices.pkl', 'rb') as f:
        indices = pickle.load(f)
    return cosine_sim, movie_data, indices

@st.cache_data(show_spinner=False)
def fetch_poster(movie_id, api_key):
    if not api_key:
        return None
    try:
        url = f"https://api.themoviedb.org/3/movie/{int(movie_id)}?api_key={api_key}"
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        poster_path = r.json().get('poster_path')
        if poster_path:
            return f"https://image.tmdb.org/t/p/w342{poster_path}"
    except Exception:
        pass
    return None

def get_recommendations(title, cosine_sim, movie_data, indices):
    idx = indices[title]
    sim_scores = sorted(enumerate(cosine_sim[idx]), key=lambda x: x[1], reverse=True)[1:11]
    return movie_data.iloc[[i[0] for i in sim_scores]]

cosine_sim, movie_data, indices = load_data()

try:
    api_key = st.secrets["TMDB_API_KEY"]
except (KeyError, FileNotFoundError):
    api_key = ""
    st.warning("⚠️ TMDB_API_KEY not set in secrets — posters will not display.")

all_titles = sorted(movie_data['title'].tolist())

st.markdown("""
<div style="padding: 1.5rem 0 1rem 0;">
    <h1 style="color: white; font-weight: 700; margin: 0;">🎬 Movie Recommender</h1>
    <p style="color: #a8b2c1; margin: 0.4rem 0 0 0; font-size: 1rem;">Discover films you'll love — matched by cast, director, genre & keywords.</p>
</div>
""", unsafe_allow_html=True)
st.divider()

selected = st.selectbox(
    "Search for a movie",
    options=all_titles,
    index=None,
    placeholder="Start typing a movie title..."
)

if selected:
    sel = movie_data[movie_data['title'] == selected].iloc[0]

    col_img, col_info = st.columns([1, 4])
    with col_img:
        poster = fetch_poster(sel['id'], api_key)
        if poster:
            st.markdown(f'<img src="{poster}" width="180" style="border-radius:6px;display:block;">', unsafe_allow_html=True)
        else:
            st.markdown('<div style="background:#112240;border:1px solid #1e3a5f;border-radius:6px;width:180px;height:270px;display:flex;align-items:center;justify-content:center;font-size:48px;">🎬</div>', unsafe_allow_html=True)
    with col_info:
        year_str = f" ({int(sel['year'])})" if sel['year'] > 0 else ""
        st.markdown(f"## {sel['title']}{year_str}")
        if sel['vote_count'] >= 10:
            rating_html = f'<span style="color:#5eead4;font-weight:600;">TMDB</span> <strong>{sel["vote_average"]:.1f}</strong>'
        else:
            rating_html = '<span style="color:#8892b0;">No rating</span>'
        st.markdown(f'{rating_html} &nbsp;|&nbsp; {sel["genres_display"]}', unsafe_allow_html=True)
        if sel['director_display']:
            st.markdown(f"🎬 **Director:** {sel['director_display']}")
        if sel['cast_display']:
            st.markdown(f"🎭 **Cast:** {sel['cast_display']}")
        overview = str(sel['overview']) if sel['overview'] and str(sel['overview']) != 'nan' else ""
        if overview:
            preview = overview[:280] + "…" if len(overview) > 280 else overview
            st.markdown(f'<p class="overview-text">{preview}</p>', unsafe_allow_html=True)

    st.divider()
    st.markdown(f"### Movies similar to **{selected}**")

    recs = get_recommendations(selected, cosine_sim, movie_data, indices)

    with st.spinner("Fetching posters..."):
        posters = [fetch_poster(r['id'], api_key) for _, r in recs.iterrows()]

    cols = st.columns(5)
    for i, (poster_url, (_, row)) in enumerate(zip(posters, recs.iterrows())):
        with cols[i % 5]:
            year_str = f" ({int(row['year'])})" if row['year'] > 0 else ""
            genres = row['genres_display'] if row['genres_display'] else "—"
            if poster_url:
                img_html = f'<div style="aspect-ratio:2/3;overflow:hidden;border-radius:6px;"><img src="{poster_url}" style="width:100%;height:100%;object-fit:cover;display:block;"></div>'
            else:
                img_html = '<div style="background:#112240;border:1px solid #1e3a5f;border-radius:6px;aspect-ratio:2/3;display:flex;align-items:center;justify-content:center;font-size:48px;">🎬</div>'
            rating_str = f'<span style="color:#5eead4;">TMDB</span> {row["vote_average"]:.1f}' if row["vote_count"] >= 10 else '<span style="color:#8892b0;">No rating</span>'
            st.markdown(f'''
<div style="margin-bottom:20px;">
    {img_html}
    <p style="font-size:15px;font-weight:600;color:#e6f1ff;margin:6px 0 3px 0;line-height:1.4;">{row["title"]}{year_str}</p>
    <p style="font-size:13px;color:#8892b0;margin:0;">{rating_str} · {genres}</p>
</div>''', unsafe_allow_html=True)

st.divider()
st.markdown(
    '<p style="text-align:center;color:#8892b0;font-size:13px;margin-bottom:4px;">'
    'Dataset: TMDB 5000 &nbsp;·&nbsp; 4,803 movies &nbsp;·&nbsp; '
    'Content-Based Filtering &nbsp;·&nbsp; CountVectorizer + Cosine Similarity'
    '</p>'
    '<p style="text-align:center;color:#8892b0;font-size:13px;margin-top:0;">'
    'Built by <a href="https://samadzaheer.github.io" target="_blank" '
    'style="color:#5eead4;text-decoration:none;">Samad Zaheer</a>'
    '</p>',
    unsafe_allow_html=True
)
