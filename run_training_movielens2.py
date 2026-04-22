"""
MovieLens → TMDB Adapter Training Script (v3 - Memory Efficient)
- Saves SVD reduced vectors (n×50) instead of n×n similarity matrix
- Trains on all ~40K quality movies with no memory issues
- Total model size: ~50MB instead of 3.4GB
- Similarity computed on-the-fly per query: O(n×50) instead of O(n²)

Usage:
    python run_training_movielens.py
"""

import pandas as pd
import numpy as np
import sys
import json
import pickle
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from nltk.stem.snowball import SnowballStemmer
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '.')

# ── Config ────────────────────────────────────────────────────────────────────
MOVIELENS_DIR = './ml-latest'
OUTPUT_DIR    = './training/models'
MAX_MOVIES    = None      # None = all quality-filtered movies (~40K)
QUALITY_MIN_VOTES = 5     # minimum ratings a movie must have
N_COMPONENTS  = 300      # SVD dims — n×50 stored, dot-product at query time
# ─────────────────────────────────────────────────────────────────────────────


def genres_to_json(pipe_genres: str) -> str:
    if pd.isna(pipe_genres) or pipe_genres.strip() == '(no genres listed)':
        return '[]'
    return json.dumps([{"name": g.strip()} for g in pipe_genres.split('|') if g.strip()])


def tags_to_keywords_json(tag_series) -> str:
    if tag_series is None or len(tag_series) == 0:
        return '[]'
    unique_tags = list(dict.fromkeys(str(t).lower().strip() for t in tag_series))[:15]
    return json.dumps([{"name": t} for t in unique_tags if t])


def parse_names(json_str):
    try:
        return [item['name'] for item in json.loads(json_str) if 'name' in item]
    except Exception:
        return []


def load_movielens(data_dir: str) -> pd.DataFrame:
    data_dir = Path(data_dir)
    print(f"Loading MovieLens data from: {data_dir.resolve()}")

    movies  = pd.read_csv(data_dir / 'movies.csv')
    ratings = pd.read_csv(data_dir / 'ratings.csv')
    links   = pd.read_csv(data_dir / 'links.csv')
    print(f"  movies : {len(movies):,} | ratings: {len(ratings):,}")

    tags_path = data_dir / 'tags.csv'
    if tags_path.exists():
        tags = pd.read_csv(tags_path)
        print(f"  tags   : {len(tags):,}")
        tag_agg = (
            tags.groupby('movieId')['tag']
            .apply(tags_to_keywords_json)
            .reset_index()
            .rename(columns={'tag': 'keywords'})
        )
    else:
        tag_agg = pd.DataFrame(columns=['movieId', 'keywords'])

    rating_agg = (
        ratings.groupby('movieId')
        .agg(vote_average=('rating', 'mean'), vote_count=('rating', 'count'))
        .reset_index()
    )

    df = (
        movies
        .merge(rating_agg, on='movieId', how='left')
        .merge(tag_agg,    on='movieId', how='left')
        .merge(links,      on='movieId', how='left')
    )

    # Extract year and clean title
    df['year'] = (
        df['title'].str.extract(r'\((\d{4})\)$')[0]
        .apply(pd.to_numeric, errors='coerce').astype('Int64')
    )
    df['title'] = df['title'].str.replace(r'\s*\(\d{4}\)$', '', regex=True).str.strip()

    df['id']     = df['movieId']
    df['status'] = 'Released'
    df['genres'] = df['genres'].apply(genres_to_json)
    df['keywords'] = df['keywords'].fillna('[]')

    # Derive production company from primary genre
    df['production_companies'] = df['genres'].apply(
        lambda g: json.dumps([{"name": (parse_names(g)[0].replace(' ','') + ' Studios') if parse_names(g) else 'Independent'}])
    )
    df['production_countries'] = '[]'

    # Overview
    def make_overview(row):
        genres = parse_names(row['genres'])
        genre_str = ', '.join(genres) if genres else 'various genres'
        year = f" ({row['year']})" if pd.notna(row.get('year')) else ''
        return f"{row['title']}{year} is a {genre_str} film."

    df['overview']     = df.apply(make_overview, axis=1)
    df['tagline']      = ''
    df['vote_average'] = df['vote_average'].fillna(0).round(2)
    df['vote_count']   = df['vote_count'].fillna(0).astype(int)
    df['popularity']   = df['vote_count'].astype(float)
    df['release_date'] = df['year'].apply(lambda y: f"{int(y)}-01-01" if pd.notna(y) else '')
    df['imdb_id']      = df['imdbId'].apply(lambda x: f"tt{int(x):07d}" if pd.notna(x) else '')
    df['poster_path']  = ''

    keep = [
        'id', 'title', 'genres', 'keywords', 'production_companies',
        'production_countries', 'overview', 'tagline', 'vote_average',
        'vote_count', 'popularity', 'release_date', 'imdb_id', 'poster_path', 'status'
    ]
    df = df[keep].copy()
    print(f"  merged : {len(df):,} movies ready\n")
    return df


def build_soup(row, stemmer):
    genres    = parse_names(row['genres'])
    keywords  = parse_names(row['keywords'])
    companies = parse_names(row['production_companies'])
    overview  = str(row['overview']).split()[:50]

    stemmed_kw   = [stemmer.stem(k.lower().replace(' ', '')) for k in keywords[:15]]
    clean_genres = [g.lower().replace(' ', '') for g in genres]
    clean_comp   = [c.lower().replace(' ', '') for c in companies[:3]]
    comp_weighted = [clean_comp[0]] * 2 if clean_comp else []

    parts = (
        stemmed_kw +
        clean_genres * 2 +
        comp_weighted + clean_comp +
        [w.lower() for w in overview]
    )
    return ' '.join(parts)


def main():
    if not Path(MOVIELENS_DIR).exists():
        print(f"\nERROR: Folder not found: {MOVIELENS_DIR}")
        print("Download ml-latest.zip from:")
        print("  https://files.grouplens.org/datasets/movielens/ml-latest.zip")
        sys.exit(1)

    # ── Load ──────────────────────────────────────────────────────────────
    df = load_movielens(MOVIELENS_DIR)

    # ── Quality filter ────────────────────────────────────────────────────
    df = df[df['vote_count'] >= QUALITY_MIN_VOTES].copy()
    df = df[df['status'] == 'Released'].copy()
    print(f"After quality filter ({QUALITY_MIN_VOTES}+ votes): {len(df):,} movies")

    # ── Extra metadata columns ────────────────────────────────────────────
    df['genres_list']  = df['genres'].apply(parse_names)
    df['primary_company'] = df['production_companies'].apply(
        lambda x: parse_names(x)[0] if parse_names(x) else None
    )

    # ── Quality sort ──────────────────────────────────────────────────────
    df['quality_score'] = df['vote_average'] * np.log1p(df['vote_count'])
    df = df.sort_values('quality_score', ascending=False)
    if MAX_MOVIES:
        df = df.head(MAX_MOVIES)
        print(f"Limited to top {MAX_MOVIES} movies")
    df = df.reset_index(drop=True)

    # ── Soup ──────────────────────────────────────────────────────────────
    stemmer = SnowballStemmer('english')
    print("Building feature soup...")
    df['soup'] = df.apply(lambda r: build_soup(r, stemmer), axis=1)
    df = df[df['soup'].str.len() > 20].dropna(subset=['title'])
    df = df.drop_duplicates(subset=['title'], keep='first').reset_index(drop=True)
    print(f"Final dataset: {len(df):,} movies")

    # ── TF-IDF ────────────────────────────────────────────────────────────
    n_movies     = len(df)
    max_features = 10000 if n_movies < 10000 else 15000 if n_movies < 100000 else 20000
    print(f"Building TF-IDF (max_features={max_features})...")
    tfidf = TfidfVectorizer(
        analyzer='word', ngram_range=(1, 2),
        min_df=3, max_df=0.7,
        stop_words='english', max_features=max_features,
        sublinear_tf=True
    )
    tfidf_matrix = tfidf.fit_transform(df['soup'])
    print(f"TF-IDF shape: {tfidf_matrix.shape}")

    # ── SVD — store vectors only, NOT the n×n matrix ─────────────────────
    n_comp = min(N_COMPONENTS, tfidf_matrix.shape[0] - 1, tfidf_matrix.shape[1] - 1)
    print(f"Running SVD (n_components={n_comp}) ...")
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    vectors = svd.fit_transform(tfidf_matrix).astype(np.float32)   # (n, n_comp)

    # L2-normalise: cosine_similarity(a, B) = a · B^T when both are unit vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    vectors /= norms

    print(f"SVD vectors shape : {vectors.shape}")
    print(f"Explained variance: {svd.explained_variance_ratio_.sum():.3f}")
    print(f"Memory: {vectors.nbytes/1024**2:.1f} MB  "
          f"(full matrix would be {n_movies**2*4/1024**2:.0f} MB)")

    # ── Save ──────────────────────────────────────────────────────────────
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    # Metadata parquet
    meta = df[[
        'id', 'title', 'release_date', 'primary_company',
        'genres_list', 'vote_average', 'vote_count', 'popularity',
        'overview', 'imdb_id', 'poster_path'
    ]].rename(columns={'genres_list': 'genres'})
    meta.to_parquet(out / 'movie_metadata.parquet', compression='gzip', index=True)

    # SVD vectors — the only matrix we store
    np.save(out / 'svd_vectors.npy', vectors)

    # Title → index
    with open(out / 'title_to_idx.json', 'w') as f:
        json.dump(pd.Series(df.index, index=df['title']).to_dict(), f)

    # Genre → indices (for genre-based search in views.py)
    genre_index = {}
    for idx, genres in enumerate(df['genres_list']):
        for g in genres:
            genre_index.setdefault(g, []).append(idx)
    with open(out / 'genre_index.json', 'w') as f:
        json.dump(genre_index, f)

    # Pickles
    with open(out / 'tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf, f)
    with open(out / 'svd_model.pkl', 'wb') as f:
        pickle.dump(svd, f)

    # Config
    config = {
        'n_movies': len(df),
        'use_svd': True,
        'n_components': n_comp,
        'storage': 'svd_vectors',   # signals views.py to use on-demand mode
        'dataset': 'MovieLens ml-latest'
    }
    with open(out / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    total_mb = sum(f.stat().st_size for f in out.iterdir() if f.is_file()) / 1024**2
    print(f"\nSaved to {out}  (total: {total_mb:.1f} MB)")
    print(f"\nDone! {len(df):,} movies trained.")
    print("Start the server:  python manage.py runserver")


if __name__ == '__main__':
    main()