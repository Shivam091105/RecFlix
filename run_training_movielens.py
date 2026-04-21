"""
MovieLens → TMDB Adapter Training Script
Converts MovieLens ml-latest-small dataset into the format expected
by the existing MovieRecommenderTrainer and trains the model.

Usage:
    python run_training_movielens.py
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# ── Make sure project root is on the path ─────────────────────────────────────
sys.path.insert(0, '.')

# ── Config ────────────────────────────────────────────────────────────────────
MOVIELENS_DIR = './ml-latest-small'   # folder you extracted
OUTPUT_DIR    = './training/models'   # where model files will be saved
MAX_MOVIES    = None                  # None = all (~9K for small dataset)
QUALITY       = 'low'                 # 'low'=5+ ratings, 'medium'=50+, 'high'=500+


def load_movielens(data_dir: str) -> pd.DataFrame:
    """
    Load and merge MovieLens CSVs into a TMDB-compatible DataFrame.
    The MovieRecommenderTrainer expects these columns:
        id, title, genres, keywords, production_companies,
        production_countries, overview, tagline, vote_average,
        vote_count, popularity, release_date, imdb_id,
        poster_path, status
    """
    data_dir = Path(data_dir)
    print(f"Loading MovieLens data from {data_dir} ...")

    # ── movies.csv ─────────────────────────────────────────────────────────
    movies = pd.read_csv(data_dir / 'movies.csv')
    # columns: movieId, title, genres  (genres pipe-separated)

    # ── ratings.csv ────────────────────────────────────────────────────────
    ratings = pd.read_csv(data_dir / 'ratings.csv')
    # columns: userId, movieId, rating, timestamp

    # ── tags.csv ───────────────────────────────────────────────────────────
    tags = pd.read_csv(data_dir / 'tags.csv')
    # columns: userId, movieId, tag, timestamp

    # ── links.csv ──────────────────────────────────────────────────────────
    links = pd.read_csv(data_dir / 'links.csv')
    # columns: movieId, imdbId, tmdbId

    print(f"  movies   : {len(movies):,} rows")
    print(f"  ratings  : {len(ratings):,} rows")
    print(f"  tags     : {len(tags):,} rows")
    print(f"  links    : {len(links):,} rows")

    # ── Aggregate ratings per movie ────────────────────────────────────────
    rating_agg = (
        ratings
        .groupby('movieId')
        .agg(vote_average=('rating', 'mean'),
             vote_count=('rating', 'count'))
        .reset_index()
    )

    # ── Aggregate tags per movie (used as keywords) ────────────────────────
    tag_agg = (
        tags
        .groupby('movieId')['tag']
        .apply(lambda ts: str(list(ts.str.lower().unique()[:15])))
        .reset_index()
        .rename(columns={'tag': 'keywords_raw'})
    )

    # ── Merge everything ───────────────────────────────────────────────────
    df = (
        movies
        .merge(rating_agg,  on='movieId', how='left')
        .merge(tag_agg,     on='movieId', how='left')
        .merge(links,       on='movieId', how='left')
    )

    # ── Extract year from title, e.g. "Toy Story (1995)" → 1995 ───────────
    df['year'] = df['title'].str.extract(r'\((\d{4})\)$').astype('Int64')
    df['title_clean'] = df['title'].str.replace(r'\s*\(\d{4}\)$', '', regex=True).str.strip()

    # ── Build TMDB-compatible columns ──────────────────────────────────────

    # id
    df['id'] = df['movieId']

    # title  (use cleaned title without year)
    df['title'] = df['title_clean']

    # genres  → pipe-separated string → list-of-dicts format that
    #           parse_json_column can handle via the fallback comma split
    df['genres'] = df['genres'].apply(
        lambda g: str([{'name': genre}
                        for genre in g.split('|')
                        if genre != '(no genres listed)'])
        if pd.notna(g) else '[]'
    )

    # keywords  → already built as string repr of list (from tags)
    df['keywords'] = df['keywords_raw'].fillna('[]')

    # production_companies / production_countries  → empty (not in MovieLens)
    df['production_companies'] = '[]'
    df['production_countries'] = '[]'

    # overview  → use genres as a fallback description
    df['overview'] = df['title'].apply(
        lambda t: f"A movie titled {t}."
    )

    # tagline
    df['tagline'] = ''

    # vote_average / vote_count
    df['vote_average'] = df['vote_average'].fillna(0).round(2)
    df['vote_count']   = df['vote_count'].fillna(0).astype(int)

    # popularity  → use vote_count as proxy
    df['popularity'] = df['vote_count'].astype(float)

    # release_date  → use year
    df['release_date'] = df['year'].apply(
        lambda y: f"{y}-01-01" if pd.notna(y) else ''
    )

    # imdb_id  → MovieLens stores it without "tt" prefix
    df['imdb_id'] = df['imdbId'].apply(
        lambda x: f"tt{int(x):07d}" if pd.notna(x) else ''
    )

    # poster_path  → not available in MovieLens
    df['poster_path'] = ''

    # status  → required by trainer filter; mark all as Released
    df['status'] = 'Released'

    # ── Keep only the columns the trainer needs ────────────────────────────
    keep = [
        'id', 'title', 'genres', 'keywords',
        'production_companies', 'production_countries',
        'overview', 'tagline',
        'vote_average', 'vote_count', 'popularity',
        'release_date', 'imdb_id', 'poster_path', 'status'
    ]
    df = df[keep].copy()

    print(f"  merged df: {len(df):,} rows ready for training")
    return df


def main():
    # ── Check dataset folder exists ────────────────────────────────────────
    if not Path(MOVIELENS_DIR).exists():
        print(f"\n❌  Folder not found: {MOVIELENS_DIR}")
        print("    Download ml-latest-small.zip from:")
        print("    https://files.grouplens.org/datasets/movielens/ml-latest-small.zip")
        print("    Extract it so you have:")
        print(f"    {MOVIELENS_DIR}/movies.csv")
        sys.exit(1)

    # ── Load & convert data ────────────────────────────────────────────────
    df = load_movielens(MOVIELENS_DIR)

    # ── Patch the trainer to accept a DataFrame directly ──────────────────
    from training.train import MovieRecommenderTrainer

    trainer = MovieRecommenderTrainer(
        output_dir=OUTPUT_DIR,
        use_dimensionality_reduction=False,  # Fast for ~9K movies
        n_components=500
    )

    # Monkey-patch load_data so it returns our pre-built DataFrame
    trainer.load_data = lambda _path: df

    # ── Run training pipeline ──────────────────────────────────────────────
    print("\n🎬 Starting training ...\n")
    result_df, sim_matrix = trainer.train(
        data_path=MOVIELENS_DIR,   # passed to load_data (ignored via patch)
        quality_threshold=QUALITY,
        max_movies=MAX_MOVIES,
    )

    print(f"\n✅  Training complete!")
    print(f"   Movies in model : {len(result_df):,}")
    print(f"   Similarity shape: {sim_matrix.shape}")
    print(f"\n👉  Now start the server:")
    print(f"   python manage.py runserver")


if __name__ == '__main__':
    main()
