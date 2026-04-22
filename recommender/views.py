
"""
Movie Recommendation System Views
"""
import ast
import json
import logging
import os
import threading
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_http_methods
from rapidfuzz import fuzz
from rapidfuzz import process as fuzz_process

logger = logging.getLogger(__name__)

_RECOMMENDER = None
_MODEL_LOADING = False
_MODEL_LOAD_PROGRESS = 0
_LOADING_THREAD = None
_LOAD_ERROR = None


def _parse_genres(val) -> list:
    """Convert any genre format to a plain list of genre name strings."""
    if isinstance(val, list):
        out = []
        for item in val:
            if isinstance(item, dict):
                name = item.get("name", "")
                if name:
                    out.append(str(name))
            elif isinstance(item, str) and item.strip():
                out.append(item.strip())
        return out

    if isinstance(val, str):
        s = val.strip()
        if not s or s in ("[]", "nan", "None"):
            return []
        # Try JSON
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return _parse_genres(parsed)
        except Exception:
            pass
        # Try Python literal
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return _parse_genres(parsed)
        except Exception:
            pass
        # Plain comma-separated fallback
        parts = []
        for token in s.strip("[]").split(","):
            clean = token.strip().strip(chr(39)).strip(chr(34)).strip()
            if clean:
                parts.append(clean)
        return parts

    return []


def _fmt_rating(vote_average) -> str:
    """MovieLens is 0-5 scale; multiply by 2 to display as /10."""
    try:
        v = float(vote_average)
        if v <= 5.0:
            v = v * 2
        return "{:.1f}/10".format(v)
    except Exception:
        return "N/A"


class MovieRecommender:
    def __init__(self, model_dir="models", progress_callback=None):
        self.model_dir = Path(model_dir)
        self.metadata = None
        self.svd_vectors = None
        self.title_to_idx = None
        self.genre_index = None
        self.config = None
        self._mode = "svd"
        self._load_models(progress_callback)

    def _load_models(self, progress_callback=None):
        def prog(p):
            if progress_callback:
                progress_callback(p)

        logger.info("Loading models from %s", self.model_dir)
        prog(10)

        self.metadata = pd.read_parquet(self.model_dir / "movie_metadata.parquet")
        prog(30)

        vec_path = self.model_dir / "svd_vectors.npy"
        if vec_path.exists():
            self.svd_vectors = np.load(vec_path)
            self._mode = "svd"
        elif (self.model_dir / "similarity_matrix.npz").exists():
            from scipy.sparse import load_npz
            self.svd_vectors = load_npz(self.model_dir / "similarity_matrix.npz").toarray()
            self._mode = "matrix"
        else:
            self.svd_vectors = np.load(self.model_dir / "similarity_matrix.npy")
            self._mode = "matrix"
        prog(65)

        with open(self.model_dir / "title_to_idx.json") as f:
            self.title_to_idx = json.load(f)
        prog(80)

        genre_path = self.model_dir / "genre_index.json"
        if genre_path.exists():
            with open(genre_path) as f:
                self.genre_index = json.load(f)
        else:
            self.genre_index = {}
            for idx, genres in enumerate(self.metadata["genres"]):
                for g in _parse_genres(genres):
                    self.genre_index.setdefault(g, []).append(idx)
        prog(90)

        with open(self.model_dir / "config.json") as f:
            self.config = json.load(f)
        prog(100)

        logger.info("Loaded %s movies | mode=%s", self.config["n_movies"], self._mode)

    def _sim_scores_for(self, movie_idx: int) -> np.ndarray:
        if self._mode == "svd":
            vec = self.svd_vectors[movie_idx]
            return self.svd_vectors @ vec
        return self.svd_vectors[movie_idx]

    def find_movie_exact(self, title: str) -> Optional[str]:
        lower_map = {k.lower(): k for k in self.title_to_idx}
        return lower_map.get(title.lower())

    def fuzzy_match(self, title: str, n: int = 5) -> List[Dict]:
        results = fuzz_process.extract(
            title,
            self.title_to_idx.keys(),
            scorer=fuzz.WRatio,
            limit=n,
            score_cutoff=40,
        )
        return [{"title": r[0], "score": round(r[1], 1)} for r in results]

    def search_movies(self, query: str, n: int = 20) -> List[str]:
        q = query.lower()
        return [t for t in self.title_to_idx if q in t.lower()][:n]

    def get_all_genres(self) -> List[str]:
        return sorted(self.genre_index.keys())

    def _production(self, val) -> str:
        if pd.isna(val):
            return "Independent"
        s = str(val)
        if s.endswith("Studios"):
            return "Independent"
        return s

    def _movie_dict(self, idx: int, score: float = None) -> Dict:
        m = self.metadata.iloc[idx]
        genres_list = _parse_genres(m["genres"])
        # Fallback: look up genre_index if parse returned nothing
        if not genres_list and self.genre_index:
            genres_list = [g for g, idxs in self.genre_index.items() if idx in idxs][:4]
        genres_str = ", ".join(genres_list[:3]) if genres_list else "N/A"
        title = str(m["title"])
        imdb = str(m["imdb_id"]) if pd.notna(m["imdb_id"]) else None
        return {
            "title": title,
            "release_date": str(m["release_date"])[:10] if pd.notna(m["release_date"]) else "Unknown",
            "production": self._production(m["primary_company"]),
            "genres": genres_str,
            "genres_list": genres_list[:4],
            "rating": _fmt_rating(m["vote_average"]),
            "votes": "{:,}".format(int(m["vote_count"])) if pd.notna(m["vote_count"]) else "N/A",
            "similarity_score": "{:.3f}".format(score) if score is not None else None,
            "imdb_id": imdb,
            "poster_url": None,
            "google_link": "https://www.google.com/search?q=" + "+".join(title.split()) + "+movie",
            "imdb_link": "https://www.imdb.com/title/" + imdb if imdb else None,
        }

    def get_by_genre(self, genre: str, n: int = 20, min_rating: float = 0) -> List[Dict]:
        indices = self.genre_index.get(genre, [])
        movies = []
        for idx in indices:
            m = self.metadata.iloc[idx]
            rating_val = float(m["vote_average"])
            display_rating = rating_val * 2 if rating_val <= 5.0 else rating_val
            if display_rating < min_rating:
                continue
            genres_list = _parse_genres(m["genres"])
            title = str(m["title"])
            imdb = str(m["imdb_id"]) if pd.notna(m["imdb_id"]) else None
            movies.append({
                "title": title,
                "release_date": str(m["release_date"])[:10] if pd.notna(m["release_date"]) else "Unknown",
                "production": self._production(m["primary_company"]),
                "genres": ", ".join(genres_list[:3]) if genres_list else "N/A",
                "rating": _fmt_rating(m["vote_average"]),
                "votes": "{:,}".format(int(m["vote_count"])),
                "imdb_id": imdb,
                "poster_url": None,
                "google_link": "https://www.google.com/search?q=" + "+".join(title.split()) + "+movie",
                "imdb_link": "https://www.imdb.com/title/" + imdb if imdb else None,
            })
        movies.sort(key=lambda x: float(x["rating"].replace("/10", "") or 0), reverse=True)
        return movies[:n]

    def get_recommendations(self, movie_title: str, n: int = 15, min_rating: float = None) -> Dict:
        matched = self.find_movie_exact(movie_title)
        auto_corrected = False

        if not matched:
            fuzzy_hits = self.fuzzy_match(movie_title, n=5)
            if fuzzy_hits and fuzzy_hits[0]["score"] >= 85:
                matched = fuzzy_hits[0]["title"]
                auto_corrected = True
            else:
                return {
                    "error": "Movie '{}' not found in the dataset.".format(movie_title),
                    "fuzzy_suggestions": fuzzy_hits,
                    "all_genres": self.get_all_genres(),
                }

        movie_idx = self.title_to_idx[matched]
        sim_scores = self._sim_scores_for(movie_idx)
        source = self.metadata.iloc[movie_idx]
        source_genres = _parse_genres(source["genres"])

        ranked = np.argsort(sim_scores)[::-1]
        recommendations = []
        for idx in ranked:
            if int(idx) == movie_idx:
                continue
            if len(recommendations) >= n:
                break
            m = self.metadata.iloc[idx]
            if min_rating is not None:
                rv = float(m["vote_average"])
                display = rv * 2 if rv <= 5.0 else rv
                if display < min_rating:
                    continue
            recommendations.append(self._movie_dict(int(idx), float(sim_scores[idx])))

        return {
            "query_movie": matched,
            "auto_corrected": auto_corrected,
            "original_query": movie_title if auto_corrected else None,
            "source_movie": {
                "production": self._production(source["primary_company"]),
                "rating": _fmt_rating(source["vote_average"]),
                "genres": ", ".join(source_genres[:3]) if source_genres else "N/A",
            },
            "recommendations": recommendations,
            "all_genres": self.get_all_genres(),
        }


# ── Background loading ────────────────────────────────────────────────────────

def _load_model_in_background():
    global _RECOMMENDER, _MODEL_LOADING, _MODEL_LOAD_PROGRESS, _LOAD_ERROR
    _MODEL_LOADING = True
    _MODEL_LOAD_PROGRESS = 0
    _LOAD_ERROR = None

    model_dir = getattr(settings, "MODEL_DIR", os.environ.get("MODEL_DIR", "training/models"))
    if not Path(model_dir).exists():
        model_dir = "static"
        logger.warning("Model directory not found, falling back to static/")

    try:
        def cb(p):
            global _MODEL_LOAD_PROGRESS
            _MODEL_LOAD_PROGRESS = p
            logger.info("Model loading progress: %s%%", p)

        _RECOMMENDER = MovieRecommender(model_dir, cb)
        _MODEL_LOADING = False
        _MODEL_LOAD_PROGRESS = 100
        logger.info("Model loaded successfully")
    except Exception as e:
        _MODEL_LOADING = False
        _LOAD_ERROR = str(e)
        logger.error("Failed to load recommender: %s", e)


def _start_model_loading():
    global _LOADING_THREAD, _RECOMMENDER, _MODEL_LOADING
    if _RECOMMENDER is None and not _MODEL_LOADING:
        if _LOADING_THREAD is None or not _LOADING_THREAD.is_alive():
            logger.info("Starting model loading in background...")
            _LOADING_THREAD = threading.Thread(target=_load_model_in_background, daemon=True)
            _LOADING_THREAD.start()


def _get_recommender():
    global _RECOMMENDER, _LOAD_ERROR
    if _RECOMMENDER is None:
        _start_model_loading()
        if _LOAD_ERROR:
            raise Exception(_LOAD_ERROR)
        return None
    return _RECOMMENDER


# ── Views ─────────────────────────────────────────────────────────────────────

@require_http_methods(["GET", "POST"])
def main(request):
    _start_model_loading()
    recommender = _get_recommender()

    loading_ctx = {"all_movie_names": [], "total_movies": 0}

    if recommender is None:
        if request.method == "GET":
            return render(request, "recommender/index.html", loading_ctx)
        return render(request, "recommender/index.html", {
            **loading_ctx,
            "error_message": "Model is still loading. Please wait and try again.",
        })

    titles_list = list(recommender.title_to_idx.keys())
    all_genres = recommender.get_all_genres()
    base_ctx = {
        "all_movie_names": titles_list,
        "total_movies": len(titles_list),
        "all_genres": all_genres,
    }

    if request.method == "GET":
        return render(request, "recommender/index.html", base_ctx)

    movie_name = request.POST.get("movie_name", "").strip()
    if not movie_name:
        return render(request, "recommender/index.html", {
            **base_ctx,
            "error_message": "Please enter a movie name.",
        })

    result = recommender.get_recommendations(movie_name, n=15)

    if "error" in result:
        return render(request, "recommender/index.html", {
            **base_ctx,
            "input_movie_name": movie_name,
            "error_message": result["error"],
            "fuzzy_suggestions": result.get("fuzzy_suggestions", []),
            "show_genre_search": True,
        })

    return render(request, "recommender/result.html", {
        **base_ctx,
        "input_movie_name": result["query_movie"],
        "auto_corrected": result.get("auto_corrected", False),
        "original_query": result.get("original_query"),
        "source_movie": result["source_movie"],
        "recommended_movies": result["recommendations"],
        "total_recommendations": len(result["recommendations"]),
    })


@require_http_methods(["GET"])
def search_movies(request):
    query = request.GET.get("q", "").strip()
    if len(query) < 2:
        return JsonResponse({"movies": [], "count": 0})
    try:
        rec = _get_recommender()
        if rec is None:
            return JsonResponse({"movies": [], "count": 0, "loading": True})
        return JsonResponse({"movies": rec.search_movies(query, n=20), "count": 0})
    except Exception as e:
        logger.error("Search error: %s", e)
        return JsonResponse({"error": "Search failed"}, status=500)


@require_http_methods(["GET"])
def genre_search(request):
    genre = request.GET.get("genre", "").strip()
    n = int(request.GET.get("n", 20))
    min_rating = float(request.GET.get("min_rating", 0))

    if not genre:
        return JsonResponse({"error": "genre parameter required"}, status=400)

    try:
        rec = _get_recommender()
        if rec is None:
            return JsonResponse({"movies": [], "loading": True})
        movies = rec.get_by_genre(genre, n=n, min_rating=min_rating)
        return JsonResponse({"genre": genre, "movies": movies, "count": len(movies)})
    except Exception as e:
        logger.error("Genre search error: %s", e)
        return JsonResponse({"error": str(e)}, status=500)


@require_http_methods(["GET"])
def model_status(request):
    global _RECOMMENDER, _MODEL_LOADING, _MODEL_LOAD_PROGRESS, _LOAD_ERROR
    _start_model_loading()
    if _LOAD_ERROR:
        return JsonResponse({"loaded": False, "progress": 0, "status": "error", "error": _LOAD_ERROR})
    elif _RECOMMENDER is not None:
        return JsonResponse({"loaded": True, "progress": 100, "status": "ready"})
    elif _MODEL_LOADING:
        return JsonResponse({"loaded": False, "progress": _MODEL_LOAD_PROGRESS, "status": "loading"})
    else:
        return JsonResponse({"loaded": False, "progress": 0, "status": "initializing"})


@require_http_methods(["GET"])
def health_check(request):
    try:
        rec = _get_recommender()
        return JsonResponse({
            "status": "healthy",
            "movies_loaded": rec.config["n_movies"],
            "model_dir": str(rec.model_dir),
            "model_loaded": True,
        })
    except Exception as e:
        return JsonResponse({"status": "unhealthy", "error": str(e)}, status=503)