"""Step 9 - Streamlit dashboard over the real pipeline outputs (Gold layer + ALS outputs).

    streamlit run dashboard/app.py
"""
import glob
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

DATA = Path(__file__).resolve().parent.parent / "data"
st.set_page_config(page_title="Movie Recommender - Big Data Analytics", page_icon="🎬", layout="wide")


@st.cache_data
def load(rel):
    p = DATA / rel
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


@st.cache_data
def load_metrics():
    files = glob.glob(str(DATA / "output" / "metrics" / "part-*"))
    return json.loads(open(files[0]).read()) if files else {}


metrics = load_metrics()
kpis, movie_stats, users = load("gold/yearly_kpis"), load("gold/movie_stats"), load("gold/user_activity")
if movie_stats.empty:
    st.error(f"No pipeline outputs found in {DATA}. Run `python run_pipeline.py` or scripts/dataproc_run.ps1 first.")
    st.stop()

st.title("🎬 Movie Recommendation System using Big Data Analytics")
st.caption(f"Dataset: MovieLens **{metrics.get('dataset', '?')}** (GroupLens Research) · "
           "Pipeline: PySpark Bronze → Silver → Gold → MLlib ALS on Google Cloud Dataproc")

c = st.columns(6)
c[0].metric("Ratings", f"{int(kpis.num_ratings.sum()):,}")
c[1].metric("Users", f"{len(users):,}")
c[2].metric("Movies rated", f"{len(movie_stats):,}")
c[3].metric("Avg rating", f"{(kpis.num_ratings * kpis.avg_rating).sum() / kpis.num_ratings.sum():.2f}")
c[4].metric("Model RMSE", metrics.get("test_rmse", "–"),
            delta=f"{metrics['test_rmse'] - metrics['baseline_rmse_global_mean']:.3f} vs baseline" if metrics else None,
            delta_color="inverse")
c[5].metric("Precision@10", metrics.get("precision_at_10", "–"))

tabs = st.tabs(["📈 Trends", "🎭 Movies & Genres", "👥 Users", "⭐ Recommendations", "🧠 Model", "🧹 Data Quality"])

with tabs[0]:
    k = kpis.sort_values("rating_year")
    l, r = st.columns(2)
    l.plotly_chart(px.bar(k, x="rating_year", y="num_ratings", title="Ratings per year"), use_container_width=True)
    r.plotly_chart(px.line(k, x="rating_year", y="active_users", markers=True, title="Active users per year"),
                   use_container_width=True)
    gt = load("gold/genre_trends")
    top_genres = gt.groupby("genre").num_ratings.sum().nlargest(8).index
    l, r = st.columns(2)
    l.plotly_chart(px.line(gt[gt.genre.isin(top_genres)].sort_values("rating_year"), x="rating_year", y="num_ratings",
                           color="genre", title="Ratings per year – top 8 genres"), use_container_width=True)
    r.plotly_chart(px.line(gt[gt.genre.isin(top_genres)].sort_values("rating_year"), x="rating_year", y="avg_rating",
                           color="genre", title="Average rating per year – top 8 genres"), use_container_width=True)
    dist = load("gold/rating_distribution")
    st.plotly_chart(px.bar(dist, x="rating", y="count", title="Rating distribution"), use_container_width=True)

with tabs[1]:
    tg = load("gold/top_movies_by_genre")
    genre = st.selectbox("Genre", sorted(tg.genre.unique()), index=sorted(tg.genre.unique()).index("Drama")
                         if "Drama" in set(tg.genre) else 0)
    st.dataframe(tg[tg.genre == genre].sort_values("rank")[
                     ["rank", "clean_title", "release_year", "num_ratings", "avg_rating", "weighted_rating"]],
                 hide_index=True, use_container_width=True)
    l, r = st.columns(2)
    most = movie_stats.nlargest(15, "num_ratings").sort_values("num_ratings")
    l.plotly_chart(px.bar(most, x="num_ratings", y="clean_title", orientation="h", title="Most-rated movies"),
                   use_container_width=True)
    tags = load("gold/top_tags").head(20).sort_values("uses")
    r.plotly_chart(px.bar(tags, x="uses", y="tag", orientation="h", title="Top user tags"), use_container_width=True)
    sim = load("output/similar_movies")
    if not sim.empty:
        st.subheader("Because you watched… (item-item similarity from ALS factors)")
        pick = st.selectbox("Movie", sorted(sim.clean_title.unique()))
        st.dataframe(sim[sim.clean_title == pick].sort_values("rank")[["rank", "similar_title", "similarity"]],
                     hide_index=True, use_container_width=True)

with tabs[2]:
    l, r = st.columns(2)
    seg = users.segment.value_counts().reset_index()
    l.plotly_chart(px.pie(seg, names="segment", values="count", title="User segments (by #ratings)"),
                   use_container_width=True)
    r.plotly_chart(px.histogram(users, x="avg_rating", nbins=40, title="Distribution of users' average rating"),
                   use_container_width=True)
    st.plotly_chart(px.histogram(users[users.num_ratings < users.num_ratings.quantile(0.99)], x="num_ratings",
                                 nbins=60, title="Ratings per user (99th pct cut)"), use_container_width=True)

with tabs[3]:
    recs, fav = load("output/user_recs_top10"), load("gold/user_top_rated")
    ids = set(recs.userId.unique())
    uid = int(st.number_input(f"User ID (1 – {max(ids):,})", min_value=int(min(ids)), max_value=int(max(ids)),
                              value=int(min(ids)), step=1))
    if uid not in ids:
        st.warning("No recommendations for this user ID.")
    else:
        u = users[users.userId == uid].iloc[0]
        st.write(f"**{u.num_ratings:,}** ratings · avg **{u.avg_rating:.2f}** · segment **{u.segment}**")
        l, r = st.columns(2)
        l.subheader("Their favourite movies")
        l.dataframe(fav[fav.userId == uid].sort_values("rank")[["clean_title", "rating"]], hide_index=True,
                    use_container_width=True)
        r.subheader("Top-10 recommendations (unseen)")
        rr = recs[recs.userId == uid].sort_values("rank").copy()
        rr["genres"] = rr.genres.apply(lambda g: ", ".join(g) if g is not None else "")
        r.dataframe(rr[["rank", "clean_title", "release_year", "genres", "predicted_rating"]], hide_index=True,
                    use_container_width=True)

with tabs[4]:
    if metrics:
        c = st.columns(4)
        c[0].metric("Test RMSE", metrics["test_rmse"])
        c[1].metric("Test MAE", metrics["test_mae"])
        c[2].metric("Baseline RMSE (global mean)", metrics["baseline_rmse_global_mean"])
        c[3].metric("NDCG@10", metrics["ndcg_at_10"])
        st.write(f"Best ALS params: rank **{metrics['best_rank']}**, regParam **{metrics['best_regParam']}** · "
                 f"train/val/test rows: {metrics['train_rows']:,} / {metrics['val_rows']:,} / {metrics['test_rows']:,}")
        grid = pd.DataFrame(metrics["grid"])
        grid["params"] = "rank=" + grid["rank"].astype(str) + ", reg=" + grid.regParam.astype(str)
        st.plotly_chart(px.bar(grid, x="params", y="val_rmse", title="Hyper-parameter search (validation RMSE)"),
                        use_container_width=True)
    pred = load("output/test_predictions_sample")
    if not pred.empty:
        pred["error"] = pred.prediction - pred.rating
        l, r = st.columns(2)
        l.plotly_chart(px.box(pred, x="rating", y="prediction", title="Predicted vs actual rating (test sample)"),
                       use_container_width=True)
        r.plotly_chart(px.histogram(pred, x="error", nbins=60, title="Prediction error distribution"),
                       use_container_width=True)

with tabs[5]:
    dq = load("silver/dq_report")
    st.subheader("Silver-layer cleaning report")
    st.dataframe(dq, hide_index=True, use_container_width=True)
