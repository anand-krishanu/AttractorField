"""Step 9 - MovieLens insights dashboard (Streamlit) over the real pipeline outputs.

One page per business question:
  Overview         - leadership: platform health and recommender impact
  Content          - content/acquisition: genres, titles, tags, similar movies
  Audience         - growth/marketing: user segments and engagement
  Recommendations  - personalisation/CX: a user's taste vs. their Top-10
  Model quality    - data science: accuracy, tuning, error analysis
  Data pipeline    - data engineering: medallion layers, data quality, job runtimes

    streamlit run dashboard/app.py
"""
import glob
import json
import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
DATA, EVIDENCE = ROOT / "data", ROOT / "docs" / "cloud_run_evidence"
ACCENT, NEUTRAL, GRID, TEXT2 = "#185FA5", "#C9CDD3", "#ECEEF1", "#5F6B7A"
SEGMENTS = [("casual", "Casual", "under 35 ratings"), ("regular", "Regular", "35 to 99"),
            ("active", "Active", "100 to 499"), ("power", "Power", "500 or more")]

st.set_page_config(page_title="MovieLens insights", layout="wide")
st.markdown("""<style>
#MainMenu, footer, [data-testid="stDeployButton"], [data-testid="stAppDeployButton"], [data-testid="stMainMenu"] {display: none;}
header[data-testid="stHeader"] {background: transparent;}
.block-container {padding-top: 2.2rem; max-width: 1240px;}
[data-testid="stSidebarNav"]::before {content: "MovieLens insights\\A 25M ratings, Dataproc run"; white-space: pre;
  display: block; padding: 0 1rem 1rem; font-size: 15px; font-weight: 600; line-height: 1.5; color: #1F2328;}
.pt {font-size: 26px !important; font-weight: 600; margin: 0 !important; padding: 0;}
.ps {font-size: 14px !important; color: #5F6B7A; margin: 2px 0 18px !important;}
.kpi {background: #F5F6F8; border-radius: 10px; padding: 14px 16px; height: 100%;}
.kl {font-size: 13px !important; color: #5F6B7A; margin: 0 !important;}
.kv {font-size: 26px !important; font-weight: 600; margin: 2px 0 0 !important; line-height: 1.25;}
.kn {font-size: 12px !important; color: #5F6B7A; margin: 4px 0 0 !important;}
.kn.good {color: #1E7B45;}
.ct {font-size: 15px !important; font-weight: 600; margin: 0 !important;}
.cs {font-size: 12px !important; color: #5F6B7A; margin: 2px 0 6px !important;}
.pill {display: inline-block; font-size: 12px !important; padding: 3px 10px; border-radius: 8px; margin-right: 6px;
  background: #E6F1FB; color: #0C447C;}
.pill.n {background: #F1F2F4; color: #3A4250;}
.flow {display: flex; gap: 8px; align-items: stretch; flex-wrap: wrap;}
.step {flex: 1; min-width: 150px; border: 1px solid #E2E5E9; border-radius: 10px; padding: 12px 14px;}
.step.acc {border-color: #185FA5;}
.sn {font-size: 14px !important; font-weight: 600; margin: 0 !important;}
.sd {font-size: 12px !important; color: #5F6B7A; margin: 2px 0 8px !important; min-height: 34px;}
.sm {font-size: 12px !important; margin: 0 !important;}
.arrow {align-self: center; color: #9AA3AE;}
.ins {border-left: 3px solid #185FA5; padding: 2px 0 2px 12px; margin: 0 0 12px !important; font-size: 14px !important;}
.ins b {font-weight: 600;}
.tbl {width: 100%; border-collapse: collapse; font-size: 13px;}
.tbl th {text-align: left; font-weight: 500; color: #5F6B7A; border-bottom: 1px solid #E2E5E9; padding: 6px 8px;}
.tbl td {border-bottom: 1px solid #F0F1F3; padding: 7px 8px; vertical-align: top;}
.tbl td.n, .tbl th.n {text-align: right; white-space: nowrap;}
.tbl td.g {color: #5F6B7A; font-size: 12px;}
.tbl td.a {color: #185FA5; font-weight: 600;}
</style>""", unsafe_allow_html=True)


# ---------------------------------------------------------------- data
@st.cache_resource  # shared, not copied per rerun: the recommendation tables have 1.6M rows
def load(rel):
    p = DATA / rel
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


@st.cache_data
def load_metrics():
    files = glob.glob(str(DATA / "output" / "metrics" / "part-*"))
    return json.loads(open(files[0]).read()) if files else {}


@st.cache_data
def genre_totals():
    gt = load("gold/genre_trends").copy()
    gt["weighted"] = gt.num_ratings * gt.avg_rating
    g = gt.groupby("genre").agg(num_ratings=("num_ratings", "sum"), weighted=("weighted", "sum")).reset_index()
    g["avg_rating"] = g.weighted / g.num_ratings
    g["share"] = g.num_ratings / g.num_ratings.sum()
    return g.drop(columns="weighted").sort_values("num_ratings", ascending=False)


@st.cache_data
def cloud_runs():
    jobs, sizes = pd.DataFrame(), {}
    if (EVIDENCE / "jobs.txt").exists():
        rows = [l.split() for l in (EVIDENCE / "jobs.txt").read_text(encoding="utf-8").splitlines()
                if re.match(r"^[0-9a-f]{32}\s", l)]
        jobs = pd.DataFrame([(r[1].replace(".py", ""), r[-5], f"{r[-4]} {r[-3]}", int(r[-1])) for r in rows],
                            columns=["stage", "state", "start_utc", "seconds"])
    if (EVIDENCE / "bucket_listing.txt").exists():
        sizes = {layer: size for size, layer in re.findall(r"^([\d.]+\s*\w+)\s+gs://\S+/(\w+)$",
                                                           (EVIDENCE / "bucket_listing.txt").read_text(encoding="utf-8"), re.M)}
    return jobs, sizes


def short(n):
    for div, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(n) >= div:
            return f"{n / div:.1f}{suffix}"
    return f"{n:,.0f}"


# ---------------------------------------------------------------- UI helpers
def header(title, subtitle):
    st.markdown(f"<p class='pt'>{title}</p><p class='ps'>{subtitle}</p>", unsafe_allow_html=True)


def kpis(items):
    for col, (label, value, note, tone) in zip(st.columns(len(items)), items):
        col.markdown(f"<div class='kpi'><p class='kl'>{label}</p><p class='kv'>{value}</p>"
                     f"<p class='kn {tone}'>{note}</p></div>", unsafe_allow_html=True)
    st.write("")


def card(title, subtitle, where=st):
    box = where.container(border=True)
    box.markdown(f"<p class='ct'>{title}</p><p class='cs'>{subtitle}</p>", unsafe_allow_html=True)
    return box


def style(fig, height=300, legend=False):
    fig.update_layout(template="simple_white", height=height, margin=dict(l=4, r=4, t=4, b=4), showlegend=legend,
                      font=dict(family="Segoe UI, Inter, sans-serif", size=12, color=TEXT2),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", bargap=0.25,
                      hoverlabel=dict(bgcolor="white", font_size=12),
                      legend=dict(orientation="h", y=1.08, x=0, title=None))
    fig.update_xaxes(showgrid=False, linecolor=GRID, ticks="", title=None)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, showline=False, ticks="", title=None, zeroline=False)
    return fig


def chart(box, fig, height=300, legend=False):
    box.plotly_chart(style(fig, height, legend), width="stretch", config={"displayModeBar": False})


def html_table(box, df, numeric=(), muted=(), accent=()):
    """Compact wrapping table for short lists (Streamlit's grid truncates long titles)."""
    cls = lambda c: " ".join(x for x, cols in (("n", numeric), ("g", muted), ("a", accent)) if c in cols)
    head = "".join(f"<th class='{cls(c)}'>{c}</th>" for c in df.columns)
    rows = "".join("<tr>" + "".join(f"<td class='{cls(c)}'>{v}</td>" for c, v in zip(df.columns, r)) + "</tr>"
                   for r in df.itertuples(index=False))
    box.markdown(f"<table class='tbl'><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>", unsafe_allow_html=True)


def highlight(values, chosen):
    return [ACCENT if v in chosen else NEUTRAL for v in values]


# ---------------------------------------------------------------- pages
def overview():
    m, kp, users, dist = load_metrics(), load("gold/yearly_kpis").sort_values("rating_year"), \
        load("gold/user_activity"), load("gold/rating_distribution")
    g = genre_totals()
    total = kp.num_ratings.sum()
    high = dist.loc[dist.rating >= 4, "count"].sum() / dist["count"].sum()
    gain = 1 - m["test_rmse"] / m["baseline_rmse_global_mean"]
    header("Overview", "How the catalogue is used and how well recommendations work")
    kpis([("Ratings", short(total), f"{kp.rating_year.min()} to {kp.rating_year.max()}", ""),
          ("Users", short(len(users)), f"{(users.segment == 'power').mean():.0%} are power users", ""),
          ("Average rating", f"{(kp.num_ratings * kp.avg_rating).sum() / total:.2f}", f"{high:.0%} of ratings are 4 stars or more", ""),
          ("Prediction error", f"{m['test_rmse']:.2f}", f"{gain:.0%} better than baseline", "good")])

    peak = int(kp.loc[kp.num_ratings.idxmax(), "rating_year"])
    left, right = st.columns([1.6, 1])
    box = card(f"Ratings rebounded from 2015 and peaked in {peak}" if peak >= 2015 else f"Rating activity peaked in {peak}",
               "Ratings per year, 2015 onwards highlighted", left)
    recent = kp.rating_year >= 2015
    fig = go.Figure(go.Bar(x=kp.rating_year, y=kp.num_ratings, marker_color=[ACCENT if r else NEUTRAL for r in recent],
                           hovertemplate="%{x}: %{y:,.0f} ratings<extra></extra>"))
    chart(box, fig, 300)
    top = g.head(8).iloc[::-1]
    box = card(f"{g.iloc[0].genre} leads rating volume", "Share of all ratings by genre (a movie can have several)", right)
    fig = go.Figure(go.Bar(x=top.share, y=top.genre, orientation="h", marker_color=highlight(top.genre, [g.iloc[0].genre]),
                           text=[f"{s:.0%}" for s in top.share], textposition="outside",
                           hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
    fig.update_xaxes(visible=False, range=[0, top.share.max() * 1.25])
    chart(box, fig, 300)

    big = g[g.share >= 0.01]
    best, worst = big.loc[big.avg_rating.idxmax()], big.loc[big.avg_rating.idxmin()]
    st.markdown("<p class='ct' style='margin:8px 0 10px'>Key insights</p>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"<div class='ins'><b>Audiences are generous.</b> {high:.0%} of ratings are 4 stars or more, so "
                "a 3-star rating is a negative signal on this platform.</div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='ins'><b>{best.genre} delights, {worst.genre} divides.</b> {best.genre} averages "
                f"{best.avg_rating:.2f} against {worst.avg_rating:.2f} for {worst.genre}.</div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='ins'><b>Recommendations are reliable.</b> Predictions land within "
                f"{m['test_mae']:.2f} stars on average, {gain:.0%} better than guessing the average rating.</div>",
                unsafe_allow_html=True)


def content():
    stats, tg, gt, tags, sim = load("gold/movie_stats"), load("gold/top_movies_by_genre"), load("gold/genre_trends"), \
        load("gold/top_tags"), load("output/similar_movies")
    g = genre_totals()
    header("Content", "Which genres and titles drive engagement, and what's growing")
    genres = g.genre.tolist()
    genre = st.selectbox("Genre", genres, index=genres.index("Drama") if "Drama" in genres else 0)
    row = g[g.genre == genre].iloc[0]
    yearly = gt[gt.genre == genre].sort_values("rating_year")
    full = yearly[yearly.rating_year <= 2018].tail(4)
    trend = (full.num_ratings.iloc[-1] / full.num_ratings.iloc[0] - 1) if len(full) > 1 else 0
    kpis([("Ratings in genre", short(row.num_ratings), f"{row.share:.0%} of all ratings", ""),
          ("Average rating", f"{row.avg_rating:.2f}", f"rank {int((g.avg_rating > row.avg_rating).sum()) + 1} of {len(g)} genres", ""),
          ("Movies with 50+ ratings", f"{(stats.genres.apply(lambda x: x is not None and genre in x) & (stats.num_ratings >= 50)).sum():,}", "eligible for recommendations", ""),
          ("3-year volume trend", f"{trend:+.0%}", "2015 to 2018", "good" if trend > 0 else "")])

    left, right = st.columns([1.5, 1])
    box = card(f"Best {genre} movies", "Ranked by Bayesian weighted rating, which discounts films with few ratings", left)
    t = tg[tg.genre == genre].sort_values("rank").head(10)
    box.dataframe(t[["rank", "clean_title", "release_year", "num_ratings", "weighted_rating"]], hide_index=True, width="stretch",
                  height=388, column_config={
                      "rank": st.column_config.NumberColumn("#", width="small"),
                      "clean_title": st.column_config.TextColumn("Title"),
                      "release_year": st.column_config.NumberColumn("Year", format="%d"),
                      "num_ratings": st.column_config.NumberColumn("Ratings", format="localized"),
                      "weighted_rating": st.column_config.NumberColumn("Weighted", format="%.2f")})
    big = g[g.share >= 0.005]
    best = big.loc[big.avg_rating.idxmax()].genre
    box = card(f"{best} rates highest, but volume lives elsewhere", "Total ratings (log scale) vs average rating, by genre", right)
    fig = go.Figure(go.Scatter(x=big.num_ratings, y=big.avg_rating, mode="markers+text", text=big.genre,
                               textposition="top center", textfont=dict(size=10),
                               marker=dict(size=10, color=highlight(big.genre, [genre])),
                               hovertemplate="%{text}: %{x:,.0f} ratings, avg %{y:.2f}<extra></extra>"))
    fig.update_xaxes(type="log", tickvals=[1e5, 3e5, 1e6, 3e6, 1e7], ticktext=["100K", "300K", "1M", "3M", "10M"])
    chart(box, fig, 388)

    box = card(f"{genre} activity over time", "Ratings per year and average rating", st)
    fig = go.Figure()
    fig.add_bar(x=yearly.rating_year, y=yearly.num_ratings, marker_color=NEUTRAL, name="Ratings",
                hovertemplate="%{x}: %{y:,.0f}<extra></extra>")
    fig.add_scatter(x=yearly.rating_year, y=yearly.avg_rating, yaxis="y2", mode="lines+markers", name="Average rating",
                    line=dict(color=ACCENT, width=2), hovertemplate="%{x}: %{y:.2f}<extra></extra>")
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False, range=[2.5, 4.5], dtick=0.5, tickformat=".1f"))
    chart(box, fig, 260, legend=True)

    left, right = st.columns(2)
    most = stats.nlargest(10, "num_ratings").iloc[::-1]
    box = card(f"{most.iloc[-1].clean_title} is the most-rated title", "Top 10 titles by number of ratings", left)
    fig = go.Figure(go.Bar(x=most.num_ratings, y=most.clean_title, orientation="h",
                           marker_color=highlight(most.clean_title, [most.iloc[-1].clean_title]),
                           hovertemplate="%{y}: %{x:,.0f}<extra></extra>"))
    chart(box, fig, 340)
    tt = tags.head(10).iloc[::-1]
    box = card(f"\"{tt.iloc[-1].tag}\" is the most used tag", "Most used user tags", right)
    fig = go.Figure(go.Bar(x=tt.uses, y=tt.tag, orientation="h", marker_color=highlight(tt.tag, [tt.iloc[-1].tag]),
                           hovertemplate="%{y}: %{x:,.0f}<extra></extra>"))
    chart(box, fig, 340)

    if not sim.empty:
        box = card("Find similar movies", "Nearest neighbours by cosine similarity of ALS movie factors", st)
        titles = sorted(sim.clean_title.unique())
        pick = box.selectbox("Movie", titles, index=titles.index("Matrix, The") if "Matrix, The" in titles else 0)
        s = sim[sim.clean_title == pick].sort_values("rank")[["rank", "similar_title", "similarity"]]
        box.dataframe(s, hide_index=True, width="stretch", column_config={
            "rank": st.column_config.NumberColumn("#", width="small"),
            "similar_title": st.column_config.TextColumn("Similar movie"),
            "similarity": st.column_config.ProgressColumn("Similarity", min_value=0, max_value=1, format="%.2f")})


def audience():
    users, kp = load("gold/user_activity"), load("gold/yearly_kpis").sort_values("rating_year")
    header("Audience", "Who rates, how much, and how engagement has changed")
    kpis([("Users", short(len(users)), "each rated 20 or more movies", ""),
          ("Median ratings per user", f"{users.num_ratings.median():.0f}", f"mean {users.num_ratings.mean():.0f}, long tail", ""),
          ("Power users", f"{(users.segment == 'power').mean():.1%}", "500 or more ratings each", ""),
          ("Average user rating", f"{users.avg_rating.mean():.2f}", "mean of per-user averages", "")])

    counts = users.segment.value_counts()
    seg = pd.DataFrame([(lab, rng, counts.get(key, 0)) for key, lab, rng in SEGMENTS], columns=["segment", "range", "users"])
    seg["share"] = seg.users / seg.users.sum()
    vol = users.groupby("segment").num_ratings.sum()
    power_vol = vol.get("power", 0) / vol.sum()
    left, right = st.columns(2)
    box = card(f"Power users are {seg.share.iloc[-1]:.0%} of users but give {power_vol:.0%} of ratings",
               "Users by engagement segment", left)
    fig = go.Figure(go.Bar(x=seg.segment + "<br><sub>" + seg.range + "</sub>", y=seg.users,
                           marker_color=[NEUTRAL, NEUTRAL, NEUTRAL, ACCENT], text=[f"{s:.0%}" for s in seg.share],
                           textposition="outside", hovertemplate="%{y:,.0f} users<extra></extra>"))
    fig.update_yaxes(range=[0, seg.users.max() * 1.18])
    chart(box, fig, 300)
    peak_users = kp.loc[kp.active_users.idxmax()]
    box = card(f"Active users peaked in {int(peak_users.rating_year)} at {short(peak_users.active_users)}",
               "Users who rated at least once that year", right)
    fig = go.Figure(go.Scatter(x=kp.rating_year, y=kp.active_users, mode="lines+markers", line=dict(color=ACCENT, width=2),
                               fill="tozeroy", fillcolor="rgba(24,95,165,0.08)", hovertemplate="%{x}: %{y:,.0f}<extra></extra>"))
    chart(box, fig, 300)

    left, right = st.columns(2)
    clip = users[users.num_ratings <= users.num_ratings.quantile(0.98)]
    box = card(f"Half of users rate fewer than {users.num_ratings.median():.0f} movies", "Ratings per user, top 2% excluded", left)
    fig = px.histogram(clip, x="num_ratings", nbins=60, color_discrete_sequence=[NEUTRAL])
    fig.add_vline(x=users.num_ratings.median(), line_color=ACCENT, line_dash="dot",
                  annotation_text="median", annotation_font_color=ACCENT)
    chart(box, fig, 280)
    mid = users.avg_rating.between(3, 4).mean()
    box = card(f"{mid:.0%} of users average between 3 and 4 stars", "Distribution of each user's average rating", right)
    fig = px.histogram(users, x="avg_rating", nbins=50, color_discrete_sequence=[NEUTRAL])
    fig.add_vline(x=users.avg_rating.mean(), line_color=ACCENT, line_dash="dot", annotation_text="mean",
                  annotation_font_color=ACCENT)
    chart(box, fig, 280)


def recommendations():
    recs, fav, users = load("output/user_recs_top10"), load("gold/user_top_rated"), load("gold/user_activity")
    header("Recommendations", "Check a user's taste against what the model suggests")
    ids = set(recs.userId.unique())
    c1, c2 = st.columns([1, 4])
    uid = int(c1.number_input("User ID", min_value=int(min(ids)), max_value=int(max(ids)), value=148, step=1))
    if uid not in ids:
        st.info("This user has no recommendations. Try another user ID.")
        return
    u = users[users.userId == uid].iloc[0]
    f = fav[fav.userId == uid].sort_values("rank")
    r = recs[recs.userId == uid].sort_values("rank").copy()
    fav_genres = pd.Series([x for gs in f.genres.dropna() for x in gs]).value_counts()
    top3 = ", ".join(fav_genres.index[:3])
    overlap = sum(1 for gs in r.genres if gs is not None and set(gs) & set(fav_genres.index[:3]))
    seg = dict((k, lab) for k, lab, _ in SEGMENTS)[u.segment]
    c2.markdown(f"<div style='padding-top:30px'><span class='pill'>{seg} user</span>"
                f"<span class='pill n'>{u.num_ratings:,} ratings</span><span class='pill n'>average {u.avg_rating:.2f}</span>"
                f"<span class='pill n'>favourite genres: {top3}</span></div>", unsafe_allow_html=True)
    st.write("")

    to_text = lambda gs: ", ".join(gs) if gs is not None else ""
    left, right = st.columns(2)
    box = card("Loved", "This user's highest rated movies", left)
    html_table(box, pd.DataFrame({"Title": f.clean_title, "Genres": f.genres.apply(to_text),
                                  "Rating": f.rating.map("{:.1f}".format)}), numeric=["Rating"], muted=["Genres"])
    box = card("Recommended next", f"Unseen movies ranked by predicted rating. {overlap} of 10 share a favourite genre", right)
    html_table(box, pd.DataFrame({"#": r["rank"], "Title": r.clean_title, "Genres": r.genres.apply(to_text),
                                  "Predicted": r.predicted_rating.map("{:.2f}".format)}),
               numeric=["#", "Predicted"], muted=["Genres"], accent=["Predicted"])
    st.caption("Recommendations come from Spark MLlib ALS trained on 25M ratings, limited to movies with 50 or more "
               "ratings, and exclude anything the user has already rated.")


def model_quality():
    m, pred = load_metrics(), load("output/test_predictions_sample").copy()
    gain = 1 - m["test_rmse"] / m["baseline_rmse_global_mean"]
    header("Model quality", "How accurate the ALS recommender is and how it was tuned")
    kpis([("Prediction error (RMSE)", f"{m['test_rmse']:.3f}", f"baseline {m['baseline_rmse_global_mean']:.3f}", "good"),
          ("Average error", f"{m['test_mae']:.2f} stars", "mean absolute error", ""),
          ("Precision at 10", f"{m['precision_at_10']:.1%}", "top-10 hits among liked test movies", ""),
          ("NDCG at 10", f"{m['ndcg_at_10']:.3f}", "ranking quality, 1 is perfect", "")])

    left, right = st.columns(2)
    box = card(f"ALS cuts prediction error by {gain:.0%}", "Test RMSE, lower is better", left)
    fig = go.Figure(go.Bar(x=["Global average (baseline)", f"ALS rank {m['best_rank']}"],
                           y=[m["baseline_rmse_global_mean"], m["test_rmse"]], marker_color=[NEUTRAL, ACCENT],
                           text=[f"{m['baseline_rmse_global_mean']:.3f}", f"{m['test_rmse']:.3f}"], textposition="outside"))
    fig.update_yaxes(range=[0, m["baseline_rmse_global_mean"] * 1.2])
    chart(box, fig, 280)
    grid = pd.DataFrame(m["grid"])
    grid["label"] = "rank " + grid["rank"].astype(str) + ", reg " + grid.regParam.astype(str)
    best = grid.loc[grid.val_rmse.idxmin(), "label"]
    box = card(f"Best setting: {best}", "Validation RMSE for each tuning combination", right)
    fig = go.Figure(go.Bar(x=grid.label, y=grid.val_rmse, marker_color=highlight(grid.label, [best]),
                           text=grid.val_rmse.round(3), textposition="outside"))
    fig.update_yaxes(range=[grid.val_rmse.min() - 0.03, grid.val_rmse.max() + 0.01])
    chart(box, fig, 280)

    if not pred.empty:
        pred["error"] = pred.prediction - pred.rating
        within = (pred.error.abs() <= 1).mean()
        left, right = st.columns(2)
        box = card("Predictions follow actual ratings, pulled towards the middle", "Predicted rating by actual rating (test sample)", left)
        fig = px.box(pred, x="rating", y="prediction", color_discrete_sequence=[ACCENT], points=False)
        chart(box, fig, 280)
        box = card(f"{within:.0%} of predictions are within 1 star", "Prediction error, predicted minus actual", right)
        fig = px.histogram(pred, x="error", nbins=60, color_discrete_sequence=[NEUTRAL])
        fig.add_vrect(x0=-1, x1=1, fillcolor=ACCENT, opacity=0.08, line_width=0)
        chart(box, fig, 280)

    with st.expander("How to read these metrics"):
        st.markdown(f"""
- **RMSE** is the typical size of a prediction error in stars, with big misses weighted more. Always predicting the
  average rating scores {m['baseline_rmse_global_mean']:.2f}, while ALS scores {m['test_rmse']:.2f}.
- **Precision at 10** is the share of a user's 10 recommendations that they rated 4+ in the held-out test data. It looks
  low because only about 20% of each user's ratings are hidden and a top-10 list is picked from {m.get('movies_in_model', 0):,}
  movies, so it's mainly useful for comparing model versions.
- **Training data:** {m['train_rows']:,} ratings for training, {m['val_rows']:,} for tuning and {m['test_rows']:,} for the final test.
  Only movies with {m.get('min_movie_ratings', 50)} or more ratings are modelled, to avoid unreliable recommendations.
""")


def pipeline():
    dq = load("silver/dq_report")
    jobs, sizes = cloud_runs()
    header("Data pipeline", "Medallion layers on Google Cloud, data quality checks and job runtimes")
    done = jobs[jobs.state == "DONE"].drop_duplicates("stage", keep="last") if not jobs.empty else jobs
    secs = dict(zip(done.stage, done.seconds)) if not done.empty else {}
    fmt_t = lambda s: f"{s // 60:.0f} min {s % 60:.0f} s" if s >= 60 else f"{s:.0f} s"
    steps = [("Raw", "GroupLens CSVs landed in Cloud Storage", sizes.get("raw", ""), secs.get("ingest")),
             ("Bronze", "Parquet with schemas and ingestion metadata", sizes.get("bronze", ""), secs.get("bronze")),
             ("Silver", "Cleaned, de-duplicated, partitioned by year", sizes.get("silver", ""), secs.get("silver")),
             ("Gold", "Analysis-ready tables for this dashboard", sizes.get("gold", ""), secs.get("gold")),
             ("Model", "ALS model, Top-10 lists, similar movies", sizes.get("output", ""), secs.get("train_als"))]
    html = []
    for i, (name, desc, size, t) in enumerate(steps):
        html.append(f"<div class='step{' acc' if name == 'Model' else ''}'><p class='sn'>{name}</p><p class='sd'>{desc}</p>"
                    f"<p class='sm'>{size or '–'}</p><p class='sm' style='color:#5F6B7A'>{fmt_t(t) if t else '–'}</p></div>")
        if i < len(steps) - 1:
            html.append("<div class='arrow'>&rarr;</div>")
    st.markdown("<div class='flow'>" + "".join(html) + "</div>", unsafe_allow_html=True)
    st.write("")

    removed = int(dq.removed.sum()) if not dq.empty else 0
    total = sum(secs.values())
    kpis([("Quality checks", f"{len(dq)}", "run on every load", ""),
          ("Rows removed", f"{removed:,}", "nulls, invalid values, duplicates, orphans", ""),
          ("End-to-end runtime", fmt_t(total) if total else "–", "on a 3-node Dataproc cluster", ""),
          ("Cluster", "Dataproc 2.2", "1 x e2-standard-2, 2 x e2-highmem-4", "")])

    left, right = st.columns([1.3, 1])
    box = card("Every cleaning rule is logged", "Silver layer data-quality report", left)
    if not dq.empty:
        box.dataframe(dq.rename(columns={"table": "Table", "check": "Check", "rows_before": "Before",
                                         "rows_after": "After", "removed": "Removed"}),
                      hide_index=True, width="stretch", height=330,
                      column_config={c: st.column_config.NumberColumn(format="localized") for c in ("Before", "After", "Removed")})
    if secs:
        order = [s for s in ("ingest", "bronze", "silver", "gold", "train_als") if s in secs]
        box = card("Model training dominates runtime", "Seconds per Dataproc job (latest successful run)", right)
        fig = go.Figure(go.Bar(x=[secs[s] for s in order][::-1], y=order[::-1], orientation="h",
                               marker_color=highlight(order[::-1], ["train_als"]), text=[fmt_t(secs[s]) for s in order][::-1],
                               textposition="outside"))
        fig.update_xaxes(visible=False, range=[0, max(secs.values()) * 1.3])
        chart(box, fig, 330)


# ---------------------------------------------------------------- app
if load("gold/movie_stats").empty or not load_metrics():
    st.error(f"No pipeline outputs found in {DATA}. Run `python run_pipeline.py` or scripts/dataproc_run.ps1 first.")
    st.stop()

nav = st.navigation([
    st.Page(overview, title="Overview", icon=":material/dashboard:", default=True),
    st.Page(content, title="Content", icon=":material/movie:", url_path="content"),
    st.Page(audience, title="Audience", icon=":material/group:", url_path="audience"),
    st.Page(recommendations, title="Recommendations", icon=":material/auto_awesome:", url_path="recommendations"),
    st.Page(model_quality, title="Model quality", icon=":material/monitoring:", url_path="model-quality"),
    st.Page(pipeline, title="Data pipeline", icon=":material/database:", url_path="data-pipeline"),
])
_m = load_metrics()
_jobs, _ = cloud_runs()
_last = _jobs.start_utc.max()[:10] if not _jobs.empty else "local run"
st.sidebar.caption(f"Last run {_last}  \nModel ALS, rank {_m['best_rank']}, reg {_m['best_regParam']}  \n"
                   f"Dataset MovieLens {_m['dataset']} (GroupLens)")
nav.run()
