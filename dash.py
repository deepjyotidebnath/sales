"""
Sales Analytics Dashboard (Python / Streamlit)

Install:  pip install streamlit pandas numpy matplotlib
Run:      streamlit run sales_dashboard.py

Upload a CSV, map your columns, pick options in the sidebar, and get charts,
customer metrics, a forecast and a downloadable PDF report.
"""
import io
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.backends.backend_pdf import PdfPages

FIELDS = {  # field -> (required, column-name pattern)
    "date": (True, r"date|time|day"),
    "revenue": (True, r"revenue|amount|sales|total|price"),
    "product": (False, r"product|item|sku|name"),
    "category": (False, r"categ|dept|type"),
    "customer": (False, r"customer|client|user|buyer"),
    "quantity": (False, r"qty|quant|units"),
    "order": (False, r"order[ _-]?(id|no|num)|invoice|txn|transaction"),
}
FREQ = {"Daily": "D", "Weekly": "W-MON", "Monthly": "MS"}
TEAL, AMBER, RED, BLUE = "#0f8b8d", "#e9a23b", "#d1495b", "#3d5a80"


# ----------------------------------------------------------------- data ---
def sample_data(n_days=300, seed=7):
    rng = np.random.default_rng(seed)
    prods = [("Trail Backpack", "Outdoor", 89), ("Insulated Bottle", "Outdoor", 28),
             ("Wireless Earbuds", "Electronics", 120), ("Smart Watch", "Electronics", 210),
             ("Desk Lamp", "Home", 45), ("Ceramic Mug Set", "Home", 34),
             ("Yoga Mat", "Fitness", 40), ("Running Shoes", "Fitness", 95),
             ("Notebook Pack", "Office", 15), ("Standing Desk", "Office", 340)]
    start = pd.Timestamp.today().normalize() - pd.Timedelta(days=n_days)
    rows = []
    for i in range(n_days):
        d = start + pd.Timedelta(days=i)
        n = max(2, int(6 + i / 30 + 3 * np.sin(i / 7 * 2 * np.pi) + rng.random() * 5))
        for j in range(n):
            p = prods[int(rng.random() * rng.random() * len(prods))]
            q = int(rng.integers(1, 4))
            rows.append({"order_date": d.date(), "order_id": f"O{i}-{j}",
                         "customer_id": f"C{int(rng.integers(0, 80 + i * 3))}",
                         "product": p[0], "category": p[1], "quantity": q,
                         "revenue": round(p[2] * q * (0.9 + rng.random() * 0.2), 2)})
    return pd.DataFrame(rows)


def guess_column(cols, pattern):
    return next((c for c in cols if re.search(pattern, str(c), re.I)), None)


def prepare(df, mapping):
    """Rename mapped columns to standard names and clean types."""
    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df[mapping["date"]], errors="coerce")
    out["revenue"] = pd.to_numeric(
        df[mapping["revenue"]].astype(str).str.replace(r"[^0-9.\-]", "", regex=True), errors="coerce")
    for f in ("product", "category", "customer", "order"):
        out[f] = df[mapping[f]] if mapping.get(f) else np.nan
    out["quantity"] = (pd.to_numeric(df[mapping["quantity"]], errors="coerce").fillna(1)
                       if mapping.get("quantity") else 1)
    return out.dropna(subset=["date", "revenue"]).sort_values("date").reset_index(drop=True)


def bucket(dates, gran):
    if gran == "Monthly":
        return dates.dt.to_period("M").dt.start_time
    if gran == "Weekly":
        return dates.dt.to_period("W-SUN").dt.start_time
    return dates.dt.normalize()


# ------------------------------------------------------------ analytics ---
def series_by_period(df, gran, metric):
    b = bucket(df["date"], gran)
    g = df.groupby(b)
    if metric == "Revenue":
        s = g["revenue"].sum()
    elif metric == "Units sold":
        s = g["quantity"].sum()
    else:
        s = g["order"].nunique() if df["order"].notna().any() else g.size()
    idx = pd.date_range(s.index.min(), s.index.max(), freq=FREQ[gran])
    return s.reindex(idx, fill_value=0)


def customer_split(all_df, view_df, gran):
    """New vs returning customers per period (new = first purchase in the whole dataset)."""
    if not view_df["customer"].notna().any():
        return None
    first = all_df.groupby("customer")["date"].min()
    first_b = bucket(first.to_frame("date")["date"], gran)
    v = view_df.dropna(subset=["customer"]).copy()
    v["b"] = bucket(v["date"], gran)
    v["is_new"] = v["customer"].map(first_b) == v["b"]
    pairs = v.drop_duplicates(["b", "customer"])
    res = pairs.groupby(["b", "is_new"]).size().unstack(fill_value=0)
    res = res.rename(columns={True: "New", False: "Returning"})
    for c in ("New", "Returning"):
        if c not in res:
            res[c] = 0
    return res[["New", "Returning"]]


def forecast(y, horizon, method):
    y = np.asarray(y, float)
    n = len(y)
    x = np.arange(n)
    b, a = np.polyfit(x, y, 1) if n > 1 else (0.0, y[0])
    sd = float(np.std(y - (a + b * x), ddof=min(2, n - 1))) if n > 2 else 0.0
    if method == "Linear trend":
        out = a + b * (n + np.arange(horizon))
    elif method == "Moving average":
        out = np.full(horizon, y[-min(4, n):].mean())
    else:  # Holt's smoothed trend
        level, trend = y[0], (y[1] - y[0] if n > 1 else 0.0)
        for v in y[1:]:
            prev = level
            level = 0.5 * v + 0.5 * (level + trend)
            trend = 0.3 * (level - prev) + 0.7 * trend
        out = level + trend * np.arange(1, horizon + 1)
    out = np.maximum(out, 0)
    band = 1.28 * sd * np.sqrt(1 + np.arange(1, horizon + 1) / max(n, 1))  # ~80% range
    return out, band


def human(n):
    n = float(n)
    for lim, suf in ((1e6, "M"), (1e3, "K")):
        if abs(n) >= lim:
            return f"{n / lim:.2f}{suf}"
    return f"{n:,.0f}"


def compute(df_all, df, opts):
    gran, metric = opts["gran"], opts["metric"]
    R = {"gran": gran, "metric": metric}
    R["y"] = series_by_period(df, gran, metric)
    orders = df["order"].nunique() if df["order"].notna().any() else len(df)
    R["kpis"] = {"Revenue": human(df["revenue"].sum()), "Orders": human(orders),
                 "Avg order value": human(df["revenue"].sum() / max(orders, 1)),
                 "Units sold": human(df["quantity"].sum())}
    if df["customer"].notna().any():
        per = df.dropna(subset=["customer"]).groupby("customer")
        counts = per["order"].nunique() if df["order"].notna().any() else per.size()
        R["kpis"]["Customers"] = human(len(counts))
        R["kpis"]["Repeat rate"] = f"{(counts > 1).mean() * 100:.0f}%"
    R["products"] = (df.dropna(subset=["product"]).groupby("product")["revenue"].sum()
                     .nlargest(opts["top_n"]) if df["product"].notna().any() else None)
    R["categories"] = (df.dropna(subset=["category"]).groupby("category")["revenue"].sum()
                       .nlargest(8) if df["category"].notna().any() else None)
    R["customers"] = customer_split(df_all, df, gran)
    fc, band = forecast(R["y"].values, opts["horizon"], opts["method"])
    future = pd.date_range(R["y"].index[-1], periods=opts["horizon"] + 1, freq=FREQ[gran])[1:]
    R["fc"] = pd.DataFrame({"forecast": fc, "low": np.maximum(fc - band, 0), "high": fc + band}, index=future)
    total = df["revenue"].sum()
    tips = []
    if R["products"] is not None and len(R["products"]):
        tips.append(f"{R['products'].index[0]} is the top product ({R['products'].iloc[0] / total * 100:.1f}% of revenue).")
    if R["categories"] is not None and len(R["categories"]):
        tips.append(f"{R['categories'].index[0]} is the strongest category ({R['categories'].iloc[0] / total * 100:.0f}% of revenue).")
    tips.append(f"Best {gran.lower()[:-2] if gran != 'Daily' else 'day'}: {R['y'].idxmax():%Y-%m-%d} with {human(R['y'].max())} {metric.lower()}.")
    tips.append(f"Forecast for next {opts['horizon']} periods: about {human(fc.sum())} {metric.lower()} ({opts['method']}).")
    R["tips"] = tips
    return R


# ---------------------------------------------------------------- charts ---
def make_figures(R, sections):
    figs = {}
    m = R["metric"]
    if "Sales trend" in sections:
        f, ax = plt.subplots(figsize=(8, 3.6))
        ax.bar(R["y"].index, R["y"].values, width=0.8 if R["gran"] == "Daily" else 20 if R["gran"] == "Monthly" else 5, color=TEAL)
        ax.set_title(f"{R['gran']} {m.lower()}")
        f.autofmt_xdate(); f.tight_layout(); figs["Sales trend"] = f
    if "Products" in sections and R["products"] is not None:
        f, ax = plt.subplots(figsize=(8, 3.6))
        p = R["products"].iloc[::-1]
        ax.barh(p.index.astype(str), p.values, color=BLUE)
        ax.set_title("Top products by revenue"); f.tight_layout(); figs["Products"] = f
    if "Categories" in sections and R["categories"] is not None:
        f, ax = plt.subplots(figsize=(8, 3.6))
        ax.pie(R["categories"].values, labels=R["categories"].index, autopct="%1.0f%%",
               colors=[TEAL, AMBER, RED, BLUE, "#7b9e89", "#b08bbb", "#c9a66b", "#8896a8"])
        ax.set_title("Category share of revenue"); f.tight_layout(); figs["Categories"] = f
    if "Customers" in sections and R["customers"] is not None:
        f, ax = plt.subplots(figsize=(8, 3.6))
        c = R["customers"]
        ax.bar(c.index, c["Returning"], color=TEAL, label="Returning", width=0.8 if R["gran"] == "Daily" else 20 if R["gran"] == "Monthly" else 5)
        ax.bar(c.index, c["New"], bottom=c["Returning"], color=AMBER, label="New", width=0.8 if R["gran"] == "Daily" else 20 if R["gran"] == "Monthly" else 5)
        ax.set_title("New vs returning customers"); ax.legend(); f.autofmt_xdate(); f.tight_layout(); figs["Customers"] = f
    if "Forecast" in sections:
        f, ax = plt.subplots(figsize=(8, 3.6))
        y, fc = R["y"], R["fc"]
        ax.plot(y.index, y.values, color=TEAL, label="Actual")
        link = pd.concat([y.iloc[[-1]], fc["forecast"]])
        ax.plot(link.index, link.values, color=AMBER, lw=2.5, label="Forecast")
        ax.fill_between(fc.index, fc["low"], fc["high"], color=RED, alpha=0.15, label="80% range")
        ax.set_title(f"{m} forecast"); ax.legend(); f.autofmt_xdate(); f.tight_layout(); figs["Forecast"] = f
    return figs


def build_pdf(R, figs, title, period):
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        f = plt.figure(figsize=(8.27, 11.69)); f.text(0.08, 0.94, title, fontsize=20, weight="bold")
        f.text(0.08, 0.91, f"{period}  |  {R['gran']}  |  {R['metric']}", fontsize=10, color="#555")
        y = 0.85
        for k, v in R["kpis"].items():
            f.text(0.08, y, f"{k}: {v}", fontsize=12); y -= 0.03
        y -= 0.02; f.text(0.08, y, "Key takeaways", fontsize=14, weight="bold"); y -= 0.035
        for t in R["tips"]:
            f.text(0.08, y, "- " + t[:95], fontsize=10); y -= 0.028
        if "Forecast" in figs:
            y -= 0.02; f.text(0.08, y, "Forecast values", fontsize=14, weight="bold"); y -= 0.03
            for d, r in R["fc"].iterrows():
                f.text(0.08, y, f"{d:%Y-%m-%d}: {human(r.forecast)}  (range {human(r.low)} to {human(r.high)})", fontsize=10)
                y -= 0.024
                if y < 0.05: break
        pdf.savefig(f); plt.close(f)
        for fig in figs.values():
            pdf.savefig(fig)
    return buf.getvalue()


# -------------------------------------------------------------------- UI ---
def main():
    st.set_page_config(page_title="Sales Analytics Dashboard", layout="wide")
    st.title("Sales Analytics Dashboard")
    st.caption("Load your sales data, choose options in the sidebar, and get charts, a forecast and a PDF report.")

    with st.sidebar:
        st.header("1. Load data")
        up = st.file_uploader("CSV file", type="csv")
        use_sample = st.checkbox("Use sample data", value=up is None)
    if up is not None:
        raw = pd.read_csv(up)
    elif use_sample:
        raw = sample_data()
    else:
        st.info("Upload a CSV or tick 'Use sample data' in the sidebar. You need at least a date and a revenue column.")
        return

    cols = list(raw.columns)
    with st.sidebar:
        st.header("2. Map columns")
        mapping = {}
        for f, (req, pat) in FIELDS.items():
            g = guess_column(cols, pat)
            opts = cols if req else ["(none)"] + cols
            default = opts.index(g) if g in opts else 0
            v = st.selectbox(f.capitalize(), opts, index=default)
            mapping[f] = None if v == "(none)" else v

    try:
        df_all = prepare(raw, mapping)
    except Exception as e:
        st.error(f"Could not read the mapped columns: {e}")
        return
    if df_all.empty:
        st.error("No valid rows. Check that the date and revenue columns are mapped correctly.")
        return

    with st.sidebar:
        st.header("3. Options")
        gran = st.selectbox("View by", list(FREQ), index=2)
        metric = st.selectbox("Metric", ["Revenue", "Orders", "Units sold"])
        lo, hi = df_all["date"].min().date(), df_all["date"].max().date()
        rng = st.date_input("Date range", (lo, hi), min_value=lo, max_value=hi)
        top_n = st.slider("Top N products", 3, 20, 8)
        horizon = st.slider("Forecast periods", 1, 24, 6)
        method = st.selectbox("Forecast method", ["Smoothed trend (Holt)", "Linear trend", "Moving average"])
        sections = st.multiselect("Show and export", ["Sales trend", "Products", "Categories", "Customers", "Forecast"],
                                  default=["Sales trend", "Products", "Categories", "Customers", "Forecast"])

    if len(rng) != 2:
        st.warning("Pick both a start and an end date.")
        return
    df = df_all[(df_all["date"].dt.date >= rng[0]) & (df_all["date"].dt.date <= rng[1])]
    if df.empty:
        st.warning("No rows in that date range.")
        return

    R = compute(df_all, df, dict(gran=gran, metric=metric, top_n=top_n, horizon=horizon, method=method))
    cols_k = st.columns(len(R["kpis"]))
    for c, (k, v) in zip(cols_k, R["kpis"].items()):
        c.metric(k, v)

    figs = make_figures(R, sections)
    items = list(figs.items())
    for i in range(0, len(items), 2):
        row = st.columns(2)
        for c, (name, fig) in zip(row, items[i:i + 2]):
            c.pyplot(fig)

    st.subheader("Key takeaways")
    for t in R["tips"]:
        st.write("- " + t)
    if "Forecast" in sections:
        st.dataframe(R["fc"].round(1))

    d1, d2 = st.columns(2)
    d1.download_button("Download PDF report", build_pdf(R, figs, "Sales Analytics Report", f"{rng[0]} to {rng[1]}"),
                       "sales-report.pdf", "application/pdf")
    d2.download_button("Download forecast CSV", R["fc"].to_csv().encode(), "forecast.csv", "text/csv")


if __name__ == "__main__":
    main()