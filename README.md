# Marketing Intelligence Dashboard (Streamlit)

A single‑page, sleek Streamlit dashboard that connects marketing activity to business results. It loads four local CSVs, applies robust validation and normalization, and provides interactive filters, KPIs with WoW deltas, trends, and rich visuals.

## Features
- Global filters: date range, channels, tactics, states, campaigns.
- KPI header cards with WoW deltas and sparklines: Total Revenue, Gross Profit, Orders, New Customers, Total Spend, Blended ROAS, Blended CAC, AOV, Gross Margin.
- Tabs:
  - Trends: Revenue vs Spend (dual axes) with anomaly markers; Blended ROAS + 7‑day MA; optional series.
  - Channels & Tactics: Channel contribution toggle; tactic stacks; per‑channel small multiples.
  - Campaigns: Scatter (Spend vs Attributed Revenue, size=Clicks, color=Channel); leaderboard with ROAS/CPC/CTR/CPM; CSV download.
  - Geography: US choropleth by state (fallback to ranked bars); regional table with share and ROAS.
  - Insights: Auto callouts for top/bottom ROAS, WoW spend movers, anomaly summary.
- Performance: cached loads/aggregations via `st.cache_data`.
- Accessibility: clean palette, subtle gridlines, readable typography.

## Data Requirements
Place the following files under `data/` with the exact headers shown.

1) Business CSV — `data/business.csv`
Required headers (exact):
```
date
# of orders
# of new orders
new customers
total revenue
gross profit
COGS
```
Normalized immediately after load to:
```
{ "date": "date",
  "# of orders": "orders",
  "# of new orders": "new_orders",
  "new customers": "new_customers",
  "total revenue": "total_revenue",
  "gross profit": "gross_profit",
  "COGS": "cogs" }
```

2) Channel CSVs — `data/Facebook.csv`, `data/Google.csv`, `data/TikTok.csv`
Required headers (exact; note singular "impression" and space in "attributed revenue"):
```
date
tactic
state
campaign
impression
clicks
spend
attributed revenue
```
Normalized immediately after load and a `channel` column is added from the filename:
```
{ "date": "date",
  "tactic": "tactic",
  "state": "state",
  "campaign": "campaign",
  "impression": "impressions",   # normalize singular → plural
  "clicks": "clicks",
  "spend": "spend",
  "attributed revenue": "attributed_revenue" }
```

Parsing/cleaning rules:
- Dates are parsed to date (no time).
- Numerics are coerced with `pd.to_numeric(errors="coerce")` and pre‑cleaned for commas and currency symbols.
- Missing/zero denominators handled safely (results as NaN where appropriate).

## KPIs & Derived Metrics
- ctr = clicks / impressions
- cpc = spend / clicks
- cpm = 1000 * spend / impressions
- roas = attributed_revenue / spend
- aov = total_revenue / orders
- gross_margin = gross_profit / total_revenue
- blended_roas = sum(attributed_revenue) / sum(spend)
- blended_cac = sum(spend) / sum(new_customers)
- WoW deltas: last 7 days vs previous 7 days, anchored to the selected end date.

## Install & Run
Prereqs: Python 3.9+ recommended.

```bash
# optional: create and activate a virtualenv
python3 -m venv .venv
source .venv/bin/activate

# install dependencies
pip install -r requirements.txt

# run the app
streamlit run app.py
```

## Project Structure
- `app.py:1` — Streamlit app with data loading, filters, KPIs, charts, tabs, insights.
- `requirements.txt:1` — Python dependencies (`streamlit`, `pandas`, `numpy`, `plotly`, `kaleido`).
- `data/` — Place `business.csv`, `Facebook.csv`, `Google.csv`, `TikTok.csv` here.

## Exports & Interactions
- Global filters apply across all visuals.
- Campaign leaderboard CSV download reflects current filters.
- Main trend chart PNG download uses `kaleido` (already included in requirements).
- The header shows last data load time and each CSV’s last modified timestamp.

## Troubleshooting
- Missing column error: ensure headers exactly match the lists above (including case, spaces, punctuation like `#`).
- Numeric parsing: if values include symbols or commas, they will be cleaned automatically; non‑numeric strings become NaN.
- Choropleth: state must be full name (e.g., "California") or two‑letter code (e.g., "CA"). Unknown/missing states fall back to ranked bars.
- PNG export: if PNG download fails, confirm `kaleido` installed (`pip install kaleido`).
- Performance: large CSVs benefit from running in a virtualenv and using the cache; data reload occurs when files change.

## Notes
- Dashboard runs locally and never exposes external reference links.
- Styling aims for a clean, modern, and legible UI with consistent channel colorways.
