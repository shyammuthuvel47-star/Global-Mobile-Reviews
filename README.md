# Global-Mobile-Reviews
# 📱 Mobile Product Segmentation & Recommendation System

An end-to-end data analytics and machine learning pipeline that segments smartphone
models into **Budget / Mid-range / Premium** tiers using K-Means clustering, and
recommends similar phones using a **cosine-similarity** based recommendation engine —
all wrapped in an interactive **Streamlit** app.

Built as part of the *Mobile Product Segmentation and Recommendation System Using
Python and Machine Learning* capstone project (GUVI x HCL).

---

## 🎯 Problem Statement

Smartphone review data is unorganized and hard to analyze directly. This project
transforms 50,000 raw customer reviews into a structured, product-level dataset,
segments the underlying phone models into meaningful price tiers, and builds a
similarity-based recommender so a user can explore "phones like this one."

---

## 🧰 Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Data handling | Pandas, NumPy |
| Visualization (EDA) | Matplotlib, Seaborn |
| Visualization (App) | Plotly |
| Machine Learning | Scikit-learn (K-Means, StandardScaler, Cosine Similarity) |
| App / Deployment | Streamlit |

---

## 🗂️ Project Structure

```
├── Mobile Reviews Sentiment null.csv   # raw dataset (50,000 reviews)
├── Project-4.ipynb                     # full analysis notebook (cleaning → EDA → ML)
├── cleaned_mobile_reviews.csv          # cleaned review-level data (notebook output)
├── product_segments.csv                # product-level table with cluster/segment labels
├── similarity_matrix.csv               # phone-to-phone cosine similarity matrix
├── app.py                              # Streamlit application
└── README.md
```

---

## 🔁 Pipeline Overview

The notebook (`Project-4.ipynb`) follows this sequence:

1. **Data Loading & Inspection** — load the raw 50,000-row review dataset (50,000 rows × 22 columns) and inspect structure, dtypes, and missing values.
2. **Data Cleaning**
   - Dropped the corrupted `price_local` column (encoding mismatch on non-USD currency symbols).
   - Imputed missing `price_usd` / `rating` using the median for the same brand + model, falling back to the global median.
   - Derived missing `sentiment` values from `rating` (≥4 → Positive, 3 → Neutral, <3 → Negative).
   - Filled missing `source` with the mode.
   - Removed duplicate rows.
   - Fixed a mixed date format bug in `review_date` (the column contained both `DD-MM-YYYY` and `M/D/YYYY` formats), parsing each format separately and merging the result.
3. **Exploratory Data Analysis (EDA)** — brand distribution, rating distribution, sentiment distribution vs. rating (crosstab), price distribution, and country/source breakdown.
4. **Product-Level Aggregation** — reviews are aggregated from 50,000 individual rows into **one row per (brand, model)**, since segmentation is a *product* concept, not a per-review one. Aggregated features include median price, mean ratings, % positive sentiment, % verified purchases, and review count.
5. **Feature Variance Validation** — a review-level vs. product-level variance check revealed that sub-ratings (camera, battery, etc.) are essentially random noise at the review level and wash out to near-identical averages across every phone once aggregated. Only **price** and **review count** showed meaningful product-level variance — an important, documented finding that shaped the clustering feature choice below.
6. **Price Segmentation (K-Means)**
   - Feature used: `price_usd` (scaled with `StandardScaler`).
   - Optimal `k` explored via the Elbow Method (tested k = 1–6).
   - Final model fit with **k = 3**, chosen to match the standard Premium / Mid-range / Budget business framing, even though the elbow mathematically leaned toward k = 2 — a deliberate, documented override.
   - Clusters mapped to readable labels: **Budget**, **Mid-range**, **Premium**.
7. **Recommendation System**
   - Feature set: scaled numeric features (price, all 5 sub-ratings, % positive sentiment, % verified) + one-hot encoded `brand`.
   - Pairwise **cosine similarity** computed across all 22 phone models.
   - `recommend_similar_phones(model_name, top_n)` returns the top-N most similar phones to a given model.
8. **Outputs saved** — `product_segments.csv` and `similarity_matrix.csv`, consumed directly by the Streamlit app.

---

## 📊 Segmentation Results

| Segment | Price Range (USD) | Brands |
|---|---|---|
| **Budget** | ~$392 – $515 | Realme, Xiaomi, Motorola |
| **Mid-range** | ~$667 – $909 | OnePlus, Google, Samsung |
| **Premium** | ~$1,094 – $1,106 | Apple |

Clear price gaps exist between all three segments (no overlapping ranges), and the
clusters align closely with brand identity — a strong, interpretable result for a
1-feature K-Means run.

---

## 🖥️ Streamlit App

The app (`app.py`) has three sections:

1. **Product Segments by Price** — brand-filterable strip plot of all products colored by segment, plus a segment summary table.
2. **Insights & Charts** — a segment → brand → model treemap, a price-vs-rating bubble scatter, segment mix pie/bar charts, and a per-segment deep dive (tabs for Budget / Mid-range / Premium) with horizontal bar, line, and gauge charts.
3. **Find Similar Phones** — pick a brand to choose a source phone, filter the recommendation candidate pool by price/rating range (across **all** brands), and get the top-N most similar phones via cosine similarity.

### Run it locally

```bash
pip install streamlit pandas plotly
streamlit run app.py
```

Make sure `product_segments.csv` and `similarity_matrix.csv` are in the same folder
as `app.py` (generated by running the notebook end-to-end).

---

## 🔑 Key Insights & Design Decisions

- **Review-level vs. product-level clustering**: clustering was deliberately done on an aggregated *product*-level table (one row per brand+model), not on raw reviews — a cluster should represent a phone's tier, not one person's opinion.
- **Sub-ratings excluded from clustering**: despite high review-level variance (std ≈ 1.2–1.35), sub-ratings collapse to near-identical averages per product (std ≈ 0.02–0.035) once aggregated, indicating they don't reflect real per-model quality differences in this dataset. Price was the only reliable signal for segmentation.
- **k = 3 over the mathematically-favored k = 2**: chosen intentionally to match the standard Premium/Mid-range/Budget business framing, with the trade-off explicitly documented rather than silently overridden.
- **Recommendation system uses a richer feature set than clustering**: sub-ratings, sentiment %, verified-purchase %, and one-hot encoded brand are included here, since a recommender tolerates noisier features better than a hard-boundary clustering task does.
- **Brand does not dominate recommendations**: despite including brand as a feature, top matches are often cross-brand (e.g., iPhone 13's closest match is a Samsung Galaxy Note 20), showing the numeric features carry real weight in the similarity score.

---

## 📌 Future Improvements

- Incorporate additional real-world product specs (RAM, storage, screen size) if available, to enrich the clustering feature set beyond price alone.
- Experiment with weighted cosine similarity to let users tune how much brand loyalty affects recommendations.
- Add a persistence/database layer instead of static CSV files for a production deployment.
