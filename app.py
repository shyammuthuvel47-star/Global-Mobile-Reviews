"""
Mobile Product Segmentation & Recommendation System
====================================================================
Single-page Streamlit app for exploring price-segmented phone products
and finding similar phones using a precomputed cosine-similarity matrix.

Expects two files in the same folder as this script:
  - product_segments.csv   (output of the segmentation step)
  - similarity_matrix.csv  (output of the recommendation step)

Run with:
    streamlit run app.py
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(page_title="Mobile Product Segmentation & Recommender", layout="wide")
st.title("📱 Mobile Product Segmentation & Recommendation System")


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading data...")
def load_data():
    """Load the pre-generated product segments and similarity matrix."""
    product_df = pd.read_csv("product_segments.csv")
    similarity_df = pd.read_csv("similarity_matrix.csv", index_col=0)
    return product_df, similarity_df


product_df, similarity_df = load_data()


# --------------------------------------------------------------------------
# SECTION 1: Segment Overview
# --------------------------------------------------------------------------
def render_segment_overview(product_df: pd.DataFrame) -> None:
    st.header("Product Segments by Price")

    brands = st.multiselect(
        "Filter by brand",
        options=product_df["brand"].unique(),
        default=product_df["brand"].unique(),
        key="overview_brand_filter",
    )
    filtered_df = product_df[product_df["brand"].isin(brands)]

    fig = px.strip(
        filtered_df,
        x="price_usd",
        y="segment",
        color="segment",
        hover_data=["brand", "model"],
        title="Products by Segment and Price",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Segment Summary")
    summary = (
        product_df.groupby("segment")["price_usd"]
        .agg(["mean", "min", "max", "count"])
        .round(2)
    )
    st.dataframe(summary)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Share of Products by Segment")
        segment_counts = filtered_df["segment"].value_counts().reset_index()
        segment_counts.columns = ["segment", "count"]
        fig_pie = px.pie(
            segment_counts,
            names="segment",
            values="count",
            title="Segment Mix",
            hole=0.4,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.subheader("Average Price by Brand")
        avg_price_by_brand = (
            filtered_df.groupby("brand")["price_usd"].mean().round(2).sort_values(ascending=False).reset_index()
        )
        fig_bar = px.bar(
            avg_price_by_brand,
            x="brand",
            y="price_usd",
            title="Average Price by Brand",
            color="brand",
        )
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("All Products")
    st.dataframe(filtered_df[["brand", "model", "price_usd", "segment", "review_count"]])


# --------------------------------------------------------------------------
# SECTION 2: Insights & Charts (organized by segment: Budget / Mid-range / Premium)
# --------------------------------------------------------------------------
def render_insights(product_df: pd.DataFrame) -> None:
    st.header("Insights & Charts")

    feature_cols = [
        "battery_life_rating", "camera_rating", "performance_rating",
        "design_rating", "display_rating",
    ]

    segment_order = ["Budget", "Mid-range", "Premium"]
    segment_colors = {"Budget": "#2ecc71", "Mid-range": "#f39c12", "Premium": "#3498db"}

    # ------------------------------------------------------------------
    # ROW 1: Cross-segment big picture — Treemap + Scatter
    # ------------------------------------------------------------------
    st.subheader("Cross-Segment Overview")
    col1, col2 = st.columns(2)

    with col1:
        fig_treemap = px.treemap(
            product_df,
            path=["segment", "brand", "model"],
            values="review_count",
            color="segment",
            color_discrete_map=segment_colors,
            title="Segment → Brand → Model Hierarchy (sized by review count)",
        )
        st.plotly_chart(fig_treemap, use_container_width=True)

    with col2:
        fig_scatter = px.scatter(
            product_df,
            x="price_usd",
            y="rating",
            color="segment",
            size="review_count",
            hover_data=["brand", "model"],
            category_orders={"segment": segment_order},
            color_discrete_map=segment_colors,
            title="Price vs Rating (bubble size = review count)",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.divider()

    # ------------------------------------------------------------------
    # ROW 2: Segment mix + review volume
    # ------------------------------------------------------------------
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Share of Products by Segment")
        segment_counts = product_df["segment"].value_counts().reindex(segment_order).reset_index()
        segment_counts.columns = ["segment", "count"]
        fig_pie = px.pie(
            segment_counts, names="segment", values="count",
            color="segment", color_discrete_map=segment_colors,
            title="Segment Mix", hole=0.4,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col4:
        st.subheader("Total Reviews by Segment")
        reviews_by_segment = (
            product_df.groupby("segment")["review_count"].sum()
            .reindex(segment_order).reset_index()
        )
        fig_review_bar = px.bar(
            reviews_by_segment, x="segment", y="review_count",
            color="segment", color_discrete_map=segment_colors,
            category_orders={"segment": segment_order},
            title="Total Reviews by Segment",
        )
        fig_review_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_review_bar, use_container_width=True)

    st.divider()

    # ------------------------------------------------------------------
    # ROW 3: Per-segment deep dive — tabs, each with:
    #   1. Horizontal bar (price by model)
    #   2. Line chart (feature ratings across categories)
    #   3. Gauge chart (overall segment average rating)
    # ------------------------------------------------------------------
    st.subheader("Segment Deep Dive")
    tabs = st.tabs(segment_order)

    for tab, segment_name in zip(tabs, segment_order):
        with tab:
            segment_df = product_df[product_df["segment"] == segment_name]

            if segment_df.empty:
                st.info(f"No products found in the {segment_name} segment.")
                continue

            tab_col1, tab_col2, tab_col3 = st.columns([1.2, 1.2, 0.8])

            # ---- Horizontal bar: price by model ----
            with tab_col1:
                st.markdown(f"**Price by Model — {segment_name}**")
                price_by_model = segment_df[["model", "price_usd"]].sort_values("price_usd")
                fig_price = px.bar(
                    price_by_model, x="price_usd", y="model",
                    orientation="h",
                    color_discrete_sequence=[segment_colors[segment_name]],
                    title=f"{segment_name}: Price by Model",
                )
                fig_price.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig_price, use_container_width=True)

            # ---- Line chart: average feature ratings across categories ----
            with tab_col2:
                st.markdown(f"**Feature Ratings Profile — {segment_name}**")
                avg_features = segment_df[feature_cols].mean().reset_index()
                avg_features.columns = ["feature", "avg_rating"]
                avg_features["feature"] = (
                    avg_features["feature"].str.replace("_rating", "", regex=False)
                    .str.replace("_", " ").str.title()
                )
                fig_line = px.line(
                    avg_features, x="feature", y="avg_rating",
                    markers=True,
                    range_y=[0, 5],
                    color_discrete_sequence=[segment_colors[segment_name]],
                    title=f"{segment_name}: Avg Feature Ratings",
                )
                st.plotly_chart(fig_line, use_container_width=True)

            # ---- Gauge chart: overall average rating for this segment ----
            with tab_col3:
                st.markdown(f"**Overall Rating — {segment_name}**")
                overall_rating = segment_df["rating"].mean()
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=overall_rating,
                    number={"suffix": " / 5"},
                    gauge={
                        "axis": {"range": [0, 5]},
                        "bar": {"color": segment_colors[segment_name]},
                        "steps": [
                            {"range": [0, 2], "color": "#3a1f1f"},
                            {"range": [2, 3.5], "color": "#3a331f"},
                            {"range": [3.5, 5], "color": "#1f3a24"},
                        ],
                    },
                    title={"text": f"{segment_name} Avg Rating"},
                ))
                fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig_gauge, use_container_width=True)

            st.markdown(f"**Models in {segment_name}**")
            st.dataframe(
                segment_df[["brand", "model", "price_usd", "rating", "review_count"]]
                .sort_values("price_usd")
            )
# --------------------------------------------------------------------------
# SECTION 3: Product Recommender (brand picks source phone; price/rating
# filters control the CROSS-BRAND candidate pool for recommendations)
# --------------------------------------------------------------------------
def render_product_recommender(product_df: pd.DataFrame, similarity_df: pd.DataFrame) -> None:
    st.header("Find Similar Phones")

    filter_col1, filter_col2, filter_col3 = st.columns(3)

    with filter_col1:
        selected_brand = st.selectbox(
            "Select a brand (to pick your phone)",
            options=sorted(product_df["brand"].unique()),
            key="rec_brand_select",
        )

    price_min = float(product_df["price_usd"].min())
    price_max = float(product_df["price_usd"].max())

    with filter_col2:
        price_range = st.slider(
            "Price range (USD) for recommendations",
            min_value=price_min,
            max_value=price_max,
            value=(price_min, price_max),
            key="rec_price_range",
        )

    available_ratings = sorted(product_df["rating"].dropna().unique())
    rating_min = float(min(available_ratings))
    rating_max = float(max(available_ratings))

    with filter_col3:
        rating_range = st.slider(
            "Rating range for recommendations",
            min_value=rating_min,
            max_value=rating_max,
            value=(rating_min, rating_max),
            step=0.1,
            key="rec_rating_range",
        )

    brand_only_df = product_df[product_df["brand"] == selected_brand]

    if brand_only_df.empty:
        st.warning(f"No phones found for brand '{selected_brand}'.")
        return

    selected_model = st.selectbox(
        f"Select a {selected_brand} phone",
        options=sorted(brand_only_df["model"].unique()),
        key="rec_model_select",
    )

    top_n = st.slider("Number of recommendations", min_value=1, max_value=10, value=3, key="rec_top_n")

    if not selected_model:
        return

    candidate_df = product_df[
        (product_df["price_usd"].between(price_range[0], price_range[1]))
        & (product_df["rating"].between(rating_range[0], rating_range[1]))
    ].copy()

    candidate_models = candidate_df["model"].tolist()
    if selected_model in candidate_models:
        candidate_models.remove(selected_model)

    if not candidate_models:
        st.info(
            "No phones match the current Price/Rating range besides "
            f"'{selected_model}'. Widen the filters to see recommendations."
        )
        return

    similar_scores = similarity_df.loc[selected_model, candidate_models]
    similar_scores = similar_scores.sort_values(ascending=False)
    top_matches = similar_scores.head(top_n)

    st.subheader(f"Phones similar to {selected_model} (across all brands)")

    results = product_df[product_df["model"].isin(top_matches.index)].copy()
    results["similarity_score"] = results["model"].map(top_matches)
    results = results.sort_values("similarity_score", ascending=False)

    st.dataframe(results[["brand", "model", "price_usd", "segment", "rating", "similarity_score"]])

    fig2 = px.bar(
        results,
        x="model",
        y="similarity_score",
        color="brand",
        title=f"Similarity Scores for {selected_model} (colored by brand)",
    )
    st.plotly_chart(fig2, use_container_width=True)


# --------------------------------------------------------------------------
# Single-page layout (no sidebar navigation)
# --------------------------------------------------------------------------
render_segment_overview(product_df)
st.divider()
render_insights(product_df)
st.divider()
render_product_recommender(product_df, similarity_df)