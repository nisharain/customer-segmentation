import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Customer Segmentation", page_icon="🛒", layout="wide")

SEG_ORDER = ["Premium Loyalists", "Deal-Seeking Families", "Budget Browsers"]
SEG_COLOR = {"Premium Loyalists": "#E0A526", "Deal-Seeking Families": "#1F7A70", "Budget Browsers": "#6F7A94"}
ADVICE = {
    "Premium Loyalists": "High income and high spend. Offer premium wine/meat bundles, loyalty rewards, catalog and exclusive offers.",
    "Deal-Seeking Families": "Medium income, uses discounts a lot. Send coupons, family bundles, and email/website promotions.",
    "Budget Browsers": "Low spend, visits the website but rarely buys. Use low-cost entry offers and first-purchase discounts. Avoid expensive campaigns.",
}
PRODUCTS = ["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
CHANNELS = ["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]


@st.cache_resource
def load_model():
    return joblib.load("segmentation_model.pkl")


@st.cache_data
def load_data():
    try:
        return pd.read_csv("customers_with_segments.csv")
    except Exception:
        return None


bundle = load_model()
scaler, pca, kmeans = bundle["scaler"], bundle["pca"], bundle["kmeans"]
features, names = bundle["features"], bundle["names"]
data = load_data()


def predict(row: dict) -> str:
    X = pd.DataFrame([row])[features]
    cluster = int(kmeans.predict(pca.transform(scaler.transform(X)))[0])
    return names[cluster]


def colors(labels):
    return [SEG_COLOR[l] for l in labels]


# ---------- graphs ----------
def fig_scatter(row, seg):
    P = pca.transform(scaler.transform(data[features]))
    fig, ax = plt.subplots(figsize=(7, 5))
    for s in SEG_ORDER:
        m = (data["Segment"] == s).values
        ax.scatter(P[m, 0], P[m, 1], s=12, alpha=0.5, color=SEG_COLOR[s], label=s)
    new = pca.transform(scaler.transform(pd.DataFrame([row])[features]))
    ax.scatter(new[0, 0], new[0, 1], s=350, marker="*", color="red", edgecolor="black", linewidth=1.2, label="This customer", zorder=5)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title("Where this customer sits among all customers")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def fig_compare(row, seg):
    avg = data[data["Segment"] == seg]
    metrics = [("Income", "Income"), ("Total_Spent", "Total Spent"), ("Total_Purchases", "Total Purchases"), ("NumDealsPurchases", "Deal Purchases")]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.5))
    for ax, (col, label) in zip(axes, metrics):
        ax.bar(["Customer", "Segment avg"], [row[col], avg[col].mean()], color=["red", SEG_COLOR[seg]])
        ax.set_title(label, fontsize=11)
    fig.suptitle(f"This customer vs average {seg}")
    fig.tight_layout()
    return fig


def fig_sizes():
    counts = data["Segment"].value_counts().reindex(SEG_ORDER)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(counts, autopct="%1.1f%%", colors=colors(SEG_ORDER), startangle=90, textprops={"color": "white", "fontsize": 11})
    ax.legend(SEG_ORDER, loc="lower center", bbox_to_anchor=(0.5, -0.12), fontsize=9, ncol=1, frameon=False)
    ax.set_title("Customers per segment")
    fig.tight_layout()
    return fig


def fig_profile():
    cols = [("Income", "Average income"), ("Total_Spent", "Average total spend"), ("NumDealsPurchases", "Avg discount purchases"), ("Campaigns_Accepted", "Avg campaigns accepted")]
    g = data.groupby("Segment")[[c for c, _ in cols]].mean().reindex(SEG_ORDER)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    for ax, (c, t) in zip(axes, cols):
        ax.bar(range(3), g[c], color=colors(SEG_ORDER))
        ax.set_xticks(range(3)); ax.set_xticklabels(["Premium", "Deal", "Budget"])
        ax.set_title(t, fontsize=11)
    fig.tight_layout()
    return fig


def fig_group(cols, labels, title):
    g = data.groupby("Segment")[cols].mean().reindex(SEG_ORDER)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(cols)); w = 0.27
    for i, s in enumerate(SEG_ORDER):
        ax.bar(x + (i - 1) * w, g.loc[s], w, color=SEG_COLOR[s], label=s)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_title(title); ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def fig_response():
    rc = ["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5", "Response"]
    g = (data.groupby("Segment")[rc].mean().reindex(SEG_ORDER) * 100)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(rc)); w = 0.27
    for i, s in enumerate(SEG_ORDER):
        ax.bar(x + (i - 1) * w, g.loc[s], w, color=SEG_COLOR[s], label=s)
    ax.set_xticks(x); ax.set_xticklabels(["Cmp1", "Cmp2", "Cmp3", "Cmp4", "Cmp5", "Last"])
    ax.set_ylabel("% accepted"); ax.set_title("Campaign acceptance rate by segment"); ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# ---------- page ----------
st.title("🛒 Customer Segmentation")
st.caption("Predict a customer's segment with Machine Learning and see how they compare with all customers.")

tab1, tab2 = st.tabs(["🔮 Predict Segment", "📊 Segment Insights"])

with tab1:
    st.subheader("Enter Customer Details")
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age", 18, 100, 46)
        income = st.number_input("Income (yearly)", 0, 200000, 76000, step=1000)
        total_spent = st.number_input("Total Spent (last 2 years)", 0, 3000, 1325)
        children = st.number_input("Children (kids + teens at home)", 0, 5, 0)
        tenure = st.number_input("Tenure Days (days since enrollment)", 0, 1500, 350)
        recency = st.number_input("Recency (days since last purchase)", 0, 100, 48)
        campaigns = st.number_input("Campaigns Accepted (0-6)", 0, 6, 1)
    with c2:
        web = st.number_input("Web Purchases", 0, 30, 5)
        catalog = st.number_input("Catalog Purchases", 0, 30, 6)
        store = st.number_input("Store Purchases", 0, 30, 9)
        deals = st.number_input("Deal Purchases (with discount)", 0, 20, 1)
        visits = st.number_input("Web Visits per Month", 0, 25, 3)
        st.info(f"Total Purchases = Web + Catalog + Store = **{web + catalog + store}**")

    if st.button("Predict Segment", type="primary"):
        row = {
            "Age": age, "Income": income, "Total_Spent": total_spent, "Children": children,
            "Tenure_Days": tenure, "Recency": recency, "Total_Purchases": web + catalog + store,
            "NumDealsPurchases": deals, "NumWebVisitsMonth": visits, "NumWebPurchases": web,
            "NumCatalogPurchases": catalog, "NumStorePurchases": store, "Campaigns_Accepted": campaigns,
        }
        seg = predict(row)
        st.success(f"Customer Segment: {seg}")
        st.info(ADVICE[seg])
        if data is not None:
            g1, g2 = st.columns(2)
            with g1:
                st.pyplot(fig_scatter(row, seg))
            with g2:
                st.pyplot(fig_sizes())
            st.pyplot(fig_compare(row, seg))
        else:
            st.warning("Put customers_with_segments.csv in the same folder as app.py to see the graphs.")

with tab2:
    if data is None:
        st.warning("Put customers_with_segments.csv in the same folder as app.py to see the graphs.")
    else:
        st.subheader("How the 3 segments differ")
        a, b = st.columns([1, 2])
        with a:
            st.pyplot(fig_sizes())
        with b:
            st.pyplot(fig_profile())
        c, d = st.columns(2)
        with c:
            st.pyplot(fig_group(PRODUCTS, ["Wine", "Fruits", "Meat", "Fish", "Sweets", "Gold"], "Average spend per product"))
        with d:
            st.pyplot(fig_group(CHANNELS, ["Web", "Catalog", "Store"], "Average purchases per channel"))
        st.pyplot(fig_response())
        st.dataframe(data.groupby("Segment")[["Age", "Income", "Total_Spent", "Children", "Total_Purchases", "NumDealsPurchases", "NumWebVisitsMonth", "Campaigns_Accepted"]].mean().round(1).reindex(SEG_ORDER))