import os

import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.cluster import KMeans
import numpy as np

st.set_page_config(page_title="Mall Customer Segmentation Report", layout="wide")

st.title("Mall Customer Segmentation Report")
st.markdown("Breaks down mall customers into groups based on income and spending using K-means. Mess with the sidebar to filter by gender or age.")

@st.cache_data
def load_data():
    data_path = "data/Mall_Customers.csv"
    df = pd.read_csv(data_path)
    df["Annual Income"] = df["Annual Income (k$)"] * 1000  
    df = df.drop(columns=["Annual Income (k$)"])
    return df

df = load_data()

st.sidebar.header("Filter Options")
st.sidebar.markdown("Pick a gender or slide the age range to zoom in on groups.")
gender_filter = st.sidebar.selectbox("Gender", options=["All", "Male", "Female"])
age_range = st.sidebar.slider("Age Range", min_value=int(df["Age"].min()), 
                              max_value=int(df["Age"].max()), value=(18, 70))

filtered_df = df
if gender_filter != "All":
    filtered_df = filtered_df[filtered_df["Gender"] == gender_filter]
filtered_df = filtered_df[(filtered_df["Age"] >= age_range[0]) & 
                         (filtered_df["Age"] <= age_range[1])]

st.header("1. Data Overview")
st.markdown("Gives a peek at customer data and a rough summary of income, spending, age.")

col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("Dataset Preview")
    st.markdown("First 5 customers with ID, gender, age, income, spending score (1–100, higher means more cash spent).")
    st.dataframe(
        filtered_df[["CustomerID", "Gender", "Age", "Annual Income", "Spending Score (1-100)"]]
        .head()
        .style.format({"Annual Income": "${:,.0f}"})
        .set_properties(**{"text-align": "center"}),
        use_container_width=True
    )

with col2:
    st.subheader("Summary Statistics")
    st.markdown("""
    Summarizes income, spending, age:  
    - Average: Typical value.  
    - Spread: How much values vary.  
    - Minimum: Lowest value.  
    - 25%: Below this for 25% of customers.  
    - Median: Middle value.  
    - 75%: Below this for 75% of customers.  
    - Maximum: Highest value.
    """)
    summary = filtered_df[["Annual Income", "Spending Score (1-100)", "Age"]].describe().round(2)
    summary.index = ["Count", "Average", "Spread", "Minimum", "25%", "Median", "75%", "Maximum"]
    summary = summary.drop("Count")
    st.dataframe(
        summary.style.format({"Annual Income": "${:,.0f}"})
        .set_properties(**{"text-align": "center"}),
        use_container_width=True
    )

st.header("2. Exploratory Insights")
st.markdown("Plots income against spending score, colored by gender, sized by age. Hover over dots for details.")
fig = px.scatter(filtered_df, x="Annual Income", y="Spending Score (1-100)", 
                 color="Gender", size="Age", hover_data=["CustomerID"],
                 title="Annual Income vs. Spending Score")
fig.update_layout(xaxis_title="Annual Income ($)", yaxis_title="Spending Score (1-100)")
fig.update_traces(marker=dict(opacity=0.8))
st.plotly_chart(fig, use_container_width=True)

st.header("3. Customer Segmentation")
st.markdown("Splits customers into 5 groups based on income and spending with K-means.")

X = filtered_df[["Annual Income", "Spending Score (1-100)"]].values

st.subheader("Elbow Method")
st.markdown("Plots inertia for 1–10 clusters. Elbow at 5 seems about right.")
if len(X) > 0:
    inertia = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)
    fig_elbow = px.line(x=range(1, 11), y=inertia, title="Elbow Method for Optimal Clusters")
    fig_elbow.update_layout(xaxis_title="Number of Clusters", yaxis_title="Inertia")
    st.plotly_chart(fig_elbow, use_container_width=True)
else:
    st.warning("No data with these filters. Tweak them to see clustering.")

if len(X) > 0:
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    filtered_df["Cluster"] = kmeans.fit_predict(X)
    
    st.subheader("Customer Segments")
    st.markdown("Shows customers split into 5 groups by income and spending. Hover for age and gender.")
    fig_clusters = px.scatter(filtered_df, x="Annual Income", y="Spending Score (1-100)", 
                             color="Cluster", hover_data=["Age", "Gender"],
                             title="Customer Segments by Income and Spending")
    fig_clusters.update_layout(xaxis_title="Annual Income ($)", yaxis_title="Spending Score (1-100)")
    fig_clusters.update_traces(marker=dict(size=10, opacity=0.8))
    st.plotly_chart(fig_clusters, use_container_width=True)
    
    st.subheader("Cluster Profiles")
    st.markdown("Gives average income, spending score, age for each group.")
    cluster_summary = filtered_df.groupby("Cluster")[["Annual Income", "Spending Score (1-100)", "Age"]].mean().round(2)
    st.dataframe(
        cluster_summary.style.format({"Annual Income": "${:,.0f}"})
        .set_properties(**{"text-align": "center"}),
        use_container_width=True
    )

    st.subheader("Key Insights")
    st.markdown("""
    - **Cluster 0**: Looking at the average income and spending scores, this group stands out with about $100,000 income and an 80 spending score, making them best candidates to target for high-cost products.  
    - **Cluster 1**: Low income, big spenders, likely to be loyal if serviced correctly. By checking the averages, this group has around $30,000 income but spends a lot (70 score), hinting they might stick around with the right incentives.  
    - **Cluster 2**: Medium-income, low spending. Not much to derive.  
    - **Cluster 3**: Average income, balanced spender, ideal for general market.  
    - **Cluster 4**: Young, low-income, low spenders—could grow into loyal customers later.
    """)


st.markdown("Made by Mohamed Ali.")



st.markdown(" Data: Mall_Customers.csv (200 customers). cleaned, Fixedm and Deployed by Streamlit, Pandas, scikit-learn, Plotly.")
st.markdown("https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python?resource=download")