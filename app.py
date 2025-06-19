import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so
import json
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import plotly.express as px

st.set_page_config(page_title="Enhanced Dataset Visualizer", layout="wide")

st.title("📊 Enhanced Dataset Visualizer")

uploaded_file = st.file_uploader("Upload your data file (CSV, Excel, or JSON)", type=["csv", "xlsx", "json"])

@st.cache_data
def load_data(file):
    ext = file.name.split(".")[-1]
    if ext == "csv":
        return pd.read_csv(file)
    elif ext == "xlsx":
        return pd.read_excel(file)
    elif ext == "json":
        return pd.read_json(file)
    else:
        return None

def download_chart(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png")
    st.download_button("Download Chart as PNG", buf.getvalue(), "chart.png", "image/png")

if uploaded_file:
    try:
        df = load_data(uploaded_file)
        st.success("File successfully loaded!")

        with st.expander("📌 Dataset Overview", expanded=True):
            st.write("### First 5 rows")
            st.dataframe(df.head())
            st.write("**Shape:**", df.shape)
            st.write("**Missing values per column:**")
            st.dataframe(df.isnull().sum())
            st.write("**Data types:**")
            st.dataframe(df.dtypes)

        with st.sidebar:
            st.header("🔍 Chart Options")
            view = st.radio("Choose View", ["Numerical Analysis", "Categorical Analysis", "Correlation Heatmap", "Scatter Plot", "ML Playground"])

        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

        if view == "Numerical Analysis" and numeric_cols:
            column = st.selectbox("Select numeric column", numeric_cols)
            bins = st.slider("Number of bins", 5, 100, 20)
            fig, ax = plt.subplots()
            ax.hist(df[column].dropna(), bins=bins, color="skyblue", edgecolor="black")
            ax.set_title(f"Histogram of {column}")
            st.pyplot(fig)
            download_chart(fig)

        elif view == "Categorical Analysis" and cat_cols:
            column = st.selectbox("Select categorical column", cat_cols)
            top_k = st.slider("Top categories to show", 5, 30, 10)
            counts = df[column].value_counts().head(top_k)
            fig = px.bar(x=counts.index, y=counts.values, labels={'x': column, 'y': 'Count'}, title=f"Top {top_k} Categories in {column}")
            st.plotly_chart(fig, use_container_width=True)

        elif view == "Correlation Heatmap" and numeric_cols:
            fig, ax = plt.subplots(figsize=(10, 6))
            corr = df[numeric_cols].corr()
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
            ax.set_title("Correlation Heatmap")
            st.pyplot(fig)
            download_chart(fig)

        elif view == "Scatter Plot" and len(numeric_cols) >= 2:
            x_col = st.selectbox("X-axis", numeric_cols, key="x_axis")
            y_col = st.selectbox("Y-axis", numeric_cols, key="y_axis")
            fig = px.scatter(df, x=x_col, y=y_col, title=f"Scatter Plot: {x_col} vs {y_col}")
            st.plotly_chart(fig, use_container_width=True)

        elif view == "ML Playground":
            st.subheader("🧠 Machine Learning Playground")
            target_col = st.selectbox("Select target column", df.columns)
            features = st.multiselect("Select features (at least 1)", [col for col in df.columns if col != target_col])
            if features and target_col:
                df_clean = df[features + [target_col]].dropna()
                X = pd.get_dummies(df_clean[features])
                y = df_clean[target_col]

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                model_choice = st.radio("Choose model", ["Random Forest", "Logistic Regression"])
                if model_choice == "Random Forest":
                    model = RandomForestClassifier()
                else:
                    model = LogisticRegression(max_iter=1000)

                model.fit(X_train, y_train)
                preds = model.predict(X_test)

                acc = accuracy_score(y_test, preds)
                cm = confusion_matrix(y_test, preds)

                st.write(f"**Accuracy:** {acc:.2f}")
                st.write("**Confusion Matrix:**")
                st.dataframe(cm)

    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Upload a CSV, Excel, or JSON file to begin.")
