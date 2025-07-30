import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Solar Data Analysis", layout="wide")
st.title("Solar Production Data Analysis")

uploaded_file = st.sidebar.file_uploader("Upload your solar CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Info")
   # buffer = []
    df.info()
   # st.text("\n".join(buffer))

    st.subheader("Descriptive Statistics")
    st.dataframe(df.describe().T)

    if st.checkbox("Show Histograms"):
        st.subheader("Histograms")
        fig, ax = plt.subplots(figsize=(12, 8))
        df.select_dtypes(include='number').hist(ax=ax)
        st.pyplot(fig)

    if st.checkbox("Show Boxplots"):
        st.subheader("Boxplots")
        for col in df.select_dtypes(include='number').columns:
            fig, ax = plt.subplots()
            sns.boxplot(x=df[col], color='skyblue', ax=ax)
            st.pyplot(fig)

    if st.checkbox("Show Count Plots for Categorical Columns"):
        st.subheader("Count Plots")
        for col in df.select_dtypes(include='object').columns:
            fig, ax = plt.subplots()
            sns.countplot(x=col, data=df, palette='Set2', ax=ax)
            plt.xticks(rotation=90)
            st.pyplot(fig)

    if st.checkbox("Show Correlation Matrix"):
        st.subheader("Correlation Matrix")
        numerical_vars = df.select_dtypes(include='number')
        corr = numerical_vars.corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    if st.checkbox("Perform Linear Regression"):
        st.subheader("Linear Regression Model")

        numerical_columns = df.select_dtypes(include=np.number).columns.tolist()
        target_column = st.selectbox("Select target variable (Y)", numerical_columns)
        feature_columns = st.multiselect("Select feature variables (X)", [col for col in numerical_columns if col != target_column])

        if feature_columns:
            X = df[feature_columns]
            y = df[target_column]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LinearRegression()
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            st.write("**Model Coefficients:**")
            coeff_df = pd.DataFrame({"Feature": feature_columns, "Coefficient": model.coef_})
            st.dataframe(coeff_df)

            st.write("**Model Performance:**")
            st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
            st.write(f"R-squared Score: {r2_score(y_test, y_pred):.2f}")

            st.write("**Model Performance:**")
            st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
            st.write(f"R-squared Score: {r2_score(y_test, y_pred):.2f}")

            # Create and display a table of actual vs predicted values
            results_df = pd.DataFrame({
             "Actual": y_test.values,
             "Predicted": y_pred
            })
            st.subheader("Actual vs Predicted Values")
            st.dataframe(results_df.head(20))

            import io

           csv = results_df.to_csv(index=False)
           st.download_button(
           label="Download Predictions as CSV",
           data=csv,
           file_name='predicted_vs_actual.csv',
           mime='text/csv',
            
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred, alpha=0.5)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_title("Actual vs Predicted")
            st.pyplot(fig)
        else:
            st.warning("Please select at least one feature variable.")
)
else:
    st.info("Please upload a CSV file to begin.")
