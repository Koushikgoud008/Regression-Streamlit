import streamlit as st
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config("Diamond Price Prediction", layout="centered")

def load_css(file):
    with open(file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

st.sidebar.title("Model Selection")
model_name = st.sidebar.selectbox(
    "Choose Regression Model",
    [
        "Linear Regression",
        "Polynomial Regression",
        "Ridge Regression",
        "Lasso Regression",
        "ElasticNet Regression"
    ]
)

@st.cache_data
def load_data():
    return sns.load_dataset("diamonds")

df = load_data()

st.markdown(f"""
<div class="card">
<h1>Diamond Price Prediction</h1>
<p>Using <b>{model_name}</b></p>
</div>
""", unsafe_allow_html=True)

st.subheader("Dataset Preview")
st.dataframe(df.head())

x = df[["carat"]]
y = df["price"]

if model_name == "Polynomial Regression":
    poly = PolynomialFeatures(degree=2)
    x = poly.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

if model_name == "Linear Regression":
    model = LinearRegression()
elif model_name == "Polynomial Regression":
    model = LinearRegression()
elif model_name == "Ridge Regression":
    model = Ridge(alpha=1.0)
elif model_name == "Lasso Regression":
    model = Lasso(alpha=0.05)
else:
    model = ElasticNet(alpha=0.1, l1_ratio=0.5)

model.fit(x_train, y_train)
y_pred = model.predict(x_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("MODEL PERFORMANCE")
c1, c2 = st.columns(2)
c1.metric("MAE", f"{mae:.2f}")
c2.metric("RMSE", f"{mse:.2f}")
c3, c4 = st.columns(2)
c3.metric("R2", f"{r2:.3f}")

st.subheader("Residual Analysis")

residuals = y_test - y_pred

fig1, ax1 = plt.subplots()
ax1.scatter(y_pred, residuals, alpha=0.4)
ax1.axhline(0, color="red")
ax1.set_xlabel("Predicted Price")
ax1.set_ylabel("Residuals")
st.pyplot(fig1)

st.subheader("Actual vs Predicted Distribution")

fig2, ax2 = plt.subplots()
sns.kdeplot(y_test, label="Actual", ax=ax2)
sns.kdeplot(y_pred, label="Predicted", ax=ax2)
ax2.legend()
st.pyplot(fig2)

st.subheader("Predict Price")

carat = st.slider(
    "Carat",
    float(df.carat.min()),
    float(df.carat.max()),
    1.0
)

if model_name == "Polynomial Regression":
    carat_t = scaler.transform(poly.transform([[carat]]))
else:
    carat_t = scaler.transform([[carat]])

price = model.predict(carat_t)[0]

st.markdown(
    f'<div class="prediction-box">Predicted Price: ${price:.2f}</div>',
    unsafe_allow_html=True
)
