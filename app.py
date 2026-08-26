import joblib
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="Predicción de Churn Bancario",
    page_icon="🏦",
    layout="centered",
)

st.title("🏦 Sistema Predictivo de Deserción de Clientes")
st.markdown(
    "Herramienta interactiva para la detección temprana de clientes en"
    " riesgo de abandono."
)


@st.cache_resource
def load_model():
    return joblib.load("modelo_churn_random_forest.joblib")


model = load_model()

with st.form("churn_form"):
    st.subheader("Datos del Cliente")
    col1, col2 = st.columns(2)

    with col1:
        credit_score = st.number_input(
            "Puntaje Crediticio (Credit Score)",
            min_value=300,
            max_value=900,
            value=650,
        )
        age = st.number_input("Edad", min_value=18, max_value=100, value=40)
        tenure = st.number_input(
            "Antigüedad con el banco (Años)", min_value=0, max_value=10, value=3
        )
        balance = st.number_input(
            "Saldo en Cuenta ($)", min_value=0.0, value=50000.0, step=1000.0
        )
        num_products = st.selectbox(
            "Cantidad de Productos Contratados", [1, 2, 3, 4], index=0
        )

    with col2:
        has_card = st.selectbox(
            "¿Tiene Tarjeta de Crédito?",
            [("Sí", 1), ("No", 0)],
            format_func=lambda x: x[0],
        )
        is_active = st.selectbox(
            "¿Es Miembro Activo?",
            [("Sí", 1), ("No", 0)],
            format_func=lambda x: x[0],
        )
        salary = st.number_input(
            "Salario Estimado Anual ($)",
            min_value=0.0,
            value=60000.0,
            step=1000.0,
        )
        country = st.selectbox("País", ["Francia", "Alemania", "España"])
        gender = st.selectbox("Género", ["Masculino", "Femenino"])

    submit = st.form_submit_button("🔍 Evaluar Riesgo de Deserción")

if submit:
    is_germany = 1 if country == "Alemania" else 0
    is_spain = 1 if country == "España" else 0
    is_male = 1 if gender == "Masculino" else 0

    features = np.array([[
        credit_score,
        age,
        tenure,
        balance,
        num_products,
        has_card[1],
        is_active[1],
        salary,
        is_germany,
        is_spain,
        is_male,
        0,
    ]])

    prediccion = model.predict(features)[0]
    probabilidad = model.predict_proba(features)[0][1]

    st.markdown("---")
    st.subheader("Resultado de la Predicción")

    if prediccion == 1:
        st.error(
            f"⚠️ **ALERTA: Alta Probabilidad de Deserción"
            f" ({probabilidad * 100:.2f}%)**"
        )
        st.write(
            "Se recomienda aplicar estrategias proactivas de fidelización o"
            " revisión de condiciones."
        )
    else:
        st.success(
            f"✅ **CLIENTE RETENIDO (Probabilidad de fuga:"
            f" {probabilidad * 100:.2f}%)**"
        )
        st.write(
            "El cliente muestra un comportamiento financiero estable dentro de"
            " la entidad."
        )
