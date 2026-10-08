
from pathlib import Path
 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.dummy import DummyRegressor
 
REPO = "https://github.com/Nosferatu0-alt/Auto-MPG-Exploratory-Data-Analysis-Machine-Learning"
KMPL = 0.425144                 
MULTI_COLS = ["weight", "horsepower", "model_year"]
DEGREE = 2
NUM = ["cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year"]
 
st.set_page_config(page_title="Previsão de consumo (Auto MPG)", page_icon="🚗", layout="wide")
 
 
# dados
def find_csv() -> Path | None:
    here = Path(__file__).resolve().parent
    for p in [here.parent / "data" / "auto-mpg.csv", here / "auto-mpg.csv", Path("auto-mpg.csv")]:
        if p.exists():
            return p
    try:
        import kagglehub
        folder = kagglehub.dataset_download("uciml/autompg-dataset")
        p = Path(folder) / "auto-mpg.csv"
        return p if p.exists() else None
    except Exception:
        return None
 
 
@st.cache_data(show_spinner="Carregando o dataset...")
def read_csv(path: str | None, uploaded: bytes | None) -> pd.DataFrame | None:
    if uploaded is not None:
        import io
        raw = pd.read_csv(io.BytesIO(uploaded))
    elif path:
        raw = pd.read_csv(path)
    else:
        return None
    df = raw.rename(columns={"model year": "model_year", "car name": "car_name"})
    df["horsepower"] = pd.to_numeric(df["horsepower"], errors="coerce")   # '?' -> NaN
    df = df.dropna(subset=["mpg"]).drop_duplicates().reset_index(drop=True)
    return df
 
 
#modelos
def make_multi():
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("lr", LinearRegression())])
 
 
def make_poly(degree):
    return Pipeline([("scaler", StandardScaler()),
                     ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
                     ("lr", LinearRegression())])
 
 
def metrics(y, p):
    return {"MAE (mpg)": mean_absolute_error(y, p),
            "RMSE (mpg)": float(np.sqrt(mean_squared_error(y, p))),
            "R²": r2_score(y, p)}
 
 
@st.cache_resource(show_spinner="Treinando os modelos...")
def train(df: pd.DataFrame):
    X, y = df[NUM], df["mpg"]
    X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.25, random_state=42)
    X_tv, y_tv = pd.concat([X_train, X_val]), pd.concat([y_train, y_val])
 
    models = {
        "Baseline (média)": (DummyRegressor(strategy="mean"), ["weight"]),
        "Linear simples": (LinearRegression(), ["weight"]),
        "Linear múltipla": (make_multi(), MULTI_COLS),
        f"Polinomial grau {DEGREE}": (make_poly(DEGREE), ["weight"]),
    }
    rows = []
    for name, (m, cols) in models.items():
        m.fit(X_tv[cols], y_tv)
        rows.append({"Modelo": name, **metrics(y_test, m.predict(X_test[cols]))})
    return models, pd.DataFrame(rows).set_index("Modelo"), (len(X_train), len(X_val), len(X_test)), X_tv, y_tv
 
 
# app
st.title("🚗 Previsão de consumo de combustível (Auto MPG)")
st.caption("Regressão linear simples, múltipla e polinomial · Aprendizado de Máquina, Unidade 2 · "
           f"[código e relatório no GitHub]({REPO})")
 
csv_path = find_csv()
uploaded = None
df = read_csv(str(csv_path) if csv_path else None, None)
if df is None:
    st.warning("Não consegui baixar o dataset automaticamente. Envie o `auto-mpg.csv` "
               "(https://www.kaggle.com/datasets/uciml/autompg-dataset).")
    up = st.file_uploader("auto-mpg.csv", type="csv")
    if up is None:
        st.stop()
    df = read_csv(None, up.getvalue())
 
models, test_table, sizes, X_tv, y_tv = train(df)
 
tab_prev, tab_res, tab_sobre = st.tabs(["Previsão", "Resultados", "Sobre o estudo"])
 
#previsão
with tab_prev:
    wmin, wmax = int(X_tv.weight.min()), int(X_tv.weight.max())
    hmin, hmax = int(X_tv.horsepower.min()), int(X_tv.horsepower.max())
    ymin, ymax = int(X_tv.model_year.min()), int(X_tv.model_year.max())
 
    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("Características do carro")
        weight = st.slider("Peso (lb)", 1000, 6000, 3000, step=25)
        hp = st.slider("Potência (hp)", 40, 300, 100, step=1)
        year = st.slider("Ano do modelo (70 = 1970)", 65, 90, 76, step=1)
 
        outside = []
        if not wmin <= weight <= wmax: outside.append(f"peso fora de {wmin}–{wmax} lb")
        if not hmin <= hp <= hmax: outside.append(f"potência fora de {hmin}–{hmax} hp")
        if not ymin <= year <= ymax: outside.append(f"ano fora de {1900+ymin}–{1900+ymax}")
        if outside:
            st.warning("**Extrapolação:** " + "; ".join(outside) + ". Os modelos só conhecem a "
                       "faixa dos dados de treino (carros de 1970–1982); a previsão pode não fazer sentido.")
 
        x_simple = pd.DataFrame({"weight": [weight]})
        x_multi = pd.DataFrame({"weight": [weight], "horsepower": [hp], "model_year": [year]})
        preds = {
            "Linear simples": float(models["Linear simples"][0].predict(x_simple)[0]),
            "Linear múltipla": float(models["Linear múltipla"][0].predict(x_multi)[0]),
            f"Polinomial grau {DEGREE}": float(models[f"Polinomial grau {DEGREE}"][0].predict(x_simple)[0]),
        }
        st.subheader("Consumo previsto")
        for name, p in preds.items():
            st.metric(name, f"{p:.1f} mpg", f"≈ {p * KMPL:.1f} km/L", delta_color="off")
        if min(preds.values()) < float(df.mpg.min()):
            st.info(f"Alguma previsão ficou abaixo do menor consumo observado no dataset "
                    f"({df.mpg.min():.0f} mpg): sinal de extrapolação.")
        st.caption("Referência de erro: RMSE de teste da múltipla ≈ "
                   f"{test_table.loc['Linear múltipla', 'RMSE (mpg)']:.1f} mpg. "
                   "Use como estimativa de ordem de grandeza, não como medida exata.")
 
    with c2:
        st.subheader("Peso × consumo")
        grid = pd.DataFrame({"weight": np.linspace(max(1000, df.weight.min() - 200), min(6000, df.weight.max() + 200), 300)})
        fig, ax = plt.subplots(figsize=(7, 4.4))
        ax.scatter(df.weight, df.mpg, s=14, alpha=.35, color="gray", label="carros do dataset")
        ax.plot(grid.weight, models["Linear simples"][0].predict(grid), "k-", lw=2, label="linear simples")
        ax.plot(grid.weight, models[f"Polinomial grau {DEGREE}"][0].predict(grid), color="tab:blue", lw=2, label=f"polinomial grau {DEGREE}")
        ax.scatter([weight], [preds["Linear múltipla"]], marker="*", s=260, color="tab:red", zorder=5, label="múltipla (seu carro)")
        ax.axvspan(grid.weight.min(), wmin, color="orange", alpha=.08)
        ax.axvspan(wmax, grid.weight.max(), color="orange", alpha=.08)
        ax.set_xlabel("weight (peso, lb)"); ax.set_ylabel("mpg (milhas por galão)")
        ax.grid(alpha=.3); ax.legend(fontsize=8)
        st.pyplot(fig)
        st.caption("Áreas alaranjadas: fora da faixa de peso dos dados de treino. "
                   "A estrela usa também potência e ano; as curvas usam só o peso.")
 
# ---- resultados
with tab_res:
    st.subheader("Desempenho no conjunto de teste")
    st.write(f"Separação 60/20/20 (`random_state=42`): treino {sizes[0]}, validação {sizes[1]}, teste {sizes[2]}. "
             "Modelos reajustados em treino + validação; teste usado uma única vez, como no notebook.")
    st.dataframe(test_table.style.format({"MAE (mpg)": "{:.3f}", "RMSE (mpg)": "{:.3f}", "R²": "{:.3f}"}))
    st.markdown(
        "- A **múltipla** (peso, potência e ano) teve o menor erro; adicionar `model_year` ajudou mais do que aumentar o grau.\n"
        "- O polinomial de **grau 2** melhora pouco a reta; os graus 3 e 5 não melhoraram na validação.\n"
        "- Os erros são maiores nos carros muito econômicos (> 30 mpg): variância não constante."
    )
    st.caption("Os números devem coincidir com o relatório quando o dataset oficial do Kaggle é usado.")
 