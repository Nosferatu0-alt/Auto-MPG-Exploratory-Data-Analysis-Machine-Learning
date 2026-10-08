
import io
from pathlib import Path
 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
 
# ============================================================ CONFIGURAÇÕES
REPO = "https://github.com/Nosferatu0-alt/Auto-MPG-Exploratory-Data-Analysis-Machine-Learning"
KMPL = 0.425144                      # 1 mpg = 0,425144 km/L
DEGREE = 2
MULTI_COLS = ["weight", "horsepower", "model_year"]
NUMERIC_COLS = ["cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year"]
 
st.set_page_config(page_title="Auto MPG | Machine Learning", layout="wide", initial_sidebar_state="collapsed")
 
# ============================================================ ESTILO
st.markdown(
    """
    <style>
    .stApp { background-color: #f6f8fb; color: #172033; }
    .block-container { max-width: 1250px; padding-top: 2rem; padding-bottom: 3rem; }
    h1, h2, h3 { color: #172033; font-weight: 700; }
    p { color: #596579; }
 
    .hero { background: linear-gradient(135deg, #172554 0%, #1d4ed8 55%, #3b82f6 100%);
            padding: 2.2rem 2.5rem; border-radius: 18px; margin-bottom: 1.8rem;
            box-shadow: 0 10px 30px rgba(30, 64, 175, 0.18); }
    .hero-title { color: white !important; font-size: 2.25rem !important; font-weight: 750 !important; margin-bottom: 0.35rem !important; }
    .hero-subtitle { color: #dbeafe !important; font-size: 1rem; margin-bottom: 1.2rem; }
    .hero-link { display: inline-block; color: white !important; text-decoration: none;
                 border: 1px solid rgba(255,255,255,0.35); padding: 0.45rem 0.85rem;
                 border-radius: 8px; font-size: 0.85rem; transition: 0.2s; }
    .hero-link:hover { background-color: rgba(255,255,255,0.12); }
 
    .card { background: white; border: 1px solid #e5eaf1; border-radius: 14px;
            padding: 1.25rem 1.4rem; box-shadow: 0 4px 16px rgba(15, 23, 42, 0.05); margin-bottom: 1rem; }
    .card-title { color: #172033; font-size: 1.05rem; font-weight: 700; margin-bottom: 0.25rem; }
    .card-description { color: #6b7688; font-size: 0.88rem; line-height: 1.5; }
 
    .metric-card { background: white; border: 1px solid #e5eaf1; border-radius: 14px;
                   padding: 1.15rem 1.25rem; min-height: 135px; box-shadow: 0 4px 16px rgba(15, 23, 42, 0.05); }
    .metric-name { color: #667085; font-size: 0.83rem; font-weight: 600; margin-bottom: 0.45rem; }
    .metric-value { color: #172033; font-size: 1.75rem; font-weight: 750; line-height: 1.1; }
    .metric-secondary { color: #2563eb; font-size: 0.82rem; font-weight: 600; margin-top: 0.45rem; }
 
    .section-label { color: #2563eb; font-size: 0.78rem; font-weight: 750; letter-spacing: 0.08em;
                     text-transform: uppercase; margin-bottom: 0.25rem; }
    .section-title { color: #172033; font-size: 1.55rem; font-weight: 750; margin-bottom: 1rem; }
 
    div[data-baseweb="slider"] { margin-bottom: 1rem; }
    label { font-weight: 600 !important; color: #344054 !important; }
    button[data-baseweb="tab"] { font-weight: 600; color: #667085; }
    button[data-baseweb="tab"][aria-selected="true"] { color: #2563eb; }
    div[data-testid="stAlert"] { border-radius: 10px; }
 
    .footer { text-align: center; color: #98a2b3; font-size: 0.78rem; margin-top: 2.5rem;
              padding-top: 1.5rem; border-top: 1px solid #e5eaf1; }
    .footer a { color: #98a2b3; }
    </style>
    """,
    unsafe_allow_html=True,
)
 
 
def metric_card(name: str, value: str, secondary: str, value_size: str | None = None) -> str:
    style = f' style="font-size:{value_size};"' if value_size else ""
    return (f'<div class="metric-card"><div class="metric-name">{name}</div>'
            f'<div class="metric-value"{style}>{value}</div>'
            f'<div class="metric-secondary">{secondary}</div></div>')
 
 
def section(label: str, title: str) -> None:
    st.markdown(f'<div class="section-label">{label}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
 
 
def card(title: str, body: str) -> str:
    return (f'<div class="card"><div class="card-title">{title}</div>'
            f'<div class="card-description">{body}</div></div>')
 
 
# ============================================================ DADOS
def find_csv() -> Path | None:
    """Procura o dataset localmente; se não achar, tenta baixar pelo kagglehub."""
    here = Path(__file__).resolve().parent
    for path in [here.parent / "data" / "auto-mpg.csv", here / "auto-mpg.csv", Path("auto-mpg.csv")]:
        if path.exists():
            return path
    try:
        import kagglehub
        folder = kagglehub.dataset_download("uciml/autompg-dataset")
        path = Path(folder) / "auto-mpg.csv"
        return path if path.exists() else None
    except Exception:
        return None
 
 
@st.cache_data(show_spinner="Carregando o dataset...")
def read_csv(path: str | None, uploaded: bytes | None) -> pd.DataFrame | None:
    if uploaded is not None:
        raw = pd.read_csv(io.BytesIO(uploaded))
    elif path:
        raw = pd.read_csv(path)
    else:
        return None
    df = raw.rename(columns={"model year": "model_year", "car name": "car_name"})
    df["horsepower"] = pd.to_numeric(df["horsepower"], errors="coerce")      # '?' -> NaN
    return df.dropna(subset=["mpg"]).drop_duplicates().reset_index(drop=True)
 
 
# ============================================================ MODELOS
def make_multi() -> Pipeline:
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("lr", LinearRegression())])
 
 
def make_poly(degree: int) -> Pipeline:
    return Pipeline([("scaler", StandardScaler()),
                     ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
                     ("lr", LinearRegression())])
 
 
def metrics(y_true: pd.Series, predictions: np.ndarray) -> dict:
    return {"MAE (mpg)": mean_absolute_error(y_true, predictions),
            "RMSE (mpg)": float(np.sqrt(mean_squared_error(y_true, predictions))),
            "R²": r2_score(y_true, predictions)}
 
 
@st.cache_resource(show_spinner="Treinando os modelos...")
def train(df: pd.DataFrame):
    X, y = df[NUMERIC_COLS], df["mpg"]
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)
    X_tv, y_tv = pd.concat([X_train, X_val]), pd.concat([y_train, y_val])
 
    models = {
        "Baseline (média)": (DummyRegressor(strategy="mean"), ["weight"]),
        "Linear simples": (LinearRegression(), ["weight"]),
        "Linear múltipla": (make_multi(), MULTI_COLS),
        f"Polinomial grau {DEGREE}": (make_poly(DEGREE), ["weight"]),
    }
    results = []
    for name, (model, columns) in models.items():
        model.fit(X_tv[columns], y_tv)
        results.append({"Modelo": name, **metrics(y_test, model.predict(X_test[columns]))})
    return models, pd.DataFrame(results).set_index("Modelo"), (len(X_train), len(X_val), len(X_test)), X_tv, y_tv
 
 
# ============================================================ HEADER
st.markdown(
    f"""
    <div class="hero">
        <div class="hero-title">Previsão de consumo de combustível</div>
        <div class="hero-subtitle">Análise do dataset Auto MPG utilizando modelos de regressão e técnicas de Machine Learning.</div>
        <a class="hero-link" href="{REPO}" target="_blank">Código e relatório no GitHub</a>
    </div>
    """,
    unsafe_allow_html=True,
)
 
# ============================================================ DATASET
csv_path = find_csv()
df = read_csv(str(csv_path) if csv_path else None, None)
if df is None:
    st.warning("Não foi possível carregar o dataset automaticamente.")
    st.markdown("Faça o upload do arquivo `auto-mpg.csv` "
                "([Kaggle](https://www.kaggle.com/datasets/uciml/autompg-dataset)) para continuar.")
    uploaded_file = st.file_uploader("Selecionar dataset", type="csv")
    if uploaded_file is None:
        st.stop()
    df = read_csv(None, uploaded_file.getvalue())
 
models, test_table, sizes, X_tv, y_tv = train(df)
best_model = test_table["RMSE (mpg)"].idxmin()
 
# ============================================================ VISÃO GERAL
c1, c2, c3, c4 = st.columns(4)
c1.markdown(metric_card("Veículos analisados", f"{len(df)}", "registros"), unsafe_allow_html=True)
c2.markdown(metric_card("Consumo médio", f"{df.mpg.mean():.1f}", "mpg"), unsafe_allow_html=True)
c3.markdown(metric_card("Peso médio", f"{df.weight.mean():.0f}", "lb"), unsafe_allow_html=True)
c4.markdown(metric_card("Melhor modelo (teste)", best_model, "menor RMSE", "1.25rem"), unsafe_allow_html=True)
st.write("")
 
tab_prediction, tab_results, tab_about = st.tabs(["Previsão", "Resultados", "Sobre o estudo"])
 
# ============================================================ PREVISÃO
with tab_prediction:
    section("SIMULAÇÃO", "Estime o consumo de um veículo")
    col_input, col_chart = st.columns([0.85, 1.6], gap="large")
 
    with col_input:
        st.markdown(card("Características do veículo",
                         "Ajuste os parâmetros abaixo para gerar uma estimativa de consumo."),
                    unsafe_allow_html=True)
        weight_min, weight_max = int(X_tv.weight.min()), int(X_tv.weight.max())
        hp_min, hp_max = int(X_tv.horsepower.min()), int(X_tv.horsepower.max())
        year_min, year_max = int(X_tv.model_year.min()), int(X_tv.model_year.max())
 
        weight = st.slider("Peso (lb)", 1000, 6000, 3000, step=25)
        horsepower = st.slider("Potência (hp)", 40, 300, 100, step=1)
        model_year = st.slider("Ano do modelo (70 = 1970)", 65, 90, 76, step=1)
 
        outside = []
        if not weight_min <= weight <= weight_max:
            outside.append(f"peso fora de {weight_min}–{weight_max} lb")
        if not hp_min <= horsepower <= hp_max:
            outside.append(f"potência fora de {hp_min}–{hp_max} hp")
        if not year_min <= model_year <= year_max:
            outside.append(f"ano fora de {1900 + year_min}–{1900 + year_max}")
        if outside:
            st.warning("**Extrapolação:** " + "; ".join(outside) +
                       ". Os modelos só conhecem a faixa dos dados de treino (carros de 1970–1982); "
                       "a previsão pode não ser confiável.")
 
        x_simple = pd.DataFrame({"weight": [weight]})
        x_multi = pd.DataFrame({"weight": [weight], "horsepower": [horsepower], "model_year": [model_year]})
        predictions = {
            "Linear simples": float(models["Linear simples"][0].predict(x_simple)[0]),
            "Linear múltipla": float(models["Linear múltipla"][0].predict(x_multi)[0]),
            f"Polinomial grau {DEGREE}": float(models[f"Polinomial grau {DEGREE}"][0].predict(x_simple)[0]),
        }
        if min(predictions.values()) < float(df.mpg.min()):
            st.info(f"Alguma previsão ficou abaixo do menor consumo observado no dataset "
                    f"({df.mpg.min():.0f} mpg): sinal de extrapolação.")
 
    with col_chart:
        section("RESULTADO", "Consumo estimado")
        metric_cols = st.columns(3)
        for col, (name, prediction) in zip(metric_cols, predictions.items()):
            col.markdown(metric_card(name, f"{prediction:.1f} mpg", f"≈ {prediction * KMPL:.1f} km/L"),
                         unsafe_allow_html=True)
        st.write("")
        rmse_multiple = test_table.loc["Linear múltipla", "RMSE (mpg)"]
        st.markdown(card("Como interpretar",
                         "A regressão múltipla utiliza peso, potência e ano do modelo. Seu RMSE de teste foi "
                         f"aproximadamente <strong>{rmse_multiple:.1f} mpg</strong>.<br><br>"
                         "Essa métrica representa uma estimativa do erro típico do modelo e não deve ser "
                         "interpretada como uma medida exata para cada veículo."),
                    unsafe_allow_html=True)
 
    # ---- gráfico
    st.write("")
    section("VISUALIZAÇÃO", "Peso e consumo de combustível")
    grid = pd.DataFrame({"weight": np.linspace(max(1000, df.weight.min() - 200),
                                               min(6000, df.weight.max() + 200), 300)})
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#ffffff"); ax.set_facecolor("#ffffff")
    ax.scatter(df.weight, df.mpg, s=22, alpha=0.32, color="#64748b", edgecolors="none", label="Dados observados")
    ax.plot(grid.weight, models["Linear simples"][0].predict(grid), color="#172033", linewidth=2.4, label="Linear simples")
    ax.plot(grid.weight, models[f"Polinomial grau {DEGREE}"][0].predict(grid), color="#2563eb",
            linewidth=2.4, label=f"Polinomial grau {DEGREE}")
    ax.scatter([weight], [predictions["Linear múltipla"]], marker="*", s=280, color="#dc2626",
               edgecolors="white", linewidths=1, zorder=5, label="Seu veículo (múltipla)")
    ax.axvspan(grid.weight.min(), weight_min, color="#f59e0b", alpha=0.07)
    ax.axvspan(weight_max, grid.weight.max(), color="#f59e0b", alpha=0.07)
    ax.set_xlabel("Peso (lb)", fontsize=10, color="#475467")
    ax.set_ylabel("Consumo (mpg)", fontsize=10, color="#475467")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#d0d5dd"); ax.spines["bottom"].set_color("#d0d5dd")
    ax.tick_params(colors="#667085")
    ax.grid(axis="y", alpha=0.18, color="#94a3b8")
    ax.legend(frameon=False, loc="upper right")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("As curvas usam só o peso; a estrela usa também potência e ano (modelo múltipla). "
               "Áreas em laranja: fora da faixa de peso dos dados de treino.")
 
# ============================================================ RESULTADOS
with tab_results:
    section("AVALIAÇÃO", "Desempenho dos modelos")
    st.markdown(card("Protocolo",
                     "Os dados foram divididos em <strong>60% treino</strong>, <strong>20% validação</strong> e "
                     "<strong>20% teste</strong> (<code>random_state=42</code>). Modelos reajustados em treino + "
                     "validação e avaliados no teste uma única vez, como no notebook.<br><br>"
                     f"Treino: <strong>{sizes[0]}</strong> &nbsp;|&nbsp; Validação: <strong>{sizes[1]}</strong> "
                     f"&nbsp;|&nbsp; Teste: <strong>{sizes[2]}</strong>"),
                unsafe_allow_html=True)
    st.write("")
    st.dataframe(test_table.style.format({"MAE (mpg)": "{:.3f}", "RMSE (mpg)": "{:.3f}", "R²": "{:.3f}"}))
    st.write("")
    best_rmse = test_table.loc[best_model, "RMSE (mpg)"]
    r2_best = test_table.loc[best_model, "R²"]
    a, b = st.columns(2)
    a.markdown(metric_card("Melhor modelo", best_model, f"RMSE = {best_rmse:.3f} mpg", "1.35rem"), unsafe_allow_html=True)
    b.markdown(metric_card("R² do melhor modelo", f"{r2_best:.3f}", "desempenho no conjunto de teste"), unsafe_allow_html=True)
    st.write("")
    st.markdown(card("Principais conclusões",
                     "<p>A <strong>regressão linear múltipla</strong> (peso, potência e ano do modelo) apresentou "
                     "o menor erro entre os modelos avaliados.</p>"
                     "<p>A inclusão de <strong>model_year</strong> contribuiu mais para o desempenho do que "
                     "aumentar o grau do polinômio.</p>"
                     "<p>O polinomial de <strong>grau 2</strong> melhorou pouco em relação à reta; os graus 3 e 5 "
                     "não melhoraram na validação.</p>"
                     "<p>Os erros são maiores em veículos com consumo acima de 30 mpg, indicando variância "
                     "não constante.</p>"),
                unsafe_allow_html=True)
    st.caption("Os números coincidem com o relatório quando o dataset oficial do Kaggle é usado.")
 
# ============================================================ SOBRE
with tab_about:
    section("PROJETO", "Sobre o estudo")
    a, b = st.columns(2)
    a.markdown(card("Objetivo", "Este projeto utiliza o dataset <strong>Auto MPG</strong> para analisar o consumo "
                                "de combustível de diferentes veículos e comparar modelos de regressão."),
               unsafe_allow_html=True)
    b.markdown(card("Variável-alvo", "<strong>mpg</strong> — consumo de combustível em milhas por galão."),
               unsafe_allow_html=True)
    st.write("")
    cols = st.columns(3)
    for col, (title, var, desc) in zip(cols, [("Peso", "weight", "Peso do veículo em libras."),
                                              ("Potência", "horsepower", "Potência do motor em hp."),
                                              ("Ano", "model_year", "Ano do modelo (70 = 1970).")]):
        col.markdown(card(title, f"<code>{var}</code><br><br>{desc}"), unsafe_allow_html=True)
    st.write("")
    st.markdown(card("Modelos avaliados",
                     "<strong>Baseline:</strong> previsão baseada na média.<br><br>"
                     "<strong>Regressão linear simples:</strong> utiliza apenas o peso do veículo.<br><br>"
                     "<strong>Regressão linear múltipla:</strong> utiliza peso, potência e ano.<br><br>"
                     "<strong>Regressão polinomial:</strong> relação não linear entre peso e consumo (grau 2, "
                     "escolhido na validação)."),
                unsafe_allow_html=True)
    st.write("")
    st.markdown(card("Dados e limitações",
                     "Dados reais, 1970–1982, ciclo urbano e majoritariamente de carros americanos. O modelo "
                     "<strong>não vale para carros modernos</strong> (híbridos, elétricos, turbo) nem para "
                     "valores fora da faixa observada. Amostra pequena e colinearidade entre peso e potência "
                     "limitam a interpretação."),
                unsafe_allow_html=True)
 
# ============================================================ FOOTER
st.markdown(
    f"""
    <div class="footer">
        Auto MPG · Machine Learning · Regressão · Projeto acadêmico<br>
        Dados: Quinlan, R. (1993), <a href="https://archive.ics.uci.edu/dataset/9/auto+mpg" target="_blank">Auto MPG, UCI</a>
        (via <a href="https://www.kaggle.com/datasets/uciml/autompg-dataset" target="_blank">Kaggle</a>), licença CC BY 4.0.
        · <a href="{REPO}" target="_blank">Repositório</a>
    </div>
    """,
    unsafe_allow_html=True,
)
