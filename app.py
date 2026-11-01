import os
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import streamlit as st
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


st.set_page_config(
    page_title="Student Habits & Performance",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)


DATA_URL = (
    "https://raw.githubusercontent.com/arunk13/MSDA-Assignments/master/"
    "IS607Fall2015/Assignment3/student-mat.csv"
)
DATA_FILE = os.path.join("data", "student-mat.csv")

ABSENCE_FEATURES = [
    "age", "Medu", "Fedu", "traveltime", "studytime", "failures",
    "famrel", "freetime", "goout", "Dalc", "Walc", "health",
]

GRADE_FEATURES = ABSENCE_FEATURES + ["absences"]

FEATURE_NAMES = {
    "age": "Age",
    "Medu": "Mother's education",
    "Fedu": "Father's education",
    "traveltime": "Travel time",
    "studytime": "Study time",
    "failures": "Past failures",
    "famrel": "Family relationship",
    "freetime": "Free time",
    "goout": "Going out",
    "Dalc": "Workday alcohol use",
    "Walc": "Weekend alcohol use",
    "health": "Health",
    "absences": "Absences",
}


# A compact sun/moon button in the page header controls the dashboard theme.
if "dashboard_dark_mode" not in st.session_state:
    st.session_state["dashboard_dark_mode"] = False

spacer_col, theme_button_col = st.columns([0.92, 0.08])
with theme_button_col:
    current_dark_mode = st.session_state["dashboard_dark_mode"]
    theme_icon = "☀️" if current_dark_mode else "🌙"
    theme_help = "Switch to light mode" if current_dark_mode else "Switch to dark mode"
    if st.button(theme_icon, key="dashboard_theme_button", help=theme_help):
        st.session_state["dashboard_dark_mode"] = not current_dark_mode
        st.rerun()

dark_mode = st.session_state["dashboard_dark_mode"]


if dark_mode:
    theme = {
        "app_bg": "#0b1220",
        "surface": "#121c2b",
        "surface_alt": "#18263a",
        "text": "#e8eef8",
        "muted": "#a5b2c6",
        "border": "#29384e",
        "accent": "#8ab4ff",
        "grid": "#2b3b52",
        "table_head": "#1e2d43",
        "table_alt": "#172438",
        "plot_bg": "#121c2b",
        "hero_start": "#172554",
        "hero_end": "#263f73",
        "notice_bg": "#17263a",
    }
else:
    theme = {
        "app_bg": "#f4f7fb",
        "surface": "#ffffff",
        "surface_alt": "#f8fafc",
        "text": "#172033",
        "muted": "#596579",
        "border": "#e2e8f0",
        "accent": "#315fba",
        "grid": "#e4eaf2",
        "table_head": "#edf2f8",
        "table_alt": "#f8fafc",
        "plot_bg": "#ffffff",
        "hero_start": "#172554",
        "hero_end": "#334e8c",
        "notice_bg": "#eef4ff",
    }


st.markdown(
    f"""
    <style>
    :root {{ color-scheme: {'dark' if dark_mode else 'light'}; }}
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    .main {{
        background: {theme['app_bg']} !important;
        color: {theme['text']} !important;
    }}
    [data-testid="stHeader"] {{ background: transparent !important; }}
    [data-testid="stSidebar"] {{
        background: {theme['surface']} !important;
        border-right: 1px solid {theme['border']};
    }}
    [data-testid="stSidebar"] * {{ color: {theme['text']}; }}
    .block-container {{
        max-width: 1320px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }}
    .stMarkdown, .stMarkdown p, .stMarkdown li, .stCaption,
    [data-testid="stCaptionContainer"], label,
    [data-testid="stWidgetLabel"] p {{
        color: {theme['text']} !important;
    }}
    h1, h2, h3, h4, h5, h6,
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{
        color: {theme['text']} !important;
        letter-spacing: -0.02em;
    }}
    .hero {{
        background: linear-gradient(115deg, {theme['hero_start']}, {theme['hero_end']});
        padding: 30px 32px;
        border-radius: 20px;
        color: #ffffff !important;
        margin-bottom: 18px;
        box-shadow: 0 10px 26px rgba(15, 23, 42, 0.12);
    }}
    .hero h1 {{
        color: #ffffff !important;
        margin: 0 0 8px 0;
        font-size: clamp(1.7rem, 3vw, 2.35rem);
        line-height: 1.2;
    }}
    .hero p {{
        color: #e6edff !important;
        margin: 0;
        font-size: 1rem;
        line-height: 1.55;
    }}
    .dataset-note {{
        color: {theme['muted']} !important;
        margin: 0.25rem 0 1rem 0;
        font-size: 0.92rem;
    }}
    [data-testid="stMetric"] {{
        background: {theme['surface']} !important;
        border: 1px solid {theme['border']} !important;
        border-radius: 14px !important;
        padding: 16px 18px !important;
        min-height: 112px;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.035);
    }}
    [data-testid="stMetricLabel"] p {{
        color: {theme['muted']} !important;
        font-size: 0.9rem !important;
    }}
    [data-testid="stMetricValue"] div {{
        color: {theme['text']} !important;
        font-size: clamp(1.65rem, 3vw, 2.1rem) !important;
        font-weight: 700 !important;
    }}
    [data-testid="stMetricDelta"] {{ color: {theme['muted']} !important; }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0.35rem;
        border-bottom: 1px solid {theme['border']};
    }}
    .stTabs [data-baseweb="tab"] {{
        color: {theme['muted']} !important;
        background: {theme['surface']} !important;
        border-radius: 9px 9px 0 0;
        padding: 0.65rem 0.9rem;
    }}
    .stTabs [aria-selected="true"] {{
        color: {theme['accent']} !important;
        border-bottom: 3px solid {theme['accent']} !important;
    }}
    [data-testid="stAlert"] {{
        background: {theme['notice_bg']} !important;
        border: 1px solid {theme['border']} !important;
        border-radius: 12px !important;
    }}
    [data-testid="stAlert"] p {{ color: {theme['text']} !important; }}
    .table-wrap {{
        width: 100%;
        overflow-x: auto;
        border: 1px solid {theme['border']};
        border-radius: 12px;
        margin: 0.35rem 0 1rem 0;
        background: {theme['surface']};
    }}
    table.data-table {{
        width: 100%;
        border-collapse: collapse;
        color: {theme['text']};
        background: {theme['surface']};
        font-size: 0.94rem;
        min-width: 420px;
    }}
    .data-table th {{
        background: {theme['table_head']};
        color: {theme['text']};
        font-weight: 650;
        text-align: left;
        padding: 0.78rem 0.9rem;
        border-bottom: 1px solid {theme['border']};
        white-space: nowrap;
    }}
    .data-table td {{
        color: {theme['text']};
        padding: 0.72rem 0.9rem;
        border-bottom: 1px solid {theme['border']};
        white-space: nowrap;
    }}
    .data-table tbody tr:nth-child(even) {{ background: {theme['table_alt']}; }}
    .data-table tbody tr:last-child td {{ border-bottom: none; }}
    [data-testid="stExpander"] {{
        border: 1px solid {theme['border']} !important;
        border-radius: 12px !important;
        background: {theme['surface']} !important;
    }}
    div[data-testid="stButton"] button {{
        width: 46px;
        height: 46px;
        min-height: 46px;
        padding: 0;
        border-radius: 50%;
        border: 1px solid {theme['border']} !important;
        background: {theme['surface']} !important;
        color: {theme['text']} !important;
        font-size: 1.45rem !important;
        box-shadow: 0 3px 10px rgba(15, 23, 42, 0.10);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }}
    div[data-testid="stButton"] button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 5px 14px rgba(15, 23, 42, 0.16);
        border-color: {theme['accent']} !important;
    }}
    div[data-testid="stButton"] button p {{
        color: {theme['text']} !important;
        font-size: 1.45rem !important;
        line-height: 1 !important;
    }}
    hr {{ border-color: {theme['border']} !important; }}
    @media (max-width: 700px) {{
        .block-container {{ padding-left: 1rem; padding-right: 1rem; }}
        .hero {{ padding: 23px 21px; }}
        .stTabs [data-baseweb="tab"] {{ padding: 0.55rem 0.55rem; }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_data():
    if os.path.exists(DATA_FILE):
        data = pd.read_csv(DATA_FILE, sep=";")
    elif os.path.exists("student-mat.csv"):
        data = pd.read_csv("student-mat.csv", sep=";")
    else:
        data = pd.read_csv(DATA_URL, sep=";")

    required_columns = ABSENCE_FEATURES + ["absences", "G3"]
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise ValueError("The dataset is missing these columns: " + ", ".join(missing_columns))

    data = data.dropna(subset=required_columns).copy()
    data[required_columns] = data[required_columns].apply(pd.to_numeric)
    data["pass_fail"] = np.where(data["G3"] >= 10, "Pass", "Fail")
    return data


@st.cache_data(show_spinner=False)
def make_count_models(data):
    x = data[ABSENCE_FEATURES]
    y = data["absences"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    x_train_sm = sm.add_constant(x_train, has_constant="add")
    x_test_sm = sm.add_constant(x_test, has_constant="add")

    poisson_model = sm.GLM(
        y_train,
        x_train_sm,
        family=sm.families.Poisson(),
    ).fit()
    poisson_predictions = poisson_model.predict(x_test_sm)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        negative_binomial_model = sm.NegativeBinomial(y_train, x_train_sm)
        negative_binomial_result = negative_binomial_model.fit(disp=False, maxiter=200)
    negative_binomial_predictions = negative_binomial_result.predict(x_test_sm)

    dispersion = poisson_model.pearson_chi2 / poisson_model.df_resid
    results = pd.DataFrame({
        "Model": ["Poisson", "Negative Binomial"],
        "AIC": [poisson_model.aic, negative_binomial_result.aic],
        "Test RMSE": [
            np.sqrt(mean_squared_error(y_test, poisson_predictions)),
            np.sqrt(mean_squared_error(y_test, negative_binomial_predictions)),
        ],
    })

    return {
        "mean": y.mean(),
        "variance": y.var(),
        "dispersion": dispersion,
        "results": results,
        "actual": y_test,
        "poisson_predictions": poisson_predictions,
        "nb_predictions": negative_binomial_predictions,
        "nb_converged": negative_binomial_result.mle_retvals.get("converged", True),
    }


@st.cache_data(show_spinner=False)
def make_grade_model(data):
    x = data[GRADE_FEATURES]
    y = data["G3"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    residuals = y_test - predictions

    coefficients = pd.DataFrame({
        "Feature": [FEATURE_NAMES.get(name, name) for name in GRADE_FEATURES],
        "Coefficient": model.coef_,
    })
    coefficients["Absolute value"] = coefficients["Coefficient"].abs()
    coefficients = coefficients.sort_values("Absolute value", ascending=False)

    return {
        "r2": r2_score(y_test, predictions),
        "rmse": np.sqrt(mean_squared_error(y_test, predictions)),
        "actual": y_test,
        "predictions": predictions,
        "residuals": residuals,
        "coefficients": coefficients,
    }


@st.cache_data(show_spinner=False)
def make_classification_models(data):
    x = data[GRADE_FEATURES]
    y = (data["G3"] >= 10).astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    lda_model = LinearDiscriminantAnalysis()
    qda_model = QuadraticDiscriminantAnalysis()
    lda_model.fit(x_train_scaled, y_train)
    qda_model.fit(x_train_scaled, y_train)

    lda_predictions = lda_model.predict(x_test_scaled)
    qda_predictions = qda_model.predict(x_test_scaled)
    lda_probabilities = lda_model.predict_proba(x_test_scaled)[:, 1]
    qda_probabilities = qda_model.predict_proba(x_test_scaled)[:, 1]

    results = pd.DataFrame({
        "Model": ["LDA", "QDA"],
        "Accuracy": [
            accuracy_score(y_test, lda_predictions),
            accuracy_score(y_test, qda_predictions),
        ],
        "ROC-AUC": [
            roc_auc_score(y_test, lda_probabilities),
            roc_auc_score(y_test, qda_probabilities),
        ],
    })

    return {
        "results": results,
        "actual": y_test,
        "lda_predictions": lda_predictions,
        "qda_predictions": qda_predictions,
        "lda_matrix": confusion_matrix(y_test, lda_predictions, labels=[0, 1]),
        "qda_matrix": confusion_matrix(y_test, qda_predictions, labels=[0, 1]),
    }


def render_table(dataframe, formats=None):
    """Render a compact HTML table with explicit colors for both themes."""
    display_df = dataframe.copy()
    for column, number_format in (formats or {}).items():
        if column in display_df.columns:
            display_df[column] = display_df[column].map(
                lambda value: number_format.format(value) if pd.notna(value) else "—"
            )
    html = display_df.to_html(index=False, escape=True, classes="data-table", border=0)
    st.markdown(f'<div class="table-wrap">{html}</div>', unsafe_allow_html=True)


def style_plot(figure, dark):
    plot_bg = "#121c2b" if dark else "#ffffff"
    text_color = "#e8eef8" if dark else "#172033"
    grid_color = "#2b3b52" if dark else "#e4eaf2"
    figure.update_layout(
        template="plotly_dark" if dark else "plotly_white",
        paper_bgcolor=plot_bg,
        plot_bgcolor=plot_bg,
        font={"color": text_color, "size": 13},
        title={"font": {"color": text_color, "size": 17}, "x": 0.02, "xanchor": "left"},
        legend={"font": {"color": text_color}},
        hoverlabel={"font": {"color": text_color}},
        margin={"l": 42, "r": 22, "t": 65, "b": 42},
        height=360,
    )
    figure.update_xaxes(
        title_font={"color": text_color},
        tickfont={"color": text_color},
        gridcolor=grid_color,
        zerolinecolor=grid_color,
    )
    figure.update_yaxes(
        title_font={"color": text_color},
        tickfont={"color": text_color},
        gridcolor=grid_color,
        zerolinecolor=grid_color,
    )
    return figure


def show_plot(figure, dark):
    st.plotly_chart(
        style_plot(figure, dark),
        use_container_width=True,
        theme=None,
        config={"responsive": True, "displaylogo": False},
    )


try:
    with st.spinner("Loading the student dataset and training the models..."):
        df = load_data()
        count_results = make_count_models(df)
        grade_results = make_grade_model(df)
        class_results = make_classification_models(df)
except Exception as error:
    st.error("The project could not load the dataset or train the models.")
    st.write(str(error))
    st.info(
        "Check your internet connection, or download student-mat.csv from the UCI "
        "Student Performance page and place it inside the project's data folder."
    )
    st.markdown(
        "[Open the UCI Student Performance dataset]"
        "(https://archive.ics.uci.edu/dataset/320/student+performance)"
    )
    st.stop()


st.markdown(
    """
    <div class="hero">
        <h1>🎓 Student Habits &amp; Performance</h1>
        <p>Predicting school absences, final grades, and pass/fail results using lifestyle and study habits.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="dataset-note">UCI Student Performance dataset · Mathematics students · '
    '80/20 train-test split</p>',
    unsafe_allow_html=True,
)

metric1, metric2, metric3, metric4 = st.columns(4)
metric1.metric("Students", f"{len(df):,}")
metric2.metric("Average absences", f"{df['absences'].mean():.1f}")
metric3.metric("Average final grade", f"{df['G3'].mean():.1f} / 20")
metric4.metric("Pass rate", f"{(df['G3'] >= 10).mean() * 100:.1f}%")


overview_tab, absence_tab, grade_tab, classification_tab = st.tabs([
    "Overview & EDA",
    "Absence Prediction",
    "Final Grade Prediction",
    "Pass / Fail Classification",
])


with overview_tab:
    st.subheader("Dataset overview")
    st.write(
        "The project uses students' study time, past failures, family education, "
        "travel time, free time, social habits, alcohol use, health, and absences."
    )

    chart1, chart2 = st.columns(2)
    with chart1:
        absence_chart = px.histogram(
            df,
            x="absences",
            nbins=20,
            title="Distribution of student absences",
            labels={"absences": "Number of absences", "count": "Students"},
            color_discrete_sequence=["#4568a8"],
        )
        absence_chart.update_layout(bargap=0.08)
        show_plot(absence_chart, dark_mode)

    with chart2:
        grade_chart = px.histogram(
            df,
            x="G3",
            nbins=20,
            title="Distribution of final grades",
            labels={"G3": "Final grade (0–20)", "count": "Students"},
            color_discrete_sequence=["#16a085"],
        )
        grade_chart.update_layout(bargap=0.08)
        show_plot(grade_chart, dark_mode)

    eda1, eda2, eda3 = st.columns(3)
    eda1.metric("Absence mean", f"{count_results['mean']:.2f}")
    eda2.metric("Absence variance", f"{count_results['variance']:.2f}")
    eda3.metric("Zero-absence students", f"{(df['absences'] == 0).mean() * 100:.1f}%")

    if count_results["variance"] > count_results["mean"]:
        st.info(
            "The variance of absences is higher than the mean. This is an early sign "
            "of overdispersion, so Negative Binomial regression is worth testing."
        )
    else:
        st.info(
            "The variance of absences is not higher than the mean in this dataset. "
            "The model-level dispersion check below gives another way to assess Poisson fit."
        )

    with st.expander("View 10 sample student records"):
        sample = df[["age", "studytime", "failures", "goout", "absences", "G3", "pass_fail"]].head(10)
        render_table(sample)


with absence_tab:
    st.subheader("Predicting the number of absences")
    st.write(
        "Poisson regression and Negative Binomial regression are trained on the same "
        "80% training split and tested on the remaining 20%."
    )

    a, b, c = st.columns(3)
    a.metric("Mean absences", f"{count_results['mean']:.2f}")
    b.metric("Variance", f"{count_results['variance']:.2f}")
    c.metric("Poisson dispersion ratio", f"{count_results['dispersion']:.2f}")

    render_table(
        count_results["results"],
        {"AIC": "{:.2f}", "Test RMSE": "{:.2f}"},
    )

    if count_results["dispersion"] > 1.5:
        st.success(
            "The Poisson dispersion ratio is noticeably above 1. This suggests "
            "overdispersion, which Negative Binomial regression is designed to handle."
        )
    else:
        st.info(
            "The Poisson dispersion ratio is not very high. Compare AIC and test RMSE "
            "before deciding which count model performed better."
        )

    if not count_results["nb_converged"]:
        st.warning(
            "The Negative Binomial optimiser did not report full convergence. "
            "Treat its metrics with care."
        )

    comparison = go.Figure()
    comparison.add_trace(go.Scatter(
        x=count_results["actual"],
        y=count_results["poisson_predictions"],
        mode="markers",
        name="Poisson",
        marker={"color": "#4568a8", "opacity": 0.68, "size": 8},
    ))
    comparison.add_trace(go.Scatter(
        x=count_results["actual"],
        y=count_results["nb_predictions"],
        mode="markers",
        name="Negative Binomial",
        marker={"color": "#16a085", "opacity": 0.68, "size": 8},
    ))
    comparison.update_layout(
        title="Actual vs predicted absences on the test set",
        xaxis_title="Actual absences",
        yaxis_title="Predicted absences",
    )
    show_plot(comparison, dark_mode)
    st.caption(
        "Lower AIC and lower test RMSE are generally better. AIC compares fit while "
        "accounting for model complexity; RMSE measures test prediction error."
    )


with grade_tab:
    st.subheader("Predicting the final grade")
    st.write(
        "Linear Regression estimates the final Mathematics grade (G3) using lifestyle, "
        "family, study, and attendance variables. Earlier grades G1 and G2 are not used."
    )

    a, b = st.columns(2)
    a.metric("Test R²", f"{grade_results['r2']:.3f}")
    b.metric("Test RMSE", f"{grade_results['rmse']:.2f} grade points")

    grade_chart = px.scatter(
        x=grade_results["actual"],
        y=grade_results["predictions"],
        labels={"x": "Actual final grade", "y": "Predicted final grade"},
        title="Actual vs predicted final grades",
        color_discrete_sequence=["#4568a8"],
    )
    grade_chart.add_trace(go.Scatter(
        x=[0, 20],
        y=[0, 20],
        mode="lines",
        name="Perfect prediction",
        line={"color": "#e67e22", "dash": "dash"},
    ))
    show_plot(grade_chart, dark_mode)

    residual_chart = px.scatter(
        x=grade_results["predictions"],
        y=grade_results["residuals"],
        labels={"x": "Predicted grade", "y": "Residual (actual - predicted)"},
        title="Residual plot",
        color_discrete_sequence=["#16a085"],
    )
    residual_chart.add_hline(y=0, line_dash="dash", line_color="#e67e22")
    show_plot(residual_chart, dark_mode)

    st.subheader("Linear Regression coefficients")
    st.write(
        "These values show the estimated change in predicted grade for a one-unit "
        "increase in each feature, while the other included features stay fixed. "
        "They do not prove cause and effect."
    )
    render_table(
        grade_results["coefficients"][ ["Feature", "Coefficient"] ],
        {"Coefficient": "{:.3f}"},
    )


with classification_tab:
    st.subheader("Predicting pass or fail")
    st.write(
        "A student is labelled Pass when G3 is 10 or higher. Otherwise, the label is "
        "Fail. LDA and QDA are compared on the same stratified test split."
    )

    render_table(
        class_results["results"],
        {"Accuracy": "{:.3f}", "ROC-AUC": "{:.3f}"},
    )

    matrix1, matrix2 = st.columns(2)
    with matrix1:
        lda_figure = px.imshow(
            class_results["lda_matrix"],
            text_auto=True,
            x=["Predicted Fail", "Predicted Pass"],
            y=["Actual Fail", "Actual Pass"],
            color_continuous_scale="Blues",
            title="LDA confusion matrix",
        )
        lda_figure.update_layout(coloraxis_showscale=False)
        show_plot(lda_figure, dark_mode)

    with matrix2:
        qda_figure = px.imshow(
            class_results["qda_matrix"],
            text_auto=True,
            x=["Predicted Fail", "Predicted Pass"],
            y=["Actual Fail", "Actual Pass"],
            color_continuous_scale="Greens",
            title="QDA confusion matrix",
        )
        qda_figure.update_layout(coloraxis_showscale=False)
        show_plot(qda_figure, dark_mode)

    st.caption(
        "Accuracy is the fraction of correct predictions. ROC-AUC measures how well "
        "the model ranks pass cases above fail cases across different probability "
        "thresholds. A higher value is better."
    )


st.divider()
st.subheader("Project conclusion")
st.write(
    "Use the model comparison tables to select the best model for each task. "
    "For absences, compare Poisson and Negative Binomial using dispersion, AIC, "
    "and test RMSE. For final grade, use test R² and RMSE. For pass/fail, compare "
    "LDA and QDA using accuracy, ROC-AUC, and the confusion matrices."
)
st.warning(
    "This is a classroom project using a small historical dataset from two Portuguese "
    "schools. Its predictions should not be used to make real decisions about "
    "individual students. Patterns in this dataset do not prove that a lifestyle "
    "habit causes a particular grade or absence level."
)
st.caption(
    "Dataset: P. Cortez (2008), UCI Student Performance. See the README for the source and setup steps."
)
