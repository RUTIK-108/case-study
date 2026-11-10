
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from scipy import stats
from datetime import date

# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="CycleIQ | Bike Demand Intelligence",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Colour palette
BG = "#F7F7F4"
WHITE = "#FFFFFF"
TEXT = "#292929"
MUTED = "#686868"
RED = "#B7463A"
GREEN = "#52796F"
BORDER = "#E5E5DF"

API_URL = "http://127.0.0.1:8000"


# =========================================================
# 2. LIGHT THEME
# =========================================================

st.markdown("""
<style>

html, body, .stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background-color: #F7F7F4 !important;
    color: #292929 !important;
}

.main .block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

header[data-testid="stHeader"] {
    background-color: #F7F7F4 !important;
    border-bottom: 1px solid #E5E5DF !important;
}

[data-testid="stToolbar"] {
    background-color: #F7F7F4 !important;
}

[data-testid="stHeader"] button {
    color: #292929 !important;
}

[data-testid="stDecoration"] {
    background-color: #B7463A !important;
    background-image: none !important;
    height: 3px !important;
}

#MainMenu {
    visibility: hidden;
}

h1, h2, h3, h4, h5, h6,
p, label, li,
.stMarkdown, .stText {
    color: #292929 !important;
}

[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p {
    color: #686868 !important;
}

/* Sidebar */

section[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div {
    background-color: #F0F0EB !important;
}

section[data-testid="stSidebar"] {
    border-right: 1px solid #E5E5DF !important;
}

section[data-testid="stSidebar"] * {
    color: #292929 !important;
}

section[data-testid="stSidebar"] h1 {
    color: #B7463A !important;
}

/* Metrics */

div[data-testid="stMetric"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E5E5DF !important;
    border-radius: 8px !important;
    padding: 20px !important;
}

div[data-testid="stMetricLabel"],
div[data-testid="stMetricLabel"] p,
div[data-testid="stMetricValue"],
div[data-testid="stMetricValue"] div,
div[data-testid="stMetricValue"] span {
    color: #292929 !important;
    opacity: 1 !important;
}

/* Alerts */

div[data-testid="stAlert"],
div[data-testid="stAlert"] > div {
    background: #FFFFFF !important;
    background-color: #FFFFFF !important;
    color: #292929 !important;
    border: 1px solid #E5E5DF !important;
    border-radius: 7px !important;
    box-shadow: none !important;
}

div[data-testid="stAlert"] p,
div[data-testid="stAlert"] span,
div[data-testid="stAlert"] div {
    color: #292929 !important;
    opacity: 1 !important;
}

/* Buttons */

.stButton button,
.stDownloadButton button {
    background-color: #B7463A !important;
    color: #FFFFFF !important;
    border: 1px solid #B7463A !important;
    border-radius: 5px !important;
    font-weight: 600 !important;
}

.stButton button *,
.stDownloadButton button * {
    color: #FFFFFF !important;
}

.stButton button:hover,
.stDownloadButton button:hover {
    background-color: #96372E !important;
    border-color: #96372E !important;
}

/* Inputs */

div[data-baseweb="select"] div[class*="singleValue"] {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    opacity: 1 !important;
}

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
div[data-baseweb="textarea"] > div {
    background-color: #FFFFFF !important;
    border-color: #DADAD3 !important;
}

input, textarea {
    color: #292929 !important;
}

/* Tables */

div[data-testid="stDataFrame"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E5E5DF !important;
    border-radius: 7px !important;
}

/* Expanders */

div[data-testid="stExpander"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E5E5DF !important;
    border-radius: 7px !important;
}

/* Dividers */

hr {
    border-color: #E5E5DF !important;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. CHART HELPER
# =========================================================

def create_chart(figsize=(9, 4)):

    fig, ax = plt.subplots(figsize=figsize)

    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(WHITE)

    ax.tick_params(axis="both", colors=TEXT, labelsize=9)

    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["bottom"].set_color(BORDER)
    ax.spines["left"].set_color(BORDER)

    ax.grid(axis="y", color=BORDER, linewidth=0.8)
    ax.set_axisbelow(True)

    return fig, ax


# =========================================================
# 4. LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    data = pd.read_csv("data/hour.csv")

    data["dteday"] = pd.to_datetime(data["dteday"])

    data["datetime"] = (
        data["dteday"] +
        pd.to_timedelta(data["hr"], unit="h")
    )

    data["year"] = data["yr"].map({
        0: 2011,
        1: 2012
    })

    return data


try:
    df = load_data()

except Exception as error:
    st.error(f"Unable to load dataset: {error}")
    st.stop()


# =========================================================
# 5. COMMON METRICS
# =========================================================

total_rentals = int(df["cnt"].sum())
average_demand = df["cnt"].mean()
maximum_demand = int(df["cnt"].max())

hourly_demand = df.groupby("hr")["cnt"].mean()

peak_hour = int(hourly_demand.idxmax())
peak_demand = float(hourly_demand.max())

lowest_hour = int(hourly_demand.idxmin())
lowest_demand = float(hourly_demand.min())

workingday_demand = df.groupby("workingday")["cnt"].mean()
seasonal_demand = df.groupby("season")["cnt"].mean()
yearly_demand = df.groupby("year")["cnt"].mean()

growth = None

if 2011 in yearly_demand.index and 2012 in yearly_demand.index:

    growth = (
        (yearly_demand.loc[2012] - yearly_demand.loc[2011])
        / yearly_demand.loc[2011]
    ) * 100


# =========================================================
# 6. SIDEBAR
# =========================================================

with st.sidebar:

    st.title("CYCLEIQ")
    st.caption("BIKE DEMAND INTELLIGENCE")

    st.divider()

    st.markdown("**NAVIGATION**")

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Demand Analytics",
            "Weather Analysis",
            "Model Performance",
            "Demand Prediction",
            "Business Recommendations",
            "Ask Operations Assistant"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("**PROJECT DETAILS**")

    st.caption("DATA SOURCE")
    st.write("Bike Sharing Dataset")

    st.caption("SELECTED MODEL")
    st.write("XGBoost Regressor")

    st.caption("TECHNOLOGY")
    st.write("Python · Streamlit · FastAPI")

    st.divider()

    st.caption("Bike Sharing Demand Analysis")
    st.caption("Historical data: 2011–2012")


# =========================================================
# PAGE 1: OVERVIEW
# =========================================================

if page == "Overview":

    st.title("Bike Sharing Demand")

    st.markdown(
        "Understand rental behaviour and identify patterns "
        "to support better fleet planning."
    )

    st.markdown("### Business Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Rentals", f"{total_rentals:,}")
    c2.metric("Average Hourly Demand", f"{average_demand:.0f}")
    c3.metric("Maximum Hourly Demand", maximum_demand)
    c4.metric("Model R² Score", "0.9037")

    st.markdown("### Demand Trends")

    left, right = st.columns([1.5, 1])

    with left:

        st.markdown("**Daily Rental Trend**")

        daily = (
            df.groupby("dteday")["cnt"]
            .sum()
            .reset_index()
        )

        fig, ax = create_chart((10, 4))

        ax.plot(
            daily["dteday"],
            daily["cnt"],
            color=RED,
            linewidth=1.4
        )

        ax.set_xlabel("Date")
        ax.set_ylabel("Total Bike Rentals")

        fig.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    with right:

        st.markdown("**Average Demand by Hour**")

        fig, ax = create_chart((8, 4))

        ax.bar(
            hourly_demand.index,
            hourly_demand.values,
            color=RED,
            width=0.7
        )

        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Average Rentals")

        fig.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    st.markdown("### Dataset Preview")

    st.dataframe(
        df.head(10),
        width="stretch",
        hide_index=True
    )


# =========================================================
# PAGE 2: DEMAND ANALYTICS
# =========================================================

elif page == "Demand Analytics":

    st.title("Demand Analytics")

    st.markdown(
        "Explore how time, working days and seasonal patterns "
        "influence bike rental demand."
    )

    st.divider()

    selected_year = st.selectbox(
        "Select Year",
        ["All", 2011, 2012]
    )

    filtered = df.copy()

    if selected_year != "All":
        filtered = filtered[filtered["year"] == selected_year]

    left, right = st.columns(2)

    with left:

        st.markdown("**Hourly Demand**")

        hourly = filtered.groupby("hr")["cnt"].mean()

        fig, ax = create_chart()

        ax.plot(
            hourly.index,
            hourly.values,
            color=RED,
            marker="o",
            markersize=4,
            linewidth=2
        )

        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Average Rentals")

        fig.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    with right:

        st.markdown("**Working Day vs Non-working Day**")

        working = filtered.groupby("workingday")["cnt"].mean()

        working.index = [
            "Non-working Day" if x == 0 else "Working Day"
            for x in working.index
        ]

        fig, ax = create_chart()

        ax.bar(
            working.index,
            working.values,
            color=RED,
            width=0.55
        )

        ax.set_ylabel("Average Rentals")

        fig.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    st.divider()

    left, right = st.columns(2)

    with left:

        st.markdown("**Monthly Demand**")

        monthly = filtered.groupby("mnth")["cnt"].mean()

        fig, ax = create_chart()

        ax.plot(
            monthly.index,
            monthly.values,
            color=GREEN,
            marker="o",
            linewidth=2
        )

        ax.set_xlabel("Month")
        ax.set_ylabel("Average Rentals")
        ax.set_xticks(range(1, 13))

        fig.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    with right:

        st.markdown("**Temperature vs Demand**")

        fig, ax = create_chart()

        ax.scatter(
            filtered["temp"],
            filtered["cnt"],
            color=RED,
            alpha=0.3,
            s=12
        )

        ax.set_xlabel("Normalized Temperature")
        ax.set_ylabel("Bike Rentals")

        fig.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close(fig)


# =========================================================
# PAGE 3: WEATHER ANALYSIS
# =========================================================

elif page == "Weather Analysis":

    st.title("Weather Analysis")

    st.markdown(
        "Analyse the relationship between weather conditions "
        "and bike rental demand."
    )

    st.divider()

    weather_names = {
        1: "Clear",
        2: "Mist / Cloudy",
        3: "Light Rain / Snow",
        4: "Heavy Rain / Snow"
    }

    weather_data = (
        df.groupby("weathersit")["cnt"]
        .agg(["mean", "count"])
        .reset_index()
    )

    weather_data["Weather"] = weather_data["weathersit"].map(
        weather_names
    )

    left, right = st.columns(2)

    with left:

        st.markdown("**Average Demand by Weather**")

        fig, ax = create_chart()

        ax.bar(
            weather_data["Weather"],
            weather_data["mean"],
            color=RED,
            width=0.6
        )

        ax.set_ylabel("Average Rentals")
        ax.tick_params(axis="x", rotation=15)

        fig.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    with right:

        st.markdown("**Records by Weather**")

        fig, ax = create_chart()

        ax.bar(
            weather_data["Weather"],
            weather_data["count"],
            color=GREEN,
            width=0.6
        )

        ax.set_ylabel("Number of Records")
        ax.tick_params(axis="x", rotation=15)

        fig.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    st.markdown("### Weather Summary")

    display_weather = weather_data[
        ["Weather", "mean", "count"]
    ].rename(columns={
        "mean": "Average Rentals",
        "count": "Number of Records"
    })

    st.dataframe(
        display_weather.round(2),
        width="stretch",
        hide_index=True
    )


# =========================================================
# PAGE 4: MODEL PERFORMANCE
# =========================================================

elif page == "Model Performance":

    st.title("Model Performance")

    st.markdown(
        "Comparison of machine learning models and evaluation "
        "of the final selected XGBoost model."
    )

    st.divider()

    c1, c2, c3 = st.columns(3)

    c1.metric("Test MAE", "45.21")
    c2.metric("Test RMSE", "68.44")
    c3.metric("Test R²", "0.9037")

    st.caption(
        "MAE and RMSE are measured in bikes per hour. "
        "The test dataset contains 3,476 observations."
    )

    st.divider()

    st.markdown("### Model Comparison")

    comparison = pd.DataFrame({
        "Model": [
            "XGBoost",
            "Random Forest",
            "Hist Gradient Boosting",
            "Linear Regression"
        ],
        "Cross Validation MAE": [
            53.355,
            56.176,
            56.294,
            96.400
        ]
    }).sort_values("Cross Validation MAE")

    fig, ax = create_chart((10, 4))

    ax.barh(
        comparison["Model"],
        comparison["Cross Validation MAE"],
        color=RED,
        height=0.55
    )

    ax.set_xlabel("Cross Validation MAE (Lower is Better)")
    ax.invert_yaxis()

    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    st.dataframe(
        comparison.round(2),
        width="stretch",
        hide_index=True
    )

    st.divider()

    st.markdown("### Feature Importance")

    importance = pd.DataFrame({
        "Feature": [
            "is_peak_hour",
            "hour_cos",
            "hour_sin",
            "yr",
            "workingday",
            "temp",
            "atemp",
            "is_weekend",
            "season",
            "weathersit"
        ],
        "Importance": [
            0.228458,
            0.179413,
            0.127051,
            0.105889,
            0.090260,
            0.051351,
            0.050250,
            0.039850,
            0.028839,
            0.025282
        ]
    }).sort_values("Importance")

    fig, ax = create_chart((10, 5))

    ax.barh(
        importance["Feature"],
        importance["Importance"],
        color=GREEN,
        height=0.6
    )

    ax.set_xlabel("Feature Importance")

    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    st.divider()

    st.markdown("### Key Findings")

    st.markdown("""
    1. XGBoost achieved the lowest cross-validation MAE among
       the evaluated models.

    2. Peak-hour and time-related features have high importance.

    3. The final model achieved an R² score of 0.9037.

    4. The test MAE of 45.21 indicates an average absolute
       prediction error of approximately 45 bikes per hour.
    """)


# =========================================================
# PAGE 5: DEMAND PREDICTION
# =========================================================

elif page == "Demand Prediction":

    st.title("Bike Demand Prediction")

    st.markdown(
        "Enter date, time and weather conditions to estimate "
        "hourly bike rental demand."
    )

    st.divider()

    st.markdown("### Prediction Inputs")

    left, right = st.columns(2)

    with left:

        selected_date = st.date_input(
            "Date",
            value=date(2012, 7, 15),
            min_value=date(2011, 1, 1),
            max_value=date(2012, 12, 31)
        )

        selected_hour = st.slider(
            "Hour of Day", 0, 23, 17
        )

        holiday = st.selectbox(
            "Is it a holiday?",
            [0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes"
        )

        workingday = st.selectbox(
            "Is it a working day?",
            [0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes"
        )

    with right:

        weather = st.selectbox(
            "Weather Condition",
            [1, 2, 3, 4],
            format_func=lambda x: {
                1: "Clear / Partly Cloudy",
                2: "Mist / Cloudy",
                3: "Light Rain / Snow",
                4: "Heavy Rain / Snow"
            }[x]
        )

        temperature = st.slider(
            "Temperature", 0.0, 1.0, 0.7, 0.01
        )

        feels_like = st.slider(
            "Feels-like Temperature", 0.0, 1.0, 0.65, 0.01
        )

        humidity = st.slider(
            "Humidity", 0.0, 1.0, 0.5, 0.01
        )

        windspeed = st.slider(
            "Windspeed", 0.0, 1.0, 0.2, 0.01
        )

    month = selected_date.month

    if month in [1, 2, 3]:
        season = 1
    elif month in [4, 5, 6]:
        season = 2
    elif month in [7, 8, 9]:
        season = 3
    else:
        season = 4

    weekday = (selected_date.weekday() + 1) % 7

    st.caption(
        "Temperature, feels-like temperature, humidity and "
        "windspeed use normalized values between 0 and 1."
    )

    st.divider()

    if st.button("Predict Bike Demand"):

        input_data = {
            "dteday": selected_date.isoformat(),
            "season": season,
            "yr": selected_date.year - 2011,
            "mnth": month,
            "hr": selected_hour,
            "holiday": holiday,
            "weekday": weekday,
            "workingday": workingday,
            "weathersit": weather,
            "temp": temperature,
            "atemp": feels_like,
            "hum": humidity,
            "windspeed": windspeed
        }

        try:

            response = requests.post(
                f"{API_URL}/predict",
                json=input_data,
                timeout=10
            )

            if response.status_code == 200:

                result = response.json()

                st.success("Prediction generated successfully.")

                st.markdown("### Prediction Result")

                st.metric(
                    "Predicted Hourly Bike Demand",
                    f"{result['predicted_demand']:,.0f} bikes"
                )

            else:

                try:
                    error_message = response.json().get(
                        "detail",
                        "Unable to generate prediction."
                    )
                except ValueError:
                    error_message = "Unable to generate prediction."

                st.error(f"Prediction failed: {error_message}")

        except requests.exceptions.ConnectionError:

            st.error(
                "Cannot connect to the prediction server. "
                "Please start the FastAPI backend."
            )

        except requests.exceptions.Timeout:

            st.error("The prediction server took too long to respond.")

        except requests.exceptions.RequestException as error:

            st.error(f"Request failed: {error}")


# =========================================================
# PAGE 6: BUSINESS RECOMMENDATIONS
# =========================================================

elif page == "Business Recommendations":

    st.title("Business Recommendations")

    st.markdown(
        "A data-driven assessment of bike demand to support "
        "fleet expansion and operational planning."
    )

    st.divider()

    # -----------------------------------------------------
    # CEO QUESTION
    # -----------------------------------------------------

    st.markdown("### The CEO's Question")

    st.markdown(
        "**Should we purchase additional bikes to meet customer demand?**"
    )

    st.info(
        "Before investing in additional bikes, we need to understand "
        "whether demand is consistently high, concentrated in specific "
        "hours, or changing across working days and seasons."
    )

    # -----------------------------------------------------
    # DEMAND OVERVIEW
    # -----------------------------------------------------

    st.markdown("### Demand Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Average Hourly Demand", f"{average_demand:.0f}")
    c2.metric("Peak Demand Hour", f"{peak_hour:02d}:00")
    c3.metric("Peak Average Rentals", f"{peak_demand:.0f}")

    if growth is not None:
        c4.metric("2011–2012 Growth", f"{growth:.1f}%")
    else:
        c4.metric("Yearly Growth", "Unavailable")

    st.caption(
        "Metrics are calculated from historical completed rental records."
    )

    st.divider()

    # -----------------------------------------------------
    # 1. PEAK DEMAND
    # -----------------------------------------------------

    st.markdown("### 1. When Does Demand Peak?")

    fig, ax = create_chart((11, 4))

    ax.bar(
        hourly_demand.index,
        hourly_demand.values,
        color=[
            RED if hour in [7, 8, 9, 16, 17, 18, 19] else GREEN
            for hour in hourly_demand.index
        ],
        width=0.7
    )

    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Average Hourly Rentals")
    ax.set_xticks(range(24))

    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    peak_hours_table = (
        hourly_demand.sort_values(ascending=False)
        .head(5)
        .reset_index()
    )

    peak_hours_table.columns = ["Hour", "Average Rentals"]

    peak_hours_table["Hour"] = peak_hours_table["Hour"].apply(
        lambda x: f"{x:02d}:00"
    )

    st.markdown("**Top 5 demand hours**")

    st.dataframe(
        peak_hours_table.round(2),
        width="stretch",
        hide_index=True
    )

    st.write(
        f"The highest average demand occurs at {peak_hour:02d}:00, "
        f"with approximately {peak_demand:.0f} rentals per hour."
    )

    st.divider()

    # -----------------------------------------------------
    # 2. HYPOTHESIS TESTING
    # -----------------------------------------------------

    st.markdown("### 2. Testable Hypothesis")

    st.markdown(
        "**Business question:** Is bike rental demand significantly "
        "higher during peak hours?"
    )

    st.markdown("""
    **Null Hypothesis (H0):** Average hourly bike demand during
    peak hours is equal to or lower than non-peak hours.

    **Alternative Hypothesis (H1):** Average hourly bike demand
    during peak hours is significantly higher than non-peak hours.
    """)

    peak_hours_list = [7, 8, 9, 16, 17, 18, 19]

    peak_data = df.loc[
        df["hr"].isin(peak_hours_list), "cnt"
    ]

    non_peak_data = df.loc[
        ~df["hr"].isin(peak_hours_list), "cnt"
    ]

    peak_mean = peak_data.mean()
    non_peak_mean = non_peak_data.mean()
    difference = peak_mean - non_peak_mean

    test_result = stats.ttest_ind(
        peak_data,
        non_peak_data,
        equal_var=False,
        alternative="greater"
    )

    peak_var = peak_data.var(ddof=1)
    non_peak_var = non_peak_data.var(ddof=1)

    peak_n = len(peak_data)
    non_peak_n = len(non_peak_data)

    standard_error = np.sqrt(
        peak_var / peak_n + non_peak_var / non_peak_n
    )

    degrees_freedom = (
        (peak_var / peak_n + non_peak_var / non_peak_n) ** 2
        /
        (
            (peak_var / peak_n) ** 2 / (peak_n - 1)
            +
            (non_peak_var / non_peak_n) ** 2 / (non_peak_n - 1)
        )
    )

    critical_value = stats.t.ppf(
        0.975, degrees_freedom
    )

    ci_lower = difference - critical_value * standard_error
    ci_upper = difference + critical_value * standard_error

    a, b, c = st.columns(3)

    a.metric("Peak-hour Average", f"{peak_mean:.1f}")
    b.metric("Non-peak Average", f"{non_peak_mean:.1f}")
    c.metric("Difference", f"{difference:.1f} bikes/hour")

    st.markdown("**Statistical Test Results**")

    st.write("Test: Welch's independent two-sample t-test")
    st.write("Significance level: 0.05")
    st.write(f"P-value: {test_result.pvalue:.6f}")

    st.write(
        f"95% confidence interval for the difference: "
        f"{ci_lower:.2f} to {ci_upper:.2f} bikes/hour"
    )

    if test_result.pvalue < 0.05 and difference > 0:

        st.success(
            "Reject H0. Historical data provides statistical "
            "evidence that peak-hour average demand is higher."
        )

    else:

        st.info(
            "Insufficient statistical evidence to conclude that "
            "peak-hour average demand is higher."
        )

    st.caption(
        "This test compares historical hourly observations. "
        "It does not establish that customers were unable to rent bikes."
    )

    st.divider()

    # -----------------------------------------------------
    # 3. EXPERIMENT DESIGN
    # -----------------------------------------------------

    st.markdown("### 3. Proposed Experiment Design")

    st.markdown(
        "**Objective:** Determine whether allocating additional bikes "
        "to high-demand locations reduces unmet customer demand."
    )

    experiment = pd.DataFrame({
        "Element": [
            "Treatment group",
            "Control group",
            "Duration",
            "Primary metric",
            "Secondary metrics",
            "Decision rule"
        ],
        "Experiment Design": [
            "Comparable locations receive additional bikes during peak hours.",
            "Comparable locations continue with normal fleet allocation.",
            "Run the pilot for 2–4 weeks, covering weekdays and weekends.",
            "Percentage of rental requests that cannot be fulfilled.",
            "Bike utilisation, completed rentals and idle bike count.",
            "Expand only if unmet demand decreases meaningfully without excessive idle capacity."
        ]
    })

    st.dataframe(
        experiment,
        width="stretch",
        hide_index=True
    )

    st.markdown("**How we will evaluate the experiment**")

    st.markdown("""
    1. Select comparable locations based on historical demand.
    2. Assign locations to treatment and control groups.
    3. Record rental requests, failed attempts and available bikes.
    4. Compare the unmet-demand rate between both groups.
    5. Check whether the additional bikes are actually being used.
    6. Use the results to decide whether fleet expansion is justified.
    """)

    st.warning(
        "The current dataset does not contain location-level inventory "
        "or failed rental attempts. These additional operational data "
        "must be collected during the pilot."
    )

    st.divider()

    # -----------------------------------------------------
    # 4. WORKING DAY ANALYSIS
    # -----------------------------------------------------

    st.markdown("### 4. Does Demand Change Across the Week?")

    working_hourly = (
        df.groupby(["hr", "workingday"])["cnt"]
        .mean()
        .unstack()
    )

    fig, ax = create_chart((11, 4))

    if 1 in working_hourly.columns:

        ax.plot(
            working_hourly.index,
            working_hourly[1],
            label="Working Day",
            color=RED,
            marker="o",
            markersize=3
        )

    if 0 in working_hourly.columns:

        ax.plot(
            working_hourly.index,
            working_hourly[0],
            label="Non-working Day",
            color=GREEN,
            marker="o",
            markersize=3
        )

    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Average Rentals")
    ax.set_xticks(range(24))
    ax.legend(frameon=False)

    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    if 1 in workingday_demand.index and 0 in workingday_demand.index:

        working_avg = workingday_demand.loc[1]
        nonworking_avg = workingday_demand.loc[0]

        a, b = st.columns(2)

        a.metric("Working Day Average", f"{working_avg:.1f}")
        b.metric("Non-working Day Average", f"{nonworking_avg:.1f}")

    st.write(
        "Hourly demand profiles differ between working and "
        "non-working days. Fleet allocation should consider day type."
    )

    st.divider()

    # -----------------------------------------------------
    # 5. SEASONAL ANALYSIS
    # -----------------------------------------------------

    st.markdown("### 5. How Does Demand Change by Season?")

    fig, ax = create_chart((10, 4))

    season_values = seasonal_demand.sort_index()

    ax.bar(
        [f"Season {int(x)}" for x in season_values.index],
        season_values.values,
        color=GREEN,
        width=0.6
    )

    ax.set_ylabel("Average Hourly Rentals")

    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    st.dataframe(
        season_values.rename("Average Rentals").reset_index().round(2),
        width="stretch",
        hide_index=True
    )

    st.caption("Season labels are shown as dataset codes.")

    st.write(
        "Seasonal variation indicates that demand planning should "
        "account for changes throughout the year."
    )

    st.divider()

    # -----------------------------------------------------
    # 6. YEARLY GROWTH
    # -----------------------------------------------------

    st.markdown("### 6. Is Demand Growing?")

    yearly_table = yearly_demand.rename(
        "Average Hourly Rentals"
    ).reset_index()

    fig, ax = create_chart((8, 4))

    ax.bar(
        yearly_demand.index.astype(str),
        yearly_demand.values,
        color=RED,
        width=0.5
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("Average Hourly Rentals")

    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    st.dataframe(
        yearly_table.round(2),
        width="stretch",
        hide_index=True
    )

    if growth is not None:

        if growth > 0:

            st.success(
                f"Average hourly rentals increased by {growth:.1f}% "
                "from 2011 to 2012."
            )

        elif growth < 0:

            st.warning(
                f"Average hourly rentals decreased by {abs(growth):.1f}% "
                "from 2011 to 2012."
            )

        else:

            st.info("Average hourly rentals remained unchanged.")

    st.caption(
        "A two-year comparison indicates historical change, "
        "but is not sufficient to establish a long-term trend."
    )

    st.divider()

    # -----------------------------------------------------
    # 7. CEO RECOMMENDATION
    # -----------------------------------------------------

    st.markdown("### 7. Recommendation to the CEO")

    st.markdown(
        "#### Recommendation: Investigate utilisation before expanding the fleet"
    )

    st.write(
        "Historical data shows demand peaks, differences between "
        "working and non-working days, seasonal variation and "
        "changes between years."
    )

    st.markdown("**Recommended actions**")

    st.markdown("""
    1. **Optimise redistribution:** Align bike availability with
       morning and evening demand peaks.

    2. **Use day-specific planning:** Develop different allocation
       strategies for working days and non-working days.

    3. **Plan for seasonal variation:** Adjust operational planning
       based on historical seasonal demand.

    4. **Monitor demand growth:** Continue tracking demand before
       making a large capital investment.

    5. **Evaluate fleet expansion:** Purchase additional bikes only
       when operational data demonstrates persistent unmet demand.
    """)

    st.divider()

    # -----------------------------------------------------
    # 8. ADDITIONAL DATA
    # -----------------------------------------------------

    st.markdown("### 8. What Additional Data Do We Need?")

    st.warning(
        "The dataset contains completed rentals. It does not contain "
        "the number of customers who could not rent a bike."
    )

    missing_information = pd.DataFrame({
        "Required Data": [
            "Available bikes by location and hour",
            "Failed rental attempts",
            "Bike utilisation rate",
            "Idle bikes by location",
            "Maintenance and unavailable bikes",
            "Customer demand by location"
        ],
        "Why It Matters": [
            "Measure actual fleet availability",
            "Identify unmet customer demand",
            "Understand how efficiently bikes are used",
            "Identify opportunities for redistribution",
            "Separate unavailable bikes from usable fleet",
            "Identify location-specific shortages"
        ]
    })

    st.dataframe(
        missing_information,
        width="stretch",
        hide_index=True
    )

    st.divider()

    # -----------------------------------------------------
    # 9. FINAL DECISION
    # -----------------------------------------------------

    st.markdown("### Final Business Decision")

    st.markdown("""
    **Do not approve a fleet expansion based only on rental volume.**

    First measure fleet utilisation, failed rental attempts and
    location-level shortages.

    If demand consistently exceeds available capacity even after
    redistribution, additional bikes may be justified.
    """)

    st.divider()

    # -----------------------------------------------------
    # 10. EXPORT
    # -----------------------------------------------------

    st.markdown("### Export Business Analysis")

    business_summary = pd.DataFrame({
        "Metric": [
            "Total Rentals",
            "Average Hourly Demand",
            "Maximum Hourly Demand",
            "Peak Demand Hour",
            "Peak Average Rentals",
            "Lowest Demand Hour",
            "Lowest Average Rentals",
            "2011 Average Demand",
            "2012 Average Demand",
            "2011-2012 Growth Percentage",
            "Peak Hour Average",
            "Non-Peak Hour Average",
            "Peak vs Non-Peak Difference",
            "Hypothesis Test P-value"
        ],
        "Value": [
            total_rentals,
            round(average_demand, 2),
            maximum_demand,
            peak_hour,
            round(peak_demand, 2),
            lowest_hour,
            round(lowest_demand, 2),
            round(yearly_demand.get(2011, np.nan), 2),
            round(yearly_demand.get(2012, np.nan), 2),
            round(growth, 2) if growth is not None else None,
            round(peak_mean, 2),
            round(non_peak_mean, 2),
            round(difference, 2),
            round(test_result.pvalue, 8)
        ]
    })

    st.download_button(
        label="Download Business Analysis CSV",
        data=business_summary.to_csv(index=False),
        file_name="cycleiq_business_analysis.csv",
        mime="text/csv"
    )


# =========================================================
# PAGE 7: OPERATIONS ASSISTANT
# =========================================================

elif page == "Ask Operations Assistant":

    st.title("Operations Assistant")

    st.markdown(
        "Ask questions about historical bike rental demand "
        "using natural language."
    )

    st.caption(
        "This version answers predefined analytical questions "
        "directly from the dataset. It does not use an external LLM."
    )

    st.divider()

    question = st.text_input(
        "Ask a question",
        placeholder="Example: Which hour has the highest demand?"
    )

    if st.button("Get Answer"):

        if not question.strip():

            st.warning("Please enter a question.")

        else:

            q = question.lower().strip()

            st.markdown("### Answer")

            if any(word in q for word in ["peak hour", "highest demand hour", "busiest hour"]):

                st.write(
                    f"The highest average demand occurs at "
                    f"{peak_hour:02d}:00, with approximately "
                    f"{peak_demand:.0f} rentals per hour."
                )

            elif any(word in q for word in ["lowest demand", "quietest hour", "least demand"]):

                st.write(
                    f"The lowest average demand occurs at "
                    f"{lowest_hour:02d}:00, with approximately "
                    f"{lowest_demand:.0f} rentals per hour."
                )

            elif any(word in q for word in ["average demand", "average rentals", "average bikes"]):

                st.write(
                    f"The average hourly rental demand is "
                    f"{average_demand:.2f} bikes."
                )

            elif any(word in q for word in ["total rentals", "total bikes", "total demand"]):

                st.write(
                    f"The dataset contains {total_rentals:,} completed rentals."
                )

            elif any(word in q for word in ["working day", "weekday"]):

                if 1 in workingday_demand.index:

                    st.write(
                        f"Average working-day demand is "
                        f"{workingday_demand.loc[1]:.2f} bikes per hour."
                    )

                if 0 in workingday_demand.index:

                    st.write(
                        f"Average non-working-day demand is "
                        f"{workingday_demand.loc[0]:.2f} bikes per hour."
                    )

            elif any(word in q for word in ["weather", "rain", "clear"]):

                weather_names = {
                    1: "Clear",
                    2: "Mist / Cloudy",
                    3: "Light Rain / Snow",
                    4: "Heavy Rain / Snow"
                }

                weather_result = (
                    df.groupby("weathersit")["cnt"]
                    .mean()
                    .sort_values(ascending=False)
                )

                best_weather = int(weather_result.index[0])

                st.write(
                    f"The highest average demand is observed in "
                    f"{weather_names[best_weather]} weather, "
                    f"at {weather_result.iloc[0]:.2f} rentals per hour."
                )

            elif any(word in q for word in ["growth", "2011", "2012", "increase"]):

                if growth is not None:

                    st.write(
                        f"Average hourly demand changed by "
                        f"{growth:.2f}% between 2011 and 2012."
                    )

                else:

                    st.write("Yearly comparison is unavailable.")

            elif any(word in q for word in ["season", "seasonal"]):

                best_season = int(seasonal_demand.idxmax())

                st.write(
                    f"Season {best_season} has the highest average "
                    f"demand at {seasonal_demand.max():.2f} rentals per hour."
                )

            elif any(word in q for word in ["recommendation", "buy bikes", "purchase bikes", "additional bikes"]):

                st.write(
                    "The historical data shows demand variation and "
                    "peak hours, but does not prove that customers "
                    "could not rent bikes. Measure failed rental "
                    "attempts, fleet utilisation and location-level "
                    "availability before purchasing additional bikes."
                )

            else:

                st.info(
                    "I could not match that question to an available "
                    "analysis. Try asking about peak hours, average "
                    "demand, weather, working days, seasons or yearly growth."
                )

    st.divider()

    st.markdown("### Example Questions")

    examples = [
        "Which hour has the highest demand?",
        "What is the average hourly demand?",
        "How many total rentals are recorded?",
        "Does demand change on working days?",
        "Which weather has the highest demand?",
        "How much did demand grow between 2011 and 2012?",
        "Should we purchase additional bikes?"
    ]

    for example in examples:
        st.markdown(f"- {example}")


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "CYCLEIQ | Bike Sharing Demand Intelligence | "
    "Python · Streamlit · FastAPI"
)
