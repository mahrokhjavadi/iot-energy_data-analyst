from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd

# ============================================
# Load dataset (cleaned + clustering)
# ============================================
df = pd.read_csv('C:\\0_DA\\iot-energy_data-analyst\\outputs\\4_result_clustering\\clustering_results.csv ')

# Convert timestamp
df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

# Time features
df["Hour"] = df["Timestamp"].dt.hour
df["Weekday"] = df["Timestamp"].dt.day_name()
df["Month"] = df["Timestamp"].dt.month

# ============================================
# Initialize Dash App
# ============================================
app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])

# ============================================
# Layout
# ============================================
app.layout = dbc.Container(fluid=True, children=[

    dbc.Row([
        dbc.Col(html.H1("Smart Energy IoT Dashboard",
                        className="text-center mt-4 mb-4"), width=12)
    ]),

    dbc.Tabs([

        # ==========================================================
        # RQ1: Consumption Patterns
        # ==========================================================
        dbc.Tab(label="RQ1: Consumption Patterns", children=[
            html.Br(),

            dcc.Dropdown(
                id="rq1-agg",
                options=[
                    {"label": "Hourly", "value": "Hour"},
                    {"label": "Weekly", "value": "Weekday"},
                    {"label": "Monthly", "value": "Month"},
                ],
                value="Hour",
                clearable=False,
                style={"width": "40%"}
            ),

            dcc.Graph(id="rq1-plot"),
        ]),

        # ==========================================================
        # RQ2: Power & Quality Relations
        # ==========================================================
        dbc.Tab(label="RQ2: Power & Quality Relations", children=[
            html.Br(),

            dcc.Dropdown(
                id="rq2-feature",
                options=[
                    {"label": "Voltage (V)", "value": "Voltage (V)"},
                    {"label": "Reactive Power (kVAR)", "value": "Reactive Power (kVAR)"},
                    {"label": "Power Factor", "value": "PowerFactor"},
                ],
                value="Voltage (V)",
                clearable=False,
                style={"width": "40%"}
            ),

            dcc.Graph(id="rq2-scatter"),
        ]),

        # ==========================================================
        # RQ3: Clustering (KMeans, Hierarchical, DBSCAN, GMM)
        # ==========================================================
        dbc.Tab(label="RQ3: Clustering & Anomalies", children=[
            html.Br(),

            dcc.Dropdown(
                id="rq3-model",
                options=[
                    {"label": "KMeans", "value": "Cluster_KMeans"},
                    {"label": "Hierarchical", "value": "Cluster_Hierarchical"},
                    {"label": "DBSCAN", "value": "Cluster_DBSCAN"},
                    {"label": "GMM", "value": "Cluster_GMM"},
                ],
                value="Cluster_KMeans",
                clearable=False,
                style={"width": "40%"}
            ),

            dcc.Graph(id="rq3-scatter"),
            dcc.Graph(id="rq3-timeseries")
        ]),
    ])
])





# ==========================================================
# CALLBACKS
# ==========================================================

# ---------------- RQ1 --------------------
@app.callback(
    Output("rq1-plot", "figure"),
    Input("rq1-agg", "value")
)

def update_rq1(agg):
    fig = px.line(
        df.groupby(agg)["Power Consumption (kW)"].mean().reset_index(),
        x=agg,
        y="Power Consumption (kW)",
        title=f"Average Power Consumption by {agg}"
    )
    return fig


# ---------------- RQ2 --------------------
@app.callback(
    Output("rq2-scatter", "figure"),
    Input("rq2-feature", "value")
)

def update_rq2(feature):
    fig = px.scatter(
        df.sample(5000),
        x="Power Consumption (kW)",
        y=feature,
        color="Month",
        title=f"Power Consumption vs {feature}"
    )
    return fig


# ---------------- RQ3 --------------------
@app.callback(
    Output("rq3-scatter", "figure"),
    Output("rq3-timeseries", "figure"),
    Input("rq3-model", "value")
)

def update_rq3(model):

    # Scatter plot using real features (Voltage vs Power Consumption)
    fig1 = px.scatter(
        df,
        x="Voltage (V)",
        y="Power Consumption (kW)",
        color=model,
        title=f"Clustering Visualization using Voltage vs Consumption ({model})",
        opacity=0.8
    )

    # Select one cluster value to show the time-series
    cluster_value = df[model].unique()[0]
    df_c = df[df[model] == cluster_value]

    fig2 = px.line(
        df_c.head(3000),
        x="Timestamp",
        y="Power Consumption (kW)",
        title=f"Time-Series for {model} — Cluster {cluster_value}",
        markers=False
    )

    return fig1, fig2



# ==========================================================
# Run Server
# ==========================================================
if __name__ == "__main__":
    app.run(debug=True)
