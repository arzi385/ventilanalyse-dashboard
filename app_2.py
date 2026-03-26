import streamlit as st

import pandas as pd

import plotly.graph_objects as go

from plotly.subplots import make_subplots

from datetime import datetime

import os

import re

# ── Anlagenzuordnung ──────────────────────────────────────────────

zuordnung = {

    "F2251151TV6017": "Lufterhitzer",

    "F2251151TV6027": "Lufterhitzer",

    "F1311221TV6177": "Lufterhitzer",

    "F1311211TV5807": "WRG",

    "F1311221TV6137": "Lufterhitzer",

    "F1311221TV6147": "Lufterhitzer",

    "F1311221TV6157": "Lufterhitzer",

    "F1311221TV6167": "Lufterhitzer",

    "F1311221TV6117": "Lufterhitzer",

    "F1311211TV6017": "Luftkuehler",

    "F1311211TV6007": "Lufterhitzer",

    "F1311212TV6007": "Lufterhitzer",

    "F1311213TV6017": "Luftkuehler",

    "F1311212TV6017": "Luftkuehler",

    "F1311213TV6007": "Lufterhitzer",

    "F1311212TV5807": "WRG",

    "F1311213TV5807": "WRG",

    "F1311221TV6057": "Lufterhitzer",

    "F1311221TV6067": "Lufterhitzer",

    "F1311221TV6077": "Lufterhitzer",

    "F1311221TV6087": "Lufterhitzer",

    "F1311221TV6017": "Lufterhitzer",

    "F1311221TV6027": "Lufterhitzer",

    "F1313622TV6607": "Lufterhitzer",

    "F1313622TV6617": "Lufterhitzer",

    "F1311221TV6007": "Lufterhitzer",

    "F2252111TV6037": "Lufterhitzer",

    "F2251155TV6027": "Lufterhitzer",

    "F2251155TV6017": "Lufterhitzer",

    "F2251155TV6007": "Lufterhitzer",

    "F2251235TV6047": "Lufterhitzer",

    "F2251155TV6057": "Lufterhitzer",

    "F2251155TV6047": "Lufterhitzer",

    "F2251221TV6077": "Lufterhitzer",

    "F2251221TV6067": "Lufterhitzer",

    "F2821187TV2707": "WRG",

    "F2821186TV5807": "WRG",

    "F2821184TV2707": "WRG",

    "F2821183TV2707": "WRG",

    "F2821182TV2707": "WRG",

    "F2821181TV2707": "WRG",

    "F2821166TV5807": "WRG",

    "F2821165TV2707": "WRG",

    "F2821161TV6027": "WRG",

    "F2821151TV6004": "WRG",

    "F2821135TV6027": "WRG",

    "F2821131TV6008": "WRG",

    "F2821112TV5807": "WRG",

    "F2823662TV6157": "Lufterhitzer",

    "F2823662TV6147": "Lufterhitzer",

    "F2823662TV6137": "Lufterhitzer",

    "F2823662TV6127": "Lufterhitzer",

    "F2823662TV6117": "Lufterhitzer",

    "F2823662TV6107": "Lufterhitzer",

    "F2823662TV6177": "Lufterhitzer",

    "F2823662TV6187": "Lufterhitzer",

    "F2821187TV6007": "Lufterhitzer",

    "F2821186TV6007": "Lufterhitzer",

    "F2821184TV6007": "Lufterhitzer",

    "F2821184TV6008": "Lufterhitzer",

    "F2821183TV6007": "Lufterhitzer",

    "F2821182TV6007": "Lufterhitzer",

    "F2821181TV6007": "Lufterhitzer",

    "F2821166TV6007": "Lufterhitzer",

    "F2821165TV6007": "Lufterhitzer",

    "F2821161TV6007": "Lufterhitzer",

    "F2821151TV2106": "Lufterhitzer",

    "F2821151TV2116": "Lufterhitzer",

    "F2821151TV2126": "Lufterhitzer",

    "F2821136TV6007": "Lufterhitzer",

    "F2821135TV6007": "Lufterhitzer",

    "F2821131TV6007": "Lufterhitzer",

    "F2821125TV6007": "Lufterhitzer",

    "F2821112TV6007": "Lufterhitzer",

    "F2821112TV6017": "Lufterhitzer",

    "F2821187TV6017": "Luftkuehler",

    "F2821184TV6017": "Luftkuehler",

    "F2821183TV6017": "Luftkuehler",

    "F2821182TV6017": "Luftkuehler",

    "F2821181TV6017": "Luftkuehler",

    "F2821165TV6017": "Luftkuehler",

    "F2821161TV6017": "Luftkuehler",

    "F2821135TV6017": "Luftkuehler",

    "F2821131TV6017": "Luftkuehler",

    "F2821131TV6047": "Luftkuehler",

    "F2821131TV6057": "Luftkuehler",

    "F2821112TV6027": "Luftkuehler",

}


def get_anlagentyp(valve_id):
    """Gibt den Anlagentyp fuer eine Ventil-ID zurueck"""

    return zuordnung.get(valve_id, "Unbekannt")


# ── Seiten-Konfiguration ──────────────────────────────────────────

st.set_page_config(

    page_title="Ventilueberwachung Dashboard",

    page_icon="V",

    layout="wide",

    initial_sidebar_state="expanded"

)

# ── Custom CSS ─────────────────────────────────────────────────────

st.markdown("""

<style>

    .main-header {

        font-size: 2.2rem;

        font-weight: 700;

        color: #1E3A5F;

        text-align: center;

        padding: 1rem 0;

        border-bottom: 3px solid #3A7CA5;

        margin-bottom: 2rem;

        letter-spacing: 0.5px;

    }

    .metric-card {

        background: linear-gradient(135deg, #2C3E50 0%, #3A7CA5 100%);

        padding: 1.2rem;

        border-radius: 10px;

        color: white;

        text-align: center;

        box-shadow: 0 4px 12px rgba(0,0,0,0.12);

        margin-bottom: 0.5rem;

    }

    .metric-card h3 {

        margin: 0;

        font-size: 0.8rem;

        opacity: 0.85;

        font-weight: 400;

    }

    .metric-card .valve-type {

        margin: 0.2rem 0 0 0;

        font-size: 0.7rem;

        opacity: 0.7;

        font-style: italic;

    }

    .metric-card h1 {

        margin: 0.4rem 0 0 0;

        font-size: 2rem;

    }

    .metric-card .status-label {

        margin: 0.2rem 0 0 0;

        font-size: 0.8rem;

        font-weight: 500;

    }

    .status-good {

        background: linear-gradient(135deg, #1B7A3D 0%, #2EAD5A 100%);

    }

    .status-warning {

        background: linear-gradient(135deg, #B8860B 0%, #DAA520 100%);

    }

    .status-bad {

        background: linear-gradient(135deg, #8B1A1A 0%, #C0392B 100%);

    }

    .section-title {

        font-size: 1.1rem;

        font-weight: 600;

        color: #2C3E50;

        margin: 1.5rem 0 0.8rem 0;

        padding-bottom: 0.3rem;

        border-bottom: 2px solid #e0e0e0;

    }

    .sidebar-section {

        font-size: 0.95rem;

        font-weight: 600;

        color: #2C3E50;

        margin-bottom: 0.5rem;

    }

    .stTabs [data-baseweb="tab-list"] {

        gap: 4px;

    }

    .stTabs [data-baseweb="tab"] {

        background-color: #f0f2f6;

        border-radius: 6px;

        padding: 8px 16px;

        font-size: 0.9rem;

    }

    .empty-state {

        text-align: center;

        padding: 4rem 2rem;

        background: #f8f9fa;

        border-radius: 12px;

        margin: 2rem 0;

        border: 1px solid #e9ecef;

    }

    .empty-state h2 {

        color: #6c757d;

        font-size: 1.4rem;

        margin-bottom: 0.5rem;

    }

    .empty-state p {

        color: #adb5bd;

        font-size: 1rem;

    }

</style>

""", unsafe_allow_html=True)


# ── Hilfsfunktionen ───────────────────────────────────────────────


def get_status_color(value):
    if value < 30:

        return "#2EAD5A"

    elif value < 70:

        return "#DAA520"

    else:

        return "#C0392B"


def get_status_text(value):
    if value < 30:

        return "Normal"

    elif value < 70:

        return "Erhoht"

    else:

        return "Kritisch"


def parse_csv_file(file_path):
    try:

        with open(file_path, 'r', encoding='utf-8') as f:

            first_line = f.readline().strip()

        if first_line.startswith("sep="):

            sep = first_line.split("=")[1]

            df = pd.read_csv(file_path, sep=sep, skiprows=1, engine='python')

        else:

            df = pd.read_csv(file_path, sep=';', engine='python')

        time_col = None

        for col in df.columns:

            if 'zeit' in col.lower() or 'time' in col.lower() or 'stamp' in col.lower() or 'datum' in col.lower():
                time_col = col

                break

        if time_col is None:
            time_col = df.columns[0]

        df[time_col] = pd.to_datetime(df[time_col], dayfirst=True, errors='coerce')

        df = df.dropna(subset=[time_col])

        df = df.rename(columns={time_col: 'Zeitstempel'})

        for col in df.columns:

            if col != 'Zeitstempel':
                df[col] = df[col].astype(str).str.replace(',', '.')

                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    except Exception as e:

        st.error(f"Fehler beim Lesen von {file_path}: {e}")

        return None


def extract_valve_id(filename):
    name = os.path.splitext(filename)[0]

    name = re.sub(r'_export$', '', name, flags=re.IGNORECASE)

    return name


# ── Sidebar ────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="sidebar-section">Einstellungen</div>', unsafe_allow_html=True)

    st.markdown("---")

    uploaded_files = st.file_uploader(

        "CSV-Dateien hochladen",

        type=['csv'],

        accept_multiple_files=True,

        help="Laden Sie eine oder mehrere CSV-Dateien mit Ventildaten hoch"

    )

    st.markdown("---")

    st.markdown('<div class="sidebar-section">Anlagentyp-Filter</div>', unsafe_allow_html=True)

    # Alle verfuegbaren Typen aus dem Dictionary

    alle_typen = sorted(set(zuordnung.values()))

    alle_typen_mit_alle = ["Alle"] + alle_typen

    selected_type = st.selectbox(

        "Anlagentyp anzeigen",

        alle_typen_mit_alle,

        index=0,

        help="Waehlen Sie einen Anlagentyp, um nur diese Ventile anzuzeigen"

    )

    st.markdown("---")

    st.markdown('<div class="sidebar-section">Anzeigeoptionen</div>', unsafe_allow_html=True)

    chart_height = st.slider("Diagrammhoehe (px)", 300, 800, 500)

    show_grid = st.checkbox("Gitternetz anzeigen", value=True)

    show_rangeslider = st.checkbox("Zeitbereich-Slider", value=True)

    color_theme = st.selectbox(

        "Farbschema",

        ["Standard", "Viridis", "Plasma", "Inferno", "Magma"]

    )

    color_maps = {

        "Standard": ['#3A7CA5', '#2EAD5A', '#C0392B', '#DAA520', '#5B6ABF',

                     '#8E44AD', '#1ABC9C', '#D35400', '#27AE60', '#E74C6F'],

        "Viridis": ['#440154', '#482878', '#3e4989', '#31688e', '#26828e',

                    '#1f9e89', '#35b779', '#6ece58', '#b5de2b', '#fde725'],

        "Plasma": ['#0d0887', '#46039f', '#7201a8', '#9c179e', '#bd3786',

                   '#d8576b', '#ed7953', '#fb9f3a', '#fdca26', '#f0f921'],

        "Inferno": ['#000004', '#1b0c41', '#4a0c6b', '#781c6d', '#a52c60',

                    '#cf4446', '#ed6925', '#fb9b06', '#f7d13d', '#fcffa4'],

        "Magma": ['#000004', '#180f3d', '#440f76', '#721f81', '#9e2f7f',

                  '#cd4071', '#f1605d', '#fd9668', '#feca8d', '#fcfdbf']

    }

    colors = color_maps[color_theme]

# ── Header ─────────────────────────────────────────────────────────

st.markdown('<div class="main-header">Ventilueberwachung Dashboard</div>', unsafe_allow_html=True)

# ── Hauptbereich ───────────────────────────────────────────────────

if not uploaded_files:
    st.markdown("""

    <div class="empty-state">

        <h2>Keine Daten geladen</h2>

        <p>Bitte laden Sie CSV-Dateien ueber die Seitenleiste hoch, um das Dashboard zu starten.</p>

    </div>

    """, unsafe_allow_html=True)

    st.stop()

# ── Daten einlesen ─────────────────────────────────────────────────

all_data = {}

for uploaded_file in uploaded_files:

    temp_path = f"/tmp/{uploaded_file.name}"

    with open(temp_path, 'wb') as f:

        f.write(uploaded_file.getbuffer())

    df = parse_csv_file(temp_path)

    if df is not None:
        valve_id = extract_valve_id(uploaded_file.name)

        all_data[valve_id] = df

if not all_data:
    st.error("Keine gueltigen Daten gefunden.")

    st.stop()

# ── Filter anwenden ────────────────────────────────────────────────

if selected_type != "Alle":

    filtered_data = {

        vid: df for vid, df in all_data.items()

        if get_anlagentyp(vid) == selected_type

    }

else:

    filtered_data = all_data

if not filtered_data:
    st.warning(

        f"Keine Ventile vom Typ '{selected_type}' in den hochgeladenen Daten gefunden."

    )

    st.stop()

# ── Info-Leiste: Aktiver Filter ───────────────────────────────────

filter_label = selected_type if selected_type != "Alle" else "Alle Anlagentypen"

st.markdown(

    f'<div style="background:#EBF5FB; padding:0.5rem 1rem; border-radius:6px; '

    f'font-size:0.85rem; color:#2C3E50; margin-bottom:1rem; border-left:4px solid #3A7CA5;">'

    f'<strong>Filter:</strong> {filter_label} &nbsp;|&nbsp; '

    f'<strong>Angezeigte Ventile:</strong> {len(filtered_data)}'

    f'</div>',

    unsafe_allow_html=True

)

# ── Metriken ───────────────────────────────────────────────────────

st.markdown('<div class="section-title">Uebersicht</div>', unsafe_allow_html=True)

current_values = {}

for valve_id, df in filtered_data.items():

    value_cols = [c for c in df.columns if c != 'Zeitstempel']

    if value_cols:

        last_valid = df[value_cols[0]].dropna()

        if not last_valid.empty:
            current_values[valve_id] = last_valid.iloc[-1]

# Metrik-Karten (max 4 pro Zeile)

if current_values:

    items = list(current_values.items())

    for row_start in range(0, len(items), 4):

        row_items = items[row_start:row_start + 4]

        cols = st.columns(len(row_items))

        for col, (valve_id, value) in zip(cols, row_items):
            status = get_status_text(value)

            status_class = (

                "status-good" if status == "Normal"

                else ("status-warning" if status == "Erhoht" else "status-bad")

            )

            anlagentyp = get_anlagentyp(valve_id)

            with col:
                st.markdown(f"""

                <div class="metric-card {status_class}">

                    <h3>{valve_id}</h3>

                    <div class="valve-type">{anlagentyp}</div>

                    <h1>{value:.1f}%</h1>

                    <div class="status-label">{status}</div>

                </div>

                """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["Zeitverlauf", "Vergleich", "Detailansicht"])

# ── Tab 1: Zeitverlauf ─────────────────────────────────────────────

with tab1:
    st.markdown('<div class="section-title">Ventilstellungen im Zeitverlauf</div>', unsafe_allow_html=True)

    fig = make_subplots(rows=1, cols=1)

    for idx, (valve_id, df) in enumerate(filtered_data.items()):

        value_cols = [c for c in df.columns if c != 'Zeitstempel']

        anlagentyp = get_anlagentyp(valve_id)

        for col in value_cols:
            color = colors[idx % len(colors)]

            fig.add_trace(go.Scatter(

                x=df['Zeitstempel'],

                y=df[col],

                mode='lines',

                name=f"{valve_id}",

                line=dict(color=color, width=2),

                hovertemplate=(

                    f"<b>{valve_id}</b><br>"

                    f"Typ: {anlagentyp}<br>"

                    f"Zeit: %{{x}}<br>"

                    f"Wert: %{{y:.1f}}%<extra></extra>"

                )

            ))

    fig.update_layout(

        height=chart_height,

        template="plotly_white",

        hovermode="x unified",

        legend=dict(

            orientation="h",

            yanchor="bottom",

            y=1.02,

            xanchor="right",

            x=1,

            font=dict(size=10)

        ),

        xaxis=dict(

            title="Zeit",

            showgrid=show_grid,

            rangeslider=dict(visible=show_rangeslider)

        ),

        yaxis=dict(

            title="Ventilstellung (%)",

            showgrid=show_grid,

            range=[0, 105]

        ),

        margin=dict(l=60, r=30, t=40, b=40)

    )

    st.plotly_chart(fig, use_container_width=True)

# ── Tab 2: Vergleich ──────────────────────────────────────────────

with tab2:
    st.markdown('<div class="section-title">Vergleich der aktuellen Ventilstellungen</div>', unsafe_allow_html=True)

    if current_values:

        valve_ids = list(current_values.keys())

        values = list(current_values.values())

        bar_colors = [get_status_color(v) for v in values]

        hover_texts = [

            f"<b>{vid}</b><br>Typ: {get_anlagentyp(vid)}<br>Ventilstellung: {val:.1f}%"

            for vid, val in zip(valve_ids, values)

        ]

        fig_bar = go.Figure(data=[

            go.Bar(

                x=valve_ids,

                y=values,

                marker_color=bar_colors,

                text=[f"{v:.1f}%" for v in values],

                textposition='auto',

                hovertext=hover_texts,

                hoverinfo='text'

            )

        ])

        fig_bar.update_layout(

            height=chart_height,

            template="plotly_white",

            xaxis=dict(title="Ventil-ID", tickangle=-45),

            yaxis=dict(title="Ventilstellung (%)", range=[0, 105]),

            margin=dict(l=60, r=30, t=40, b=120)

        )

        st.plotly_chart(fig_bar, use_container_width=True)

        # Statistik-Tabelle

        st.markdown('<div class="section-title">Statistiken</div>', unsafe_allow_html=True)

        stats_data = []

        for valve_id, df in filtered_data.items():

            value_cols = [c for c in df.columns if c != 'Zeitstempel']

            anlagentyp = get_anlagentyp(valve_id)

            for col in value_cols:

                series = df[col].dropna()

                if not series.empty:
                    stats_data.append({

                        'Ventil-ID': valve_id,

                        'Anlagentyp': anlagentyp,

                        'Aktuell (%)': f"{series.iloc[-1]:.1f}",

                        'Mittelwert (%)': f"{series.mean():.1f}",

                        'Min (%)': f"{series.min():.1f}",

                        'Max (%)': f"{series.max():.1f}",

                        'Std.Abw. (%)': f"{series.std():.1f}",

                        'Status': get_status_text(series.iloc[-1])

                    })

        if stats_data:
            stats_df = pd.DataFrame(stats_data)

            st.dataframe(stats_df, use_container_width=True, hide_index=True)

# ── Tab 3: Detailansicht ──────────────────────────────────────────

with tab3:
    st.markdown('<div class="section-title">Einzelne Ventilanalyse</div>', unsafe_allow_html=True)

    # Dropdown mit Anlagentyp in Klammern

    valve_options = [

        f"{vid}  [{get_anlagentyp(vid)}]" for vid in filtered_data.keys()

    ]

    valve_id_list = list(filtered_data.keys())

    selected_label = st.selectbox("Ventil auswaehlen", valve_options)

    selected_idx = valve_options.index(selected_label)

    selected_valve = valve_id_list[selected_idx]

    if selected_valve:

        df = filtered_data[selected_valve]

        value_cols = [c for c in df.columns if c != 'Zeitstempel']

        anlagentyp = get_anlagentyp(selected_valve)

        st.markdown(

            f'<div style="background:#F4F6F7; padding:0.4rem 0.8rem; border-radius:6px; '

            f'font-size:0.85rem; color:#2C3E50; margin-bottom:1rem;">'

            f'<strong>Anlage:</strong> {selected_valve} &nbsp;|&nbsp; '

            f'<strong>Typ:</strong> {anlagentyp}'

            f'</div>',

            unsafe_allow_html=True

        )

        if value_cols:
            col_to_plot = value_cols[0]

            series = df[col_to_plot].dropna()

            m1, m2, m3, m4 = st.columns(4)

            with m1:
                st.metric("Aktueller Wert", f"{series.iloc[-1]:.1f}%")

            with m2:
                st.metric("Mittelwert", f"{series.mean():.1f}%")

            with m3:
                st.metric("Minimum", f"{series.min():.1f}%")

            with m4:
                st.metric("Maximum", f"{series.max():.1f}%")

            fig_detail = go.Figure()

            fig_detail.add_trace(go.Scatter(

                x=df['Zeitstempel'],

                y=df[col_to_plot],

                mode='lines',

                name=selected_valve,

                line=dict(color='#3A7CA5', width=2.5),

                fill='tozeroy',

                fillcolor='rgba(58, 124, 165, 0.08)',

                hovertemplate=(

                    f"<b>{selected_valve}</b><br>"

                    f"Typ: {anlagentyp}<br>"

                    f"Zeit: %{{x}}<br>"

                    f"Wert: %{{y:.1f}}%<extra></extra>"

                )

            ))

            mean_val = series.mean()

            fig_detail.add_hline(

                y=mean_val,

                line_dash="dash",

                line_color="#C0392B",

                annotation_text=f"Mittelwert: {mean_val:.1f}%",

                annotation_position="top right"

            )

            fig_detail.update_layout(

                height=chart_height,

                template="plotly_white",

                xaxis=dict(

                    title="Zeit",

                    showgrid=show_grid,

                    rangeslider=dict(visible=show_rangeslider)

                ),

                yaxis=dict(

                    title="Ventilstellung (%)",

                    showgrid=show_grid,

                    range=[0, 105]

                ),

                margin=dict(l=60, r=30, t=40, b=40)

            )

            st.plotly_chart(fig_detail, use_container_width=True)

            with st.expander("Rohdaten anzeigen"):
                st.dataframe(df, use_container_width=True, hide_index=True)

# ── Footer ─────────────────────────────────────────────────────────

st.markdown("---")

st.markdown(

    '<p style="text-align: center; color: #adb5bd; font-size: 0.8rem;">'

    'Ventilueberwachung Dashboard  |  Streamlit & Plotly'

    '</p>',

    unsafe_allow_html=True

)