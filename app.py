"""
EquiCity - Equity-Aware Decision Intelligence Platform
Main Streamlit Application — Premium Dashboard UI

An AI-powered platform that audits whether city resource allocation
is fair across all neighborhoods.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import json
from src.data_loader import load_complaints, get_summary_stats, filter_complaints
from src.fairness import compute_fairness_scores, get_disparity_report, get_recommendations
from src.decision_ledger import DecisionLedger
from src.gemini_agent import create_agent, run_agent_query

# --- Page Config ---
st.set_page_config(
    page_title="EquiCity - Equity-Aware Decision Intelligence",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Premium CSS ---
st.markdown("""
<style>

    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }

    /* Hero Header */
    .hero {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(102, 126, 234, 0.15) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero::after {
        content: '';
        position: absolute;
        bottom: -30%;
        left: -10%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(255, 107, 107, 0.1) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero h1 {
        color: #fff;
        font-size: 2.4rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
        position: relative;
        z-index: 1;
    }
    .hero .subtitle {
        color: rgba(255,255,255,0.7);
        font-size: 1.05rem;
        margin-top: 0.4rem;
        font-weight: 400;
        position: relative;
        z-index: 1;
    }
    .hero .tagline {
        display: inline-block;
        background: linear-gradient(90deg, #ff6b6b, #ee5a24);
        color: white;
        padding: 0.25rem 0.9rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-top: 0.8rem;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        position: relative;
        z-index: 1;
    }

    /* Glassmorphism Metric Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 1.4rem 1.2rem;
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    .glass-card:hover {
        border-color: rgba(102, 126, 234, 0.3);
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.1);
    }
    .glass-card .icon {
        font-size: 1.8rem;
        margin-bottom: 0.5rem;
    }
    .glass-card .value {
        font-size: 2.2rem;
        font-weight: 800;
        line-height: 1;
        margin-bottom: 0.3rem;
    }
    .glass-card .label {
        color: rgba(255, 255, 255, 0.5);
        font-size: 0.78rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    .value-red { color: #ff6b6b; }
    .value-green { color: #2ed573; }
    .value-orange { color: #ffa502; }
    .value-blue { color: #667eea; }
    .value-purple { color: #a855f7; }

    /* Alert Banner */
    .equity-alert {
        background: linear-gradient(90deg, rgba(255, 71, 87, 0.12), rgba(255, 71, 87, 0.03));
        border: 1px solid rgba(255, 71, 87, 0.25);
        border-left: 4px solid #ff4757;
        border-radius: 0 12px 12px 0;
        padding: 1rem 1.4rem;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.8rem;
    }
    .equity-alert .alert-icon {
        font-size: 1.6rem;
        flex-shrink: 0;
    }
    .equity-alert .alert-text {
        color: rgba(255, 255, 255, 0.9);
        font-size: 0.92rem;
        line-height: 1.5;
    }
    .equity-alert .alert-text strong {
        color: #ff6b6b;
    }

    /* Section Headers */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.6rem;
        border-bottom: 1px solid rgba(255,255,255,0.06);
    }
    .section-header h3 {
        color: #fff;
        font-size: 1.15rem;
        font-weight: 700;
        margin: 0;
    }
    .section-header .badge {
        background: rgba(102, 126, 234, 0.15);
        color: #667eea;
        font-size: 0.7rem;
        padding: 0.15rem 0.55rem;
        border-radius: 10px;
        font-weight: 600;
    }

    /* Income Band Cards */
    .band-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .band-card h4 {
        margin: 0 0 0.8rem 0;
        font-size: 1rem;
        font-weight: 700;
    }
    .band-low { border-top: 3px solid #ff6b6b; }
    .band-medium { border-top: 3px solid #ffa502; }
    .band-high { border-top: 3px solid #2ed573; }
    .band-stat {
        display: flex;
        justify-content: space-between;
        padding: 0.35rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.04);
        font-size: 0.85rem;
    }
    .band-stat .stat-label { color: rgba(255,255,255,0.5); }
    .band-stat .stat-value { color: #fff; font-weight: 600; }

    /* Ledger Entry */
    .ledger-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-left: 3px solid #667eea;
        border-radius: 0 12px 12px 0;
        padding: 1rem 1.2rem;
        margin-bottom: 0.5rem;
        transition: border-color 0.2s;
    }
    .ledger-card:hover {
        border-left-color: #a855f7;
    }

    /* Chat example buttons */
    .example-btn {
        background: rgba(102, 126, 234, 0.08);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 10px;
        padding: 0.6rem 0.9rem;
        color: rgba(255,255,255,0.8);
        font-size: 0.83rem;
        cursor: pointer;
        transition: all 0.2s;
        text-align: left;
        width: 100%;
    }
    .example-btn:hover {
        background: rgba(102, 126, 234, 0.15);
        border-color: rgba(102, 126, 234, 0.4);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #1a1f2e 100%);
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: rgba(255,255,255,0.03);
        padding: 4px;
        border-radius: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 10px 24px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(102, 126, 234, 0.15);
    }

    /* Hide defaults */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Plotly chart containers */
    .chart-container {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 14px;
        padding: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


# --- Initialize Session State ---
if "ledger" not in st.session_state:
    st.session_state.ledger = DecisionLedger()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "gemini_chat_history" not in st.session_state:
    st.session_state.gemini_chat_history = []
if "gemini_client" not in st.session_state:
    st.session_state.gemini_client = None


# --- Load Data ---
@st.cache_data
def get_data():
    return load_complaints()


@st.cache_data
def get_fairness():
    df = load_complaints()
    return compute_fairness_scores(df)


@st.cache_data
def get_disparity():
    df = load_complaints()
    return get_disparity_report(df)


@st.cache_data
def get_recs():
    df = load_complaints()
    return get_recommendations(df)


df = get_data()
stats = get_summary_stats(df)
fairness_data = get_fairness()
disparity_data = get_disparity()
recs_data = get_recs()


# --- Hero Header ---
st.markdown("""
<div class="hero">
    <h1>🏛️ EquiCity</h1>
    <div class="subtitle">Equity-Aware Decision Intelligence Platform</div>
    <div class="tagline">🔍 Is your city treating every neighborhood fairly?</div>
</div>
""", unsafe_allow_html=True)


# --- Sidebar ---
with st.sidebar:
    st.markdown("## ⚡ Control Panel")

    api_key = st.text_input(
        "🔑 AI API Key (Groq)",
        type="password",
        help="Get your free API key from console.groq.com",
        placeholder="Paste your Groq API key...",
    )

    if api_key and st.session_state.gemini_client is None:
        try:
            st.session_state.gemini_client = create_agent(api_key)
            st.success("✅ AI Agent online!")
        except Exception as e:
            st.error(f"Connection failed: {e}")

    st.markdown("---")

    # Sidebar stats with colored metrics
    st.markdown("### 📈 City Pulse")
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Complaints", stats["total_complaints"])
        st.metric("Fairness", f"{fairness_data['overall_fairness_score']}/100")
    with col_b:
        st.metric("Resolved", stats["resolution_rate"])
        st.metric("Avg Days", stats["avg_resolution_days"])

    st.markdown("---")
    st.caption(f"📅 Data: {stats['date_range']}")
    st.caption("🏗️ Built for Gen AI Academy APAC 2026")
    st.caption("⚡ Powered by Gemini 2.0 Flash + Streamlit")


# --- Disparity Alert ---
if disparity_data["disparity_ratio"] and disparity_data["disparity_ratio"] > 2:
    st.markdown(f"""
    <div class="equity-alert">
        <div class="alert-icon">🚨</div>
        <div class="alert-text">
            <strong>Critical Equity Alert:</strong> {disparity_data['disparity_summary']}
        </div>
    </div>
    """, unsafe_allow_html=True)


# --- Main Content ---
tab1, tab2, tab3 = st.tabs(["📊 Fairness Dashboard", "💬 AI Equity Analyst", "📋 Decision Ledger"])


# ============================
# TAB 1: FAIRNESS DASHBOARD
# ============================
with tab1:

    # --- Metric Cards Row ---
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.markdown(f"""
        <div class="glass-card">
            <div class="icon">⚖️</div>
            <div class="value value-blue">{fairness_data['overall_fairness_score']}</div>
            <div class="label">Fairness Score</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="glass-card">
            <div class="icon">📊</div>
            <div class="value value-red">{disparity_data['disparity_ratio']}x</div>
            <div class="label">Income Disparity</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="glass-card">
            <div class="icon">📋</div>
            <div class="value value-purple">{stats['total_complaints']}</div>
            <div class="label">Total Complaints</div>
        </div>""", unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="glass-card">
            <div class="icon">✅</div>
            <div class="value value-green">{stats['resolution_rate']}</div>
            <div class="label">Resolution Rate</div>
        </div>""", unsafe_allow_html=True)

    with c5:
        pending_count = len(df[df["status"].isin(["pending", "in_progress"])])
        st.markdown(f"""
        <div class="glass-card">
            <div class="icon">⏳</div>
            <div class="value value-orange">{pending_count}</div>
            <div class="label">Pending</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    # --- Charts Row 1 ---
    st.markdown('<div class="section-header"><h3>📍 Neighborhood Analysis</h3><span class="badge">CORE INSIGHT</span></div>', unsafe_allow_html=True)

    chart_col1, chart_col2 = st.columns(2)
    nb_data = pd.DataFrame(fairness_data["neighborhood_breakdown"])

    with chart_col1:
        color_map = {"low": "#ff6b6b", "medium": "#ffa502", "high": "#2ed573"}

        fig1 = px.bar(
            nb_data.sort_values("avg_resolution_days", ascending=True),
            y="neighborhood",
            x="avg_resolution_days",
            color="income_band",
            color_discrete_map=color_map,
            orientation="h",
            labels={"avg_resolution_days": "Avg Resolution (days)", "neighborhood": "", "income_band": "Income"},
            text="avg_resolution_days",
        )
        fig1.update_layout(
            title=dict(text="Resolution Time by Neighborhood", font=dict(size=15, color="white")),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="rgba(255,255,255,0.7)", size=12),
            height=380,
            showlegend=True,
            legend=dict(orientation="h", yanchor="top", y=-0.2, font=dict(size=11)),
            margin=dict(l=0, r=20, t=50, b=0),
            xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
        )
        fig1.update_traces(
            texttemplate="%{text} days",
            textposition="outside",
            textfont=dict(size=11),
        )
        st.plotly_chart(fig1, width="stretch")

    with chart_col2:
        colors = ["#ff4757" if s < 40 else "#ffa502" if s < 70 else "#2ed573" for s in nb_data["fairness_score"]]

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=nb_data["neighborhood"],
            y=nb_data["fairness_score"],
            marker=dict(
                color=colors,
                line=dict(width=0),
                cornerradius=6,
            ),
            text=[f"{s}" for s in nb_data["fairness_score"]],
            textposition="outside",
            textfont=dict(size=12, color="rgba(255,255,255,0.8)"),
        ))
        fig2.add_hline(y=70, line_dash="dot", line_color="rgba(46, 213, 115, 0.4)",
                       annotation_text="Fair (70)", annotation_font_color="rgba(46,213,115,0.6)")
        fig2.update_layout(
            title=dict(text="Fairness Score by Neighborhood", font=dict(size=15, color="white")),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="rgba(255,255,255,0.7)", size=12),
            height=380,
            yaxis=dict(range=[0, 115], title="Score (0-100)", gridcolor="rgba(255,255,255,0.04)"),
            xaxis=dict(title=""),
            margin=dict(l=0, r=0, t=50, b=0),
            showlegend=False,
        )
        st.plotly_chart(fig2, width="stretch")

    # --- Income Band Comparison ---
    st.markdown('<div class="section-header"><h3>💰 Income Band Comparison</h3><span class="badge">EQUITY GAP</span></div>', unsafe_allow_html=True)

    band_data = disparity_data["income_band_comparison"]
    b1, b2, b3 = st.columns(3)

    for col, band, css_class in zip([b1, b2, b3], ["low", "medium", "high"], ["band-low", "band-medium", "band-high"]):
        with col:
            if band in band_data:
                b = band_data[band]
                emoji = {"low": "🔴", "medium": "🟡", "high": "🟢"}[band]
                st.markdown(f"""
                <div class="band-card {css_class}">
                    <h4>{emoji} {band.upper()} INCOME</h4>
                    <div class="band-stat"><span class="stat-label">Avg Resolution</span><span class="stat-value">{b['avg_resolution_days']} days</span></div>
                    <div class="band-stat"><span class="stat-label">Median Resolution</span><span class="stat-value">{b['median_resolution_days']} days</span></div>
                    <div class="band-stat"><span class="stat-label">Avg Severity</span><span class="stat-value">{b['avg_severity']} / 5</span></div>
                    <div class="band-stat"><span class="stat-label">Total Complaints</span><span class="stat-value">{b['total_complaints']}</span></div>
                    <div class="band-stat"><span class="stat-label">Over 14 Days</span><span class="stat-value">{b['complaints_over_14_days']}</span></div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("")

    # --- Charts Row 2 ---
    st.markdown('<div class="section-header"><h3>🔥 Deep Analysis</h3><span class="badge">HEATMAP</span></div>', unsafe_allow_html=True)

    chart_col3, chart_col4 = st.columns(2)

    with chart_col3:
        resolved = df[df["status"] == "resolved"]
        heatmap_data = resolved.groupby(["neighborhood", "category"])["resolution_days"].mean().reset_index()
        heatmap_pivot = heatmap_data.pivot(index="neighborhood", columns="category", values="resolution_days").round(1)

        fig3 = px.imshow(
            heatmap_pivot,
            color_continuous_scale=["#2ed573", "#ffa502", "#ff4757"],
            labels=dict(x="Category", y="Neighborhood", color="Days"),
            aspect="auto",
            text_auto=".0f",
        )
        fig3.update_layout(
            title=dict(text="Resolution Heatmap (Category × Neighborhood)", font=dict(size=14, color="white")),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="rgba(255,255,255,0.7)", size=11),
            height=350,
            margin=dict(l=0, r=0, t=50, b=0),
        )
        st.plotly_chart(fig3, width="stretch")

    with chart_col4:
        # Severity vs Resolution scatter
        resolved_sample = resolved.copy()
        fig4 = px.scatter(
            resolved_sample,
            x="severity",
            y="resolution_days",
            color="income_band",
            color_discrete_map={"low": "#ff6b6b", "medium": "#ffa502", "high": "#2ed573"},
            opacity=0.6,
            labels={"severity": "Severity (1-5)", "resolution_days": "Resolution Days", "income_band": "Income"},
            size_max=10,
        )
        fig4.update_layout(
            title=dict(text="Severity vs Resolution Time (The Bias)", font=dict(size=14, color="white")),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="rgba(255,255,255,0.7)", size=11),
            height=350,
            xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
            legend=dict(orientation="h", yanchor="top", y=-0.2),
            margin=dict(l=0, r=0, t=50, b=0),
        )
        st.plotly_chart(fig4, width="stretch")

    # --- AI Recommendations ---
    st.markdown('<div class="section-header"><h3>🎯 AI-Generated Recommendations</h3><span class="badge">ACTION ITEMS</span></div>', unsafe_allow_html=True)

    for rec in recs_data["recommendations"][:4]:
        priority_color = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(rec["priority"], "⚪")
        with st.expander(f"{priority_color} **{rec['priority']}** — {rec['action']}", expanded=False):
            st.markdown(f"**Reason:** {rec['reason']}")
            st.markdown(f"**Expected Impact:** {rec['impact']}")


# ============================
# TAB 2: AI ASSISTANT
# ============================
with tab2:
    st.markdown("### 🤖 Ask the AI Equity Analyst")
    st.caption("Ask natural-language questions — the AI queries real data using function calling and cites specific numbers.")

    # Example questions
    st.markdown("**💡 Try these:**")
    ex_cols = st.columns(3)
    example_questions = [
        "Which neighborhoods are being underserved?",
        "Show me fairness breakdown for pothole complaints",
        "What should we prioritize next week?",
    ]

    for col, q in zip(ex_cols, example_questions):
        with col:
            if st.button(q, key=f"ex_{q[:15]}", use_container_width=True):  # noqa
                st.session_state.pending_question = q

    st.markdown("---")

    # Chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"], avatar="🧑‍💼" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])

    # Input
    user_input = st.chat_input("Ask about equity in your city...")

    if hasattr(st.session_state, "pending_question") and st.session_state.get("pending_question"):
        user_input = st.session_state.pending_question
        st.session_state.pending_question = None

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="🧑‍💼"):
            st.markdown(user_input)

        with st.chat_message("assistant", avatar="🤖"):
            if st.session_state.gemini_client is None:
                st.warning("⚠️ Enter your Gemini API key in the sidebar to enable the AI assistant.")
                response_text = "Please connect your Gemini API key."
                functions_called = []
                data_cited = {}
            else:
                with st.spinner("🔍 Analyzing data with AI..."):
                    try:
                        response_text, functions_called, data_cited, st.session_state.gemini_chat_history = run_agent_query(
                            st.session_state.gemini_client,
                            st.session_state.gemini_chat_history,
                            user_input,
                        )
                        st.markdown(response_text)

                        # Log to ledger
                        neighborhoods = []
                        for fn_data in data_cited.values():
                            if isinstance(fn_data, dict):
                                if "neighborhood_breakdown" in fn_data:
                                    neighborhoods.extend(n["neighborhood"] for n in fn_data["neighborhood_breakdown"])
                                if "by_neighborhood" in fn_data:
                                    neighborhoods.extend(fn_data["by_neighborhood"].keys())

                        st.session_state.ledger.log_decision(
                            query=user_input,
                            functions_called=functions_called,
                            data_points_cited=data_cited,
                            ai_response=response_text,
                            neighborhoods_affected=list(set(neighborhoods)),
                        )

                        if functions_called:
                            with st.expander("🔧 Tools & Data Sources Used"):
                                for fn in functions_called:
                                    st.code(f"{fn['function']}({json.dumps(fn['args'], indent=2)})", language="python")

                    except Exception as e:
                        response_text = f"Error: {str(e)}"
                        st.error(response_text)
                        functions_called = []
                        data_cited = {}

        st.session_state.chat_history.append({"role": "assistant", "content": response_text})


# ============================
# TAB 3: DECISION LEDGER
# ============================
with tab3:
    st.markdown("### 📋 Decision Audit Trail")
    st.caption("Every AI recommendation is logged with timestamps, data citations, and affected neighborhoods — ensuring full transparency and accountability.")

    ledger = st.session_state.ledger
    entries = ledger.get_entries()

    if not entries:
        st.markdown("""
        <div style="text-align: center; padding: 3rem 1rem; color: rgba(255,255,255,0.4);">
            <div style="font-size: 3rem; margin-bottom: 1rem;">📝</div>
            <div style="font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem;">No decisions logged yet</div>
            <div style="font-size: 0.85rem;">Use the AI Assistant tab to ask questions.<br>Every response will be automatically logged here with full data citations.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Summary
        lc1, lc2, lc3 = st.columns(3)
        with lc1:
            st.metric("📌 Decisions Logged", ledger.get_entry_count())
        with lc2:
            st.metric("🏘️ Neighborhoods Analyzed", len(ledger.get_neighborhoods_summary()))
        with lc3:
            st.metric("🔧 Data Queries Run", sum(len(e["functions_called"]) for e in entries))

        st.markdown("---")

        for entry in entries:
            with st.expander(f"📌 **Decision #{entry['id']}** — {entry['timestamp']}", expanded=(entry == entries[0])):
                st.markdown(f"**🗣️ Query:** {entry['query']}")
                st.markdown(f"**🤖 Response:** {entry['ai_response']}")

                if entry["neighborhoods_affected"]:
                    tags = " ".join([f"`{n}`" for n in entry["neighborhoods_affected"]])
                    st.markdown(f"**🏘️ Neighborhoods:** {tags}")

                if entry["functions_called"]:
                    st.markdown("**🔧 Functions Called:**")
                    for fn in entry["functions_called"]:
                        st.code(f"{fn['function']}({json.dumps(fn.get('args', {}), indent=2)})", language="python")

                if entry["data_points_cited"]:
                    st.markdown("**📊 Raw Data Cited:**")
                    st.json(entry["data_points_cited"])

        st.markdown("---")
        if st.button("🗑️ Clear Ledger", type="secondary"):
            ledger.clear()
            st.rerun()
