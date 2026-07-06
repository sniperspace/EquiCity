# EquiCity — Equity-Aware Decision Intelligence Platform

> *Cities resolve thousands of complaints daily — but nobody checks if that resolution is fair. EquiCity does.*

## The Problem

Modern cities collect vast amounts of citizen complaint data — potholes, broken streetlights, water leaks, garbage. These complaints get resolved, but **nobody audits whether resolution is equitable across all neighborhoods**. In practice, lower-income areas often wait 2-5x longer than affluent areas for the same severity issues. This systemic bias is invisible because no existing tool tracks it.

## Our Solution

EquiCity is an **AI-powered Equity Intelligence Platform** that:

1. **Audits fairness** — Computes a Fairness Score (0-100) for every neighborhood, comparing resolution times normalized by complaint severity
2. **Surfaces hidden bias** — Automatically detects when low-income neighborhoods are systematically underserved
3. **Explains with data** — Every AI recommendation is backed by specific numbers, not vague summaries
4. **Maintains accountability** — A Decision Ledger logs every AI query, the data it cited, and the neighborhoods affected

## Key Features

### 📊 Fairness Dashboard
- Interactive charts showing resolution time disparities by neighborhood and income band
- Heatmap of complaint categories × neighborhoods
- Real-time equity alert banners when disparities exceed thresholds

### 💬 AI Equity Analyst
- Natural language interface powered by **Gemini 2.0 Flash** with function calling
- Ask questions like *"Which neighborhoods are underserved?"* and get data-backed answers
- The AI calls real analytical functions — not just generating text

### 📋 Decision Ledger
- **Every AI decision is logged** with: timestamp, query, functions called, data cited, neighborhoods affected
- Full audit trail — click into any entry to see exactly what data justified the recommendation
- This is our **core differentiator** — an accountability layer that no traditional BI tool provides

## USP: What Makes This Different

| Aspect | Traditional BI Tools | EquiCity |
|--------|---------------------|----------|
| Focus | Efficiency metrics | **Equity and fairness** |
| Output | Static dashboards | **Auditable AI recommendations** |
| Actionability | "Here's a chart" | **"Here's the disparity, here's the fix, here's the data"** |
| Accountability | None | **Full decision ledger with data citations** |
| Accessibility | Requires data analysts | **Natural language — anyone can ask** |

## Tech Stack (Zero Cost)

| Layer | Technology | Cost |
|-------|-----------|------|
| AI | Gemini 2.0 Flash (Google AI Studio) | Free tier |
| Frontend/Backend | Streamlit | Free |
| Data | Synthetic CSV (realistic patterns) | Free |
| Charts | Plotly | Free |
| Hosting | Streamlit Community Cloud | Free |

## Setup & Run

```bash
# 1. Clone the repository
git clone <repo-url>
cd GenAI-Academy-APAC

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

Then enter your **free Gemini API key** (get one at [aistudio.google.com](https://aistudio.google.com)) in the sidebar.

## Future Roadmap

With more time and resources, EquiCity could scale to:
- **Real city data** via BigQuery integration with open government datasets
- **Multi-agent orchestration** using Vertex AI Agent Builder for automated resource reallocation
- **Predictive equity forecasting** — predicting which neighborhoods will become underserved before it happens
- **Real-time IoT integration** for live complaint tracking via Cloud Run

## Team

Built for **Gen AI Academy APAC Edition 2026**

---

*"This isn't just a dashboard — it's an accountability layer that most civic AI tools don't have."*
