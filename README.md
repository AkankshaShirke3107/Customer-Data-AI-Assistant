<div align="center">

# Customer Data AI Assistant

<p align="center">
  <img src="R.webp" alt="Customer Data AI Assistant Banner" width="100%">
</p>

**Natural language analytics for Excel data — with deterministic, hallucination-free results.**

Ask questions about your customer data in plain English. Get exact answers computed by Pandas, summarized by Gemini.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/streamlit-1.38+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Google Gemini](https://img.shields.io/badge/gemini-2.5--flash-4285F4?style=flat-square&logo=google&logoColor=white)](https://ai.google.dev)
[![License: MIT](https://img.shields.io/badge/license-MIT-22C55E?style=flat-square)](LICENSE)

[Getting Started](#getting-started) · [Architecture](#architecture) · [How It Works](#how-it-works) · [Supported Queries](#supported-queries)

</div>

---

## Overview

Business teams spend hours wrestling with pivot tables, VLOOKUP chains, and complex BI dashboards to answer straightforward questions about their data: _"How many premium customers are in Pune?"_ or _"What is the average deal size this quarter?"_

Large Language Models promise to solve this through natural language — but LLMs are unreliable at math. Feed raw data into a standard model and it will frequently hallucinate aggregations, invent statistics, and deliver confidently incorrect numbers.

**Customer Data AI Assistant** takes a different approach. It strictly separates language understanding from mathematical computation:

- **Google Gemini** classifies your question into a structured intent (the _what_).
- **Pandas** executes the actual computation deterministically (the _how_).
- **Gemini** then phrases the already-computed result as a natural language response.

The LLM never sees your raw data. It never computes a number. Every value in every answer is produced by a CPU executing DataFrame operations — not by a neural network guessing.

If Gemini is unavailable, rate-limited, or unconfigured, the application continues to operate using a built-in rule-based intent parser. No external dependency is required for the system to function.

---

## Key Features

| Feature | Description | Status |
|:--------|:------------|:------:|
| Dynamic Schema Detection | Automatically maps any Excel column to semantic roles (budget, location, status, type) without hardcoded names | Stable |
| Natural Language Queries | Supports counts, sums, averages, filters, sorts, groupings, ranges, top-N, and multi-condition queries | Stable |
| Zero-Hallucination Engine | Strict architectural separation ensures all numbers come from Pandas, never from the LLM | Stable |
| Execution Transparency | Every answer includes a full audit trail: operation, columns, rows scanned, execution time, and raw intent | Stable |
| Auto-Visualization | Dynamically selects the appropriate Plotly chart type based on the shape and semantics of the result | Stable |
| Conversational Context | Follow-up questions inherit filters from previous queries for multi-turn analysis | Stable |
| Rule-Based Fallback | Continues operating via keyword and regex parsing when the Gemini API is unavailable | Stable |
| One-Click Export | Download any filtered result as CSV or any chart as PNG directly from the interface | Stable |
| AI-Generated Insights | Automatically surfaces key patterns and statistics from uploaded datasets | Stable |
| Dark Theme UI | Premium interface with dark mode, smooth transitions, and responsive layout | Stable |

---

## Architecture

<p align="center">
  <img src="architecture.png" alt="Customer Data AI Assistant — Production Architecture Diagram" width="100%">
</p>

> **Design constraint:** Raw data never leaves the local process. Only column names and schema metadata are sent to the LLM for intent classification.

---

## Project Structure

```
customer-data-ai-assistant/
├── app.py                 Main application — UI layout, chat, rendering
├── config.py              Centralized configuration and constants
├── utils.py               File loading, validation, schema detection, profiling
├── query_engine.py        Deterministic Pandas execution engine (17 operations)
├── gemini_helper.py       Gemini API integration with retry, timeout, validation
├── charts.py              Plotly chart selection and dark-theme styling
├── requirements.txt       Pinned dependency versions
├── .env.example           Environment variable template
├── .gitignore             Security and build exclusions
└── data/
    └── sample_leads.xlsx  Bundled dataset for immediate testing
```

| Module | Responsibility |
|:-------|:---------------|
| `config.py` | All tunables in one place — model name, timeouts, retry counts, upload limits, UI constants |
| `utils.py` | Server-side file validation, Excel loading, dynamic column classification, dataset profiling |
| `query_engine.py` | Maps structured intents to Pandas operations. Contains 17 deterministic handlers and a rule-based fallback parser |
| `gemini_helper.py` | Singleton SDK configuration, retry logic with configurable timeout, intent validation against an allowlist |
| `charts.py` | Automatic chart type selection (bar, pie, histogram, box) with a dark-theme-compatible color system |

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- A Google Gemini API key ([get one free](https://aistudio.google.com/app/apikey)) — optional, the app works without it

### Installation

```bash
git clone https://github.com/AkankshaShirke3107/Customer-Data-AI-Assistant.git
cd Customer-Data-AI-Assistant

python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### Configuration

Create a `.env` file from the provided template:

```bash
cp .env.example .env
```

```ini
# .env
GEMINI_API_KEY=your_api_key_here

# Optional overrides
# GEMINI_MODEL=gemini-2.5-flash
# GEMINI_TIMEOUT_SECONDS=30
# MAX_UPLOAD_SIZE_MB=50
# LOG_LEVEL=INFO
```

| Variable | Required | Default | Description |
|:---------|:---------|:--------|:------------|
| `GEMINI_API_KEY` | No | — | Google Gemini API key. App works without it via fallback engine. |
| `GEMINI_MODEL` | No | `gemini-2.5-flash` | Model used for intent classification and summarization |
| `GEMINI_TIMEOUT_SECONDS` | No | `30` | API call timeout in seconds |
| `MAX_UPLOAD_SIZE_MB` | No | `50` | Maximum file upload size |
| `LOG_LEVEL` | No | `INFO` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |

### Running Locally

```bash
streamlit run app.py
```

Open `http://localhost:8501`. Enable **"Use sample dataset"** in the sidebar to explore immediately, or upload your own `.xlsx` file.

---

## How It Works

### 1. Upload and Schema Detection

When a file is uploaded, `utils.py` performs server-side validation (extension, file size), loads it via OpenPyXL, and runs dynamic schema detection. Column names are classified into semantic roles — budget, location, property type, status, contact, date — using keyword matching against the column headers. No column names are hardcoded.

### 2. Intent Classification

The user's natural language question is sent to Gemini along with the column names and detected schema (never the raw data). Gemini returns a structured JSON intent:

```json
{
  "operation": "greater_than",
  "column": "Budget (INR)",
  "value": 9000000,
  "conditions": [
    { "column": "Preferred Location", "op": "eq", "value": "Pune" }
  ]
}
```

The intent is validated against a strict allowlist of 17 operations before execution. If Gemini is unavailable, a rule-based parser extracts the operation from keywords and regex patterns.

### 3. Deterministic Execution

`query_engine.py` maps the validated intent to the corresponding Pandas operation:

```python
# What actually runs for "customers in Pune with budget above 90 lakhs"
df[df["Preferred Location"] == "Pune"]["Budget (INR)"] > 9000000
```

The engine tracks execution metadata: rows scanned, rows matched, filters applied, and wall-clock execution time in milliseconds.

### 4. Summarization

The raw Pandas result (the computed number or filtered DataFrame) is passed to Gemini with explicit instructions to phrase it naturally without altering any values. If Gemini is unavailable, a template-based summary is generated locally.

### 5. Visualization

`charts.py` inspects the result shape and operation type to select the appropriate chart: bar charts for rankings and groupings, pie charts for distributions, histograms for numeric spreads, box plots for comparisons. All charts use a dark-theme-compatible color palette.

### 6. Audit Trail

Every response includes an expandable execution details panel showing the full pipeline: intent classification method, exact Pandas operation, columns accessed, row counts, execution time, and the raw JSON intent.

---

## Supported Queries

| Category | Example Query | Operation |
|:---------|:-------------|:----------|
| Count | "How many customers are there?" | `count` |
| Count with filter | "How many customers are in Pune?" | `count` with condition |
| Average | "What is the average budget?" | `average` |
| Sum | "What is the total budget for Kharadi?" | `sum` with condition |
| Maximum | "Who has the highest budget?" | `max` |
| Minimum | "What is the lowest budget?" | `min` |
| Greater than | "Show customers with budget above 90 lakhs" | `greater_than` |
| Less than | "List customers with budget under 50 lakhs" | `less_than` |
| Range | "Customers with budget between 80 and 120 lakhs" | `between` |
| Top N | "Top 5 customers by budget" | `topn` |
| Bottom N | "Bottom 3 by budget" | `bottomn` |
| Group by | "Average budget by location" | `groupby` + mean |
| Distribution | "Breakdown of lead status" | `groupby` + count |
| Sort | "Sort customers by budget descending" | `sort` |
| Unique values | "What are all the locations?" | `unique` |
| Distinct count | "How many unique locations?" | `distinct_count` |
| Statistics | "Describe the budget column" | `describe` |
| Filter + list | "Show 2BHK customers in Baner" | `list` with conditions |

### Conversational Follow-ups

The engine supports multi-turn context. Filters from a previous query carry forward when the follow-up implies continuation:

```
> "Show me Pune customers"
  → 45 rows

> "Only those above 90 lakhs"
  → 12 rows (inherits Pune filter)

> "Sort them by budget"
  → 12 rows sorted descending (inherits both filters)
```

---

## Screenshots

> Replace these placeholders with actual screenshots after running the application.

| View | Description |
|:-----|:------------|
| `screenshots/hero.png` | Landing page with hero section and status badges |
| `screenshots/dashboard.png` | KPI cards and key statistics after dataset upload |
| `screenshots/chat.png` | Chat interface with user/AI messages and confidence badge |
| `screenshots/pipeline.png` | Execution details panel showing the audit trail timeline |
| `screenshots/charts.png` | Auto-generated Plotly visualization with dark theme |
| `screenshots/insights.png` | AI-generated insights section |

---

## Technical Highlights

### Modular Architecture

Each module has a single responsibility. The UI (`app.py`) never touches Pandas directly — it delegates to `query_engine.py`. Gemini calls are centralized in `gemini_helper.py` with a singleton configuration guard. Charts are decoupled in `charts.py`. All constants live in `config.py`.

### Caching

Streamlit's `@st.cache_data` is used with content-hash keys (computed from DataFrame shape, columns, and dtypes) rather than hashing the full DataFrame. Gemini responses are cached with a configurable TTL (default: 1 hour).

### Error Handling

Every layer has fallback behavior. If Gemini fails, the rule-based parser handles intent classification. If summarization fails, a template-based summary is generated. If chart generation fails, the response is returned without a visualization. The application never crashes on a bad query.

### Logging

Structured Python `logging` is configured across all modules. API call latency, response sizes, query execution times, and error states are logged for observability.

### Prompt Engineering

System prompts enforce JSON-only output from Gemini. The model receives column names and schema metadata — never raw data rows. Intent validation rejects any operation not in the 17-item allowlist, preventing prompt injection from producing unauthorized operations.

### Hallucination Prevention

This is the core architectural decision. The system is designed so that it is structurally impossible for the LLM to produce a numeric answer:

1. Gemini outputs a JSON intent (operation + column + conditions).
2. The intent is validated against a strict allowlist.
3. Pandas executes the operation deterministically.
4. Gemini receives only the computed result for natural language phrasing.

At no point does the LLM have access to raw data or the ability to compute values.

### Security

- File uploads are validated server-side for extension and size (configurable, default 50 MB)
- API keys are loaded from `.env` files, never committed to version control
- `.gitignore` prevents secrets, caches, and IDE files from leaking
- Gemini SDK is configured once via a singleton guard
- API calls use configurable timeouts and automatic retries

---

## Design Philosophy

Traditional LLM-powered data tools send raw CSVs to the model and ask it to compute answers. This approach is fundamentally unreliable:

| Concern | LLM Computation | Pandas Computation |
|:--------|:----------------|:-------------------|
| Accuracy | Probabilistic, prone to hallucination | Deterministic, exact |
| Reproducibility | May vary between calls | Identical for identical inputs |
| Auditability | Black box | Full execution trace available |
| Cost per query | API tokens for data processing | Zero — local CPU |
| Latency | Seconds (network + inference) | Milliseconds |
| Data privacy | Raw data sent to external API | Data never leaves the process |

This application uses Gemini for what language models are genuinely good at — understanding natural language intent and generating fluent prose — while delegating computation to a tool purpose-built for it.

---

## Future Improvements

| Priority | Improvement | Description |
|:---------|:------------|:------------|
| High | Automated tests | pytest suite covering all 17 query engine operations |
| High | CI/CD pipeline | GitHub Actions for linting, testing, and deployment |
| Medium | Multi-sheet support | Load multiple sheets with cross-sheet join detection |
| Medium | Database connectors | Direct connections to PostgreSQL, Snowflake, BigQuery |
| Medium | Streaming responses | Token-by-token Gemini output with typewriter effect |
| Low | User-editable schema | Manual override for auto-detected column roles |
| Low | Persistent history | SQLite-backed chat history across sessions |
| Low | Natural language charts | "Show me a pie chart of locations" |

---

## Contributing

Contributions are welcome. Please open an issue to discuss proposed changes before submitting a pull request.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a pull request

Please ensure your code follows the existing style conventions and includes appropriate docstrings.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<div align="center">

**Customer Data AI Assistant**

Built with [Streamlit](https://streamlit.io) · [Pandas](https://pandas.pydata.org) · [Plotly](https://plotly.com) · [Google Gemini](https://ai.google.dev)

</div>
