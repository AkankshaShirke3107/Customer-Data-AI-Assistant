# 📊 Customer Data AI Assistant

**Chat with your Excel data using Natural Language.**

A Streamlit application that lets you upload any customer-style Excel
spreadsheet and ask plain-English questions about it — counts, averages,
filters, top-N, group-bys, and more — with **zero hallucination**: every
number in every answer is computed by pandas, never guessed by the AI.

Google Gemini is used only to (1) understand what you're asking and
(2) phrase the already-computed pandas result as a friendly sentence.
It never invents a number on its own.

---

## ✨ Features

- **Dynamic schema detection** — no hardcoded column names. The app inspects
  whatever spreadsheet you upload and classifies columns into semantic roles
  (name, budget/price, location, property/category type, status, contact,
  dates) using keyword + statistical heuristics. Works on the bundled Pune
  real-estate leads sample *and* on differently-shaped customer sheets.
- **Natural language Q&A** over your data: counts, sums, averages, min/max,
  filters, sort, top-N / bottom-N, between-ranges, greater/less-than,
  group-by, unique values, distinct counts.
- **Zero-hallucination query engine** — Gemini only maps your question to a
  structured JSON "intent"; pandas executes it; Gemini (or a deterministic
  fallback) phrases the final sentence from the *real* computed result.
- **Rule-based fallback parser** — the app keeps working even without a
  Gemini API key (or if a Gemini call fails), using a keyword/regex-based
  intent parser as a safety net.
- **Automatic dataset profiling** — rows, columns, dtypes, missing values,
  duplicates, numeric vs categorical columns, per-column samples.
- **AI insights panel** — plain-English bullet insights generated from
  pandas-computed facts (never invented).
- **Auto-visualizations** — bar / pie / histogram / box charts, chosen
  automatically based on the question and result shape (Plotly).
- **ChatGPT-style chat UI** with history, suggested questions, confidence
  badges, and a "How was this answer calculated?" transparency panel.
- **Downloads** — export any answer's filtered data (CSV) and its summary
  (TXT).
- **Dark mode**, responsive layout, metric cards, loading states.
- **Sample dataset toggle** — try the app instantly with the bundled sample
  without uploading anything.

---

## 🏗️ Architecture

```
User
  │
  ▼
Streamlit UI (app.py)
  │
  ▼
Excel Upload ──▶ Pandas DataFrame ──▶ Dynamic Schema Detection (utils.py)
  │
  ▼
Natural-language Question
  │
  ▼
Gemini: Intent Understanding (gemini_helper.py)   [structured JSON only]
  │
  ▼
Pandas Query Execution (query_engine.py)          [ground-truth result]
  │
  ▼
Gemini: Result Summarization (gemini_helper.py)    [phrasing only]
  │
  ▼
Plotly Auto-Visualization (charts.py)
  │
  ▼
Premium Streamlit UI (chat bubbles, charts, downloads, explanations)
```

If Gemini is unavailable at any step, the app transparently falls back to
a deterministic rule-based intent parser and a template-based summary, so
the tool never breaks.

---

## 📁 Project Structure

```
customer-ai/
├── app.py              # Streamlit UI — entry point
├── utils.py            # Data loading, profiling, dynamic schema detection
├── query_engine.py      # Deterministic pandas execution + rule-based fallback parser
├── gemini_helper.py     # All Gemini API calls (intent + summarization + insights)
├── charts.py            # Automatic Plotly chart selection
├── requirements.txt
├── .env.example
├── README.md
└── data/
    └── sample_leads.xlsx  # Bundled sample dataset (Pune real-estate leads)
```

---

## 🚀 Installation

```bash
git clone <this-repo-or-copy-this-folder>
cd customer-ai
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 🔑 Environment Variables

Copy `.env.example` to `.env` and add your Gemini API key
(get one free at https://aistudio.google.com/app/apikey):

```bash
cp .env.example .env
```

```
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.0-flash   # optional override
```

> The app also lets you paste an API key directly into the sidebar for a
> quick session-only test, if you'd rather not create a `.env` file.
> Without any key, the app still runs fully using its rule-based fallback
> engine for parsing and summarizing.

## ▶️ How to Run

```bash
streamlit run app.py
```

Then open the printed local URL (typically http://localhost:8501) in your
browser. Check **"Use bundled sample dataset"** in the sidebar to try it
immediately, or upload your own `.xlsx` file.

---

## 💬 Example Queries

- "How many customers have a budget above 90 lakhs?"
- "List customers looking for 2BHK in Kharadi"
- "Which location has the highest average budget?"
- "Give me all customers interested in Baner"
- "Show me customers whose budget is between 80 and 120 lakhs"
- "What is the average budget?"
- "Top 5 customers by budget"
- "How many unique locations are there?"
- "Give me a breakdown of last call status"
- "Average budget by location"

---

## 🧠 How the No-Hallucination Guarantee Works

1. Gemini receives the question, the real column names, and the detected
   schema — and returns **only** a structured JSON object naming an
   operation (`count`, `average`, `groupby`, ...) and which columns/values
   it thinks apply.
2. `query_engine.py` executes that operation with pandas against the
   actual dataframe. This is the only place a number is ever produced.
3. Gemini is called a second time, and shown **only the already-computed
   result**, to phrase it as a natural sentence. It is explicitly
   instructed not to introduce new numbers.
4. Every chat answer has a "How was this answer calculated?" panel
   showing the exact operation, columns used, and parsed intent — so you
   can audit every answer.
5. If Gemini is unavailable or returns something unparsable at either
   step, a deterministic rule-based parser / template summary is used
   instead — the app degrades gracefully rather than guessing.

---

## 🛠️ Tech Stack

| Layer            | Technology              |
|-------------------|--------------------------|
| UI                | Streamlit                |
| Data              | Pandas, OpenPyXL         |
| AI (language only)| Google Gemini API        |
| Charts            | Plotly                   |
| Config            | python-dotenv            |

---

## 🖼️ Screenshots

> _Add screenshots here after running the app locally:_
- `docs/screenshot-overview.png` — Dataset overview + AI insights
- `docs/screenshot-chat.png` — Chat interface with an answered question
- `docs/screenshot-chart.png` — Auto-generated chart for a group-by query

---

## 🔮 Future Improvements

- Multi-file / multi-sheet support with cross-sheet joins
- User-editable schema overrides (manually re-map a detected column)
- Persistent chat history across sessions (SQLite-backed)
- Streaming token-by-token Gemini responses
- Role-based access control for shared team deployments
- Support for CSV/Google Sheets as additional input sources
- Natural-language chart requests ("show me a pie chart of X")

---

## ⚠️ Notes

- This project ships with a bundled sample dataset
  (`data/sample_leads.xlsx`, 300 rows of Pune real-estate leads) purely to
  let reviewers try the app without needing their own file. The schema
  detector was built to generalize to differently-named columns, not to
  assume this exact sheet.
- Contact/phone numbers in the sample data are synthetic and used only to
  demonstrate the "contact column" schema-detection role.
