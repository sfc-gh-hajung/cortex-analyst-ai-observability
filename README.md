# Cortex Analyst AI Observability

Evaluate and monitor Snowflake Cortex Analyst using Snowflake's native AI Observability (AI & ML → Evaluations in Snowsight).

This project enables SQL comparison evaluation: compare Cortex Analyst's generated SQL against expected ground truth SQL using built-in LLM-as-a-judge metrics.

## Features

- **Native Snowflake Integration** - Results appear directly in Snowsight under AI & ML → Evaluations
- **SQL Comparison Evaluation** - Compare generated SQL against expected SQL (ground truth)
- **Built-in Metrics** - Uses Snowflake's `correctness`, `answer_relevance`, and `coherence` metrics
- **Test Dataset Generator** - Streamlit app to build evaluation datasets from Cortex Analyst logs
- **OTel-style Instrumentation** - Uses TruLens v2.x with SpanAttributes for tracing

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Evaluation Flow                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Generate Test Dataset (analyst_evalset_generator.py)        │
│     └── Query SNOWFLAKE.LOCAL.CORTEX_ANALYST_REQUESTS()         │
│     └── Export INPUT_QUERY + OUTPUT_SQL pairs                   │
│                                                                 │
│  2. Run Evaluation (cortex_analyst_observability.py)            │
│     └── Load test dataset                                       │
│     └── Call Cortex Analyst for each query                      │
│     └── Compare generated SQL to ground truth                   │
│     └── Compute metrics (correctness, relevance, coherence)     │
│                                                                 │
│  3. View Results in Snowsight                                   │
│     └── AI & ML → Evaluations → CORTEX_ANALYST_SEMANTIC_VIEW    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Prerequisites

- Snowflake account with Cortex Analyst access
- A semantic view (e.g., `HAEBI_DEMO.HAEBI_SCHEMA.DUMMY_SEMANTIC_VIEW`)
- Programmatic Access Token (PAT) for API calls
- Python 3.8+

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/cortex-analyst-ai-observability.git
cd cortex-analyst-ai-observability

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your Snowflake credentials
```

## Configuration

Edit `.env` with your Snowflake credentials:

```bash
SNOWFLAKE_ACCOUNT=your_account_identifier
SNOWFLAKE_USER=your_username
SNOWFLAKE_PASSWORD=your_password
SNOWFLAKE_HOST=your_account.snowflakecomputing.com
SNOWFLAKE_ROLE=ACCOUNTADMIN
SNOWFLAKE_WAREHOUSE=COMPUTE_WH
SNOWFLAKE_DATABASE=HAEBI_DEMO
SNOWFLAKE_SCHEMA=HAEBI_SCHEMA
SNOWFLAKE_PAT=your_programmatic_access_token
```

### Getting a Programmatic Access Token (PAT)

1. Go to Snowsight → Admin → Security → Programmatic Access Tokens
2. Click "Generate Token"
3. Copy the token to your `.env` file

## Usage

### Option 1: Generate Test Dataset from Analyst Logs

Use the Streamlit app to create evaluation datasets from historical Cortex Analyst queries:

```bash
streamlit run analyst_evalset_generator.py
```

This app:
- Queries `SNOWFLAKE.LOCAL.CORTEX_ANALYST_REQUESTS()` for historical queries
- Lets you review, edit, and filter records
- Exports to Snowflake table or CSV

### Option 2: Use Existing Test Dataset

Edit `test_dataset.csv` with your test cases:

```csv
INPUT_QUERY,OUTPUT_SQL
"What are the total page views?","SELECT * FROM SEMANTIC_VIEW(MY_DB.MY_SCHEMA.MY_VIEW METRICS total_page_views)"
"Show data by country","SELECT * FROM SEMANTIC_VIEW(MY_DB.MY_SCHEMA.MY_VIEW METRICS total_page_views DIMENSIONS country)"
```

### Run Evaluation

```bash
python cortex_analyst_observability.py
```

The script will:
1. Load the test dataset
2. Call Cortex Analyst for each query
3. Compare generated SQL to ground truth
4. Compute metrics using LLM-as-a-judge
5. Store results in Snowflake

### View Results

1. Go to Snowsight
2. Navigate to **AI & ML → Evaluations**
3. Click on **CORTEX_ANALYST_SEMANTIC_VIEW**
4. Select your run (e.g., `sql_eval_20240129_141523`)

## Metrics

| Metric | Description | Required Attributes |
|--------|-------------|---------------------|
| `correctness` | How aligned is the output with ground truth? | INPUT, OUTPUT, GROUND_TRUTH_OUTPUT |
| `answer_relevance` | Does the output address the question? | INPUT, OUTPUT |
| `coherence` | Is the output well-structured and logical? | OUTPUT |

## Project Structure

```
cortex_analyst_ai_observability/
├── cortex_analyst_observability.py   # Main evaluation script
├── analyst_evalset_generator.py      # Streamlit app for dataset generation
├── test_dataset.csv                  # Sample test data
├── requirements.txt                  # Python dependencies
├── .env.example                      # Environment template
├── .gitignore                        # Git ignore rules
├── LICENSE                           # Apache 2.0 license
└── README.md                         # This file
```

## Customization

### Change the Semantic View

Edit `cortex_analyst_observability.py`:

```python
SEMANTIC_VIEW = "YOUR_DATABASE.YOUR_SCHEMA.YOUR_SEMANTIC_VIEW"
```

### Change the LLM Judge

```python
JUDGE_MODEL = "mistral-large2"  # or "llama3.1-70b", "claude-3-5-sonnet", etc.
```

### Add Custom Metrics

The following built-in metrics are available:
- `correctness`
- `answer_relevance`
- `coherence`
- `groundedness`
- `context_relevance`

## Troubleshooting

### Metrics not appearing in UI

1. **Check run status** - Must reach `COMPLETED` status
2. **Unique run names** - Each run needs a unique name (timestamps added automatically)
3. **Wait for async completion** - Metrics compute asynchronously (2-5 minutes)

### API errors

1. Verify PAT token is valid and not expired
2. Check role has CORTEX_USER database role
3. Ensure semantic view exists and is accessible

### Permission errors

Required privileges:
- `CORTEX_USER` database role
- `CREATE EXTERNAL AGENT` on schema
- `CREATE TASK` on schema
- `EXECUTE TASK` on account

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.

## Resources

- [Snowflake AI Observability Documentation](https://docs.snowflake.com/en/user-guide/snowflake-cortex/ai-observability)
- [TruLens Documentation](https://www.trulens.org/)
- [Cortex Analyst Documentation](https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-analyst)
