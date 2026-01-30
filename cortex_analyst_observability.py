"""
AI Observability for Cortex Analyst with Semantic Views
========================================================
This script registers a Cortex Analyst application to Snowflake's native
AI Observability (AI & ML -> Evaluations) using TruLens with OTel-style instrumentation.

Uses: HAEBI_DEMO.HAEBI_SCHEMA.DUMMY_SEMANTIC_VIEW semantic view

To run in Snowflake Notebook:
1. Import this as a notebook cell or upload to a stage
2. Install packages: trulens-core, trulens-connectors-snowflake, trulens-providers-cortex
3. Execute cells sequentially

To run locally:
1. Set environment variables (see Configuration section)
2. pip install -r requirements.txt
3. python cortex_analyst_observability.py
"""

import json
import os
import requests
import pandas as pd
from typing import List, Optional

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# For local development, set these environment variables or modify directly
SNOWFLAKE_HOST = os.getenv("SNOWFLAKE_HOST", "")
SNOWFLAKE_ACCOUNT = os.getenv("SNOWFLAKE_ACCOUNT", "")
SNOWFLAKE_USER = os.getenv("SNOWFLAKE_USER", "")
SNOWFLAKE_PASSWORD = os.getenv("SNOWFLAKE_PASSWORD", "")
SNOWFLAKE_PAT = os.getenv("SNOWFLAKE_PAT", "")  # Programmatic Access Token
SNOWFLAKE_ROLE = os.getenv("SNOWFLAKE_ROLE", "ACCOUNTADMIN")
SNOWFLAKE_WAREHOUSE = os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH")
SNOWFLAKE_DATABASE = os.getenv("SNOWFLAKE_DATABASE", "HAEBI_DEMO")
SNOWFLAKE_SCHEMA = os.getenv("SNOWFLAKE_SCHEMA", "HAEBI_SCHEMA")

# Semantic View Configuration
SEMANTIC_VIEW = "HAEBI_DEMO.HAEBI_SCHEMA.DUMMY_SEMANTIC_VIEW"

# LLM Configuration
JUDGE_MODEL = "mistral-large2"
SUMMARIZATION_MODEL = "mistral-large2"

# API Endpoints
SNOWFLAKE_ACCOUNT_URL = f"https://{SNOWFLAKE_HOST}" if SNOWFLAKE_HOST else None
ANALYST_API_ENDPOINT = "/api/v2/cortex/analyst/message"
COMPLETE_API_ENDPOINT = "/api/v2/cortex/inference:complete"

# ==============================================================================
# SNOWPARK SESSION SETUP
# ==============================================================================
from snowflake.snowpark import Session

def create_snowpark_session() -> Session:
    """Create Snowpark session for TruLens and SQL execution."""
    # When running in Snowflake Notebook, use get_active_session()
    try:
        from snowflake.snowpark.context import get_active_session
        session = get_active_session()
        print("Using active Snowflake Notebook session")
        return session
    except:
        pass
    
    # For local development
    connection_params = {
        "account": SNOWFLAKE_ACCOUNT,
        "user": SNOWFLAKE_USER,
        "password": SNOWFLAKE_PASSWORD,
        "role": SNOWFLAKE_ROLE,
        "warehouse": SNOWFLAKE_WAREHOUSE,
        "database": SNOWFLAKE_DATABASE,
        "schema": SNOWFLAKE_SCHEMA,
    }
    connection_params = {k: v for k, v in connection_params.items() if v}
    session = Session.builder.configs(connection_params).create()
    print("Created local Snowpark session")
    return session

# ==============================================================================
# TRULENS SETUP WITH OTEL INSTRUMENTATION
# ==============================================================================
from trulens.core import TruSession
from trulens.connectors.snowflake import SnowflakeConnector
from trulens.apps.app import TruApp
from trulens.core.otel.instrument import instrument
from trulens.otel.semconv.trace import SpanAttributes
from trulens.core.run import Run, RunConfig

# ==============================================================================
# CORTEX ANALYST APPLICATION CLASS
# ==============================================================================
class CortexAnalystApp:
    """
    Cortex Analyst application instrumented for Snowflake AI Observability.
    
    Uses OTel-style instrumentation with SpanAttributes for native integration
    with Snowflake's Evaluations UI.
    """
    
    def __init__(self, session: Session, semantic_view: str):
        self.session = session
        self.semantic_view = semantic_view
        self.messages = []  # Conversation history for multi-turn
        
        # Get host from session for API calls
        self._setup_api_config()
    
    def _setup_api_config(self):
        """Configure API endpoints from session or environment."""
        global SNOWFLAKE_ACCOUNT_URL, SNOWFLAKE_PAT
        
        if not SNOWFLAKE_ACCOUNT_URL:
            # Try to get from session
            try:
                account_info = self.session.sql("SELECT CURRENT_ACCOUNT_LOCATOR(), CURRENT_REGION()").collect()[0]
                account_locator = account_info[0]
                region = account_info[1]
                # Construct URL (simplified - may need adjustment for your region)
                SNOWFLAKE_ACCOUNT_URL = f"https://{account_locator}.snowflakecomputing.com"
            except:
                raise ValueError("Could not determine Snowflake account URL. Set SNOWFLAKE_HOST env var.")
        
        if not SNOWFLAKE_PAT:
            raise ValueError("SNOWFLAKE_PAT environment variable required for API calls.")
    
    def _get_api_headers(self) -> dict:
        """Get authorization headers for Cortex APIs."""
        return {
            "Authorization": f"Bearer {SNOWFLAKE_PAT}",
            "X-Snowflake-Authorization-Token-Type": "PROGRAMMATIC_ACCESS_TOKEN",
            "Content-Type": "application/json",
        }
    
    @instrument(
        span_type=SpanAttributes.SpanType.RETRIEVAL,
        attributes={
            SpanAttributes.RETRIEVAL.QUERY_TEXT: "query",
            SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS: "return",
        }
    )
    def call_cortex_analyst(self, query: str) -> dict:
        """
        Call Cortex Analyst API with semantic view to get SQL interpretation.
        
        This is the RETRIEVAL step - converting natural language to SQL.
        Returns interpretation and generated SQL as context.
        """
        # Add user message to conversation
        self.messages.append({
            "role": "user",
            "content": [{"type": "text", "text": query}]
        })
        
        request_body = {
            "messages": self.messages,
            "semantic_view": self.semantic_view,
        }
        
        response = requests.post(
            f"{SNOWFLAKE_ACCOUNT_URL}{ANALYST_API_ENDPOINT}",
            json=request_body,
            headers=self._get_api_headers(),
            timeout=60,
        )
        
        if response.status_code >= 400:
            error_msg = f"Cortex Analyst API error {response.status_code}: {response.text}"
            self.messages.pop()  # Remove failed message
            raise Exception(error_msg)
        
        result = response.json()
        request_id = response.headers.get("X-Snowflake-Request-Id", "")
        
        # Store analyst response in conversation history
        if "message" in result:
            self.messages.append({**result["message"], "request_id": request_id})
        
        # Extract context for retrieval span
        context = self._extract_context(result)
        
        return {
            "raw_response": result,
            "request_id": request_id,
            "context": context,
        }
    
    def _extract_context(self, api_response: dict) -> List[str]:
        """Extract interpretation and SQL as retrieval context."""
        contexts = []
        content = api_response.get("message", {}).get("content", [])
        
        for item in content:
            if item.get("type") == "text":
                contexts.append(f"Interpretation: {item.get('text', '')}")
            elif item.get("type") == "sql":
                contexts.append(f"SQL: {item.get('statement', '')}")
        
        return contexts if contexts else ["No context generated"]
    
    def _extract_sql(self, api_response: dict) -> Optional[str]:
        """Extract SQL statement from API response."""
        content = api_response.get("message", {}).get("content", [])
        for item in content:
            if item.get("type") == "sql":
                return item.get("statement")
        return None
    
    def _extract_interpretation(self, api_response: dict) -> str:
        """Extract interpretation text from API response."""
        content = api_response.get("message", {}).get("content", [])
        for item in content:
            if item.get("type") == "text":
                return item.get("text", "")
        return "No interpretation"
    
    def execute_sql(self, sql: str) -> str:
        """Execute SQL and return results as markdown."""
        if not sql:
            return "No SQL to execute"
        
        try:
            df = self.session.sql(sql).to_pandas()
            return df.to_markdown(index=False)
        except Exception as e:
            return f"SQL execution error: {str(e)}"
    
    @instrument(span_type=SpanAttributes.SpanType.GENERATION)
    def generate_summary(self, query: str, sql_results: str) -> str:
        """
        Generate human-readable summary from SQL results using Cortex Complete.
        
        This is the GENERATION step - LLM summarization.
        """
        prompt = f"""Summarize the following SQL query results into a clear, human-readable response.

Original Question: {query}

SQL Results:
{sql_results}

Provide a concise summary that directly answers the original question."""

        payload = {
            "model": SUMMARIZATION_MODEL,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that summarizes data query results clearly and concisely."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 500,
        }
        
        headers = self._get_api_headers()
        headers["Accept"] = "application/json, text/event-stream"
        
        response = requests.post(
            f"{SNOWFLAKE_ACCOUNT_URL}{COMPLETE_API_ENDPOINT}",
            json=payload,
            headers=headers,
            timeout=60,
        )
        
        if response.status_code != 200:
            return f"Summary generation error: {response.status_code}"
        
        # Parse SSE response
        full_content = ""
        content_type = response.headers.get("Content-Type", "")
        
        if "text/event-stream" in content_type:
            for line in response.text.split("\n"):
                line = line.strip()
                if line.startswith("data: "):
                    json_str = line[6:]
                    if json_str and json_str != "[DONE]":
                        try:
                            chunk = json.loads(json_str)
                            if "choices" in chunk and chunk["choices"]:
                                delta = chunk["choices"][0].get("delta", {})
                                full_content += delta.get("content", "")
                        except json.JSONDecodeError:
                            continue
        else:
            result = response.json()
            if "choices" in result and result["choices"]:
                full_content = result["choices"][0].get("message", {}).get("content", "")
        
        return full_content or "Unable to generate summary"
    
    @instrument(
        span_type=SpanAttributes.SpanType.RECORD_ROOT,
        attributes={
            SpanAttributes.RECORD_ROOT.INPUT: "query",
            SpanAttributes.RECORD_ROOT.OUTPUT: "return",
        }
    )
    def answer_query(self, query: str) -> str:
        """
        Main entry point - returns the generated SQL for comparison with ground truth.
        
        For SQL comparison evaluation:
        - INPUT: User's natural language question
        - OUTPUT: Generated SQL query (to compare with expected SQL)
        
        This is marked as RECORD_ROOT for tracing.
        """
        # Reset conversation for clean comparison
        self.reset_conversation()
        
        # Call Cortex Analyst to get SQL
        analyst_result = self.call_cortex_analyst(query)
        
        # Extract and return the SQL (this is what we compare to ground truth)
        sql = self._extract_sql(analyst_result["raw_response"])
        
        if sql:
            # Normalize SQL for comparison (remove request_id comments, extra whitespace)
            normalized_sql = self._normalize_sql(sql)
            return normalized_sql
        else:
            # If no SQL generated, return the interpretation
            return self._extract_interpretation(analyst_result["raw_response"])
    
    def _normalize_sql(self, sql: str) -> str:
        """
        Normalize SQL for fair comparison:
        - Remove request_id comments
        - Strip extra whitespace
        - Standardize case for keywords
        """
        import re
        # Remove the Generated by Cortex Analyst comment
        sql = re.sub(r'\s*--\s*Generated by Cortex Analyst.*', '', sql)
        # Remove extra whitespace
        sql = ' '.join(sql.split())
        # Strip leading/trailing whitespace
        sql = sql.strip()
        # Remove trailing semicolon for consistency
        sql = sql.rstrip(';').strip()
        return sql
    
    def reset_conversation(self):
        """Reset conversation history for new session."""
        self.messages = []


# ==============================================================================
# EVALUATION SETUP
# ==============================================================================
def setup_evaluation(session: Session, app: CortexAnalystApp):
    """
    Register app and create evaluation run for Snowflake AI Observability.
    
    This will appear under AI & ML -> Evaluations in Snowsight.
    """
    # Create TruLens connector
    connector = SnowflakeConnector(snowpark_session=session)
    tru_session = TruSession(connector=connector)
    
    # Register the app
    tru_app = TruApp(
        app,
        app_name="CORTEX_ANALYST_SEMANTIC_VIEW",
        app_version="v1.0",
        connector=connector,
        main_method=app.answer_query,  # Entry point for tracing
    )
    
    return tru_app, connector


def create_evaluation_run(tru_app: TruApp, run_name: str = "semantic_view_eval"):
    """
    Create an evaluation run with test dataset.
    
    The dataset should have columns (matching analyst_evalset_generator.py output):
    - INPUT_QUERY: User questions (natural language)
    - OUTPUT_SQL: Expected SQL queries (ground truth from logs)
    """
    run_config = RunConfig(
        run_name=run_name,
        description="SQL comparison evaluation for Cortex Analyst with DUMMY_SEMANTIC_VIEW",
        label="cortex_analyst_sql_eval",
        source_type="TABLE",
        dataset_name="CORTEX_ANALYST_TEST_DATA",
        dataset_spec={
            "RECORD_ROOT.INPUT": "INPUT_QUERY",
            "RECORD_ROOT.GROUND_TRUTH_OUTPUT": "OUTPUT_SQL",
        },
        llm_judge_name=JUDGE_MODEL,
    )
    
    run: Run = tru_app.add_run(run_config=run_config)
    return run


# ==============================================================================
# TEST DATA CREATION
# ==============================================================================
def create_test_dataset_from_csv(session: Session, csv_path: str = None):
    """
    Create test dataset table from CSV file or use defaults.
    
    Expected CSV columns (matching analyst_evalset_generator.py output):
    - INPUT_QUERY: User questions (natural language)
    - OUTPUT_SQL: Expected SQL queries (ground truth)
    """
    import os
    
    # Try to load from CSV first (generated by analyst_evalset_generator.py)
    if csv_path is None:
        csv_path = os.path.join(os.path.dirname(__file__), "test_dataset.csv")
    
    if os.path.exists(csv_path):
        print(f"Loading test data from CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Ensure required columns exist
        if "INPUT_QUERY" not in df.columns or "OUTPUT_SQL" not in df.columns:
            print("Warning: CSV missing required columns. Expected INPUT_QUERY, OUTPUT_SQL")
            print(f"Found columns: {df.columns.tolist()}")
            # Try to map old column names
            if "QUERY" in df.columns:
                df = df.rename(columns={"QUERY": "INPUT_QUERY"})
            if "GROUND_TRUTH_SQL" in df.columns:
                df = df.rename(columns={"GROUND_TRUTH_SQL": "OUTPUT_SQL"})
    else:
        # Fallback to hardcoded test data
        print("No CSV found, using default test data")
        test_data = [
            {
                "INPUT_QUERY": "What are the total page views?",
                "OUTPUT_SQL": "SELECT * FROM SEMANTIC_VIEW(HAEBI_DEMO.HAEBI_SCHEMA.DUMMY_SEMANTIC_VIEW METRICS total_page_views)"
            },
            {
                "INPUT_QUERY": "How many total records are there?",
                "OUTPUT_SQL": "SELECT * FROM SEMANTIC_VIEW(HAEBI_DEMO.HAEBI_SCHEMA.DUMMY_SEMANTIC_VIEW METRICS total_records)"
            },
            {
                "INPUT_QUERY": "Show page views by browser",
                "OUTPUT_SQL": "SELECT * FROM SEMANTIC_VIEW(HAEBI_DEMO.HAEBI_SCHEMA.DUMMY_SEMANTIC_VIEW METRICS total_page_views DIMENSIONS browser)"
            },
            {
                "INPUT_QUERY": "How does traffic vary by loyalty type?",
                "OUTPUT_SQL": "SELECT * FROM SEMANTIC_VIEW(HAEBI_DEMO.HAEBI_SCHEMA.DUMMY_SEMANTIC_VIEW METRICS total_page_views DIMENSIONS loyalty_type)"
            },
        ]
        df = pd.DataFrame(test_data)
    
    # Only keep the columns we need
    df = df[["INPUT_QUERY", "OUTPUT_SQL"]].copy()
    
    print(f"Test data preview:")
    print(df.head(3).to_string())
    
    # Create table in Snowflake
    session.write_pandas(
        df,
        table_name="CORTEX_ANALYST_TEST_DATA",
        auto_create_table=True,
        overwrite=True,
    )
    
    print(f"\nCreated test dataset with {len(df)} questions")
    return df


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    """
    Main execution flow:
    1. Create Snowpark session
    2. Initialize Cortex Analyst app
    3. Register with TruLens for AI Observability
    4. Create test dataset
    5. Run evaluation
    6. Compute metrics
    """
    print("=" * 60)
    print("AI Observability for Cortex Analyst with Semantic Views")
    print("=" * 60)
    
    # Step 1: Create session
    print("\n[1/6] Creating Snowpark session...")
    session = create_snowpark_session()
    
    # Step 2: Initialize app
    print("\n[2/6] Initializing Cortex Analyst app...")
    app = CortexAnalystApp(session=session, semantic_view=SEMANTIC_VIEW)
    
    # Test single query first
    print("\n[2b/6] Testing single query...")
    app.reset_conversation()
    test_response = app.answer_query("What are the total page views?")
    print(f"Test response: {test_response[:200]}...")
    
    # Step 3: Register with TruLens
    print("\n[3/6] Registering app with TruLens...")
    tru_app, connector = setup_evaluation(session, app)
    
    # Step 4: Create test dataset from CSV (uses analyst_evalset_generator.py output)
    print("\n[4/6] Creating test dataset from CSV...")
    create_test_dataset_from_csv(session)
    
    # Step 5: Create and run evaluation
    print("\n[5/6] Creating evaluation run...")
    import time
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run = create_evaluation_run(tru_app, run_name=f"sql_eval_{timestamp}")
    
    print("\n[5b/7] Starting evaluation run (this may take a few minutes)...")
    run.start()
    
    # Check invocation status
    print("\n[5c/7] Checking invocation status...")
    run.describe()
    
    # Step 6: Wait for invocation to complete before computing metrics
    print("\n[6/7] Waiting for invocation to complete...")
    
    # Must wait for INVOCATION_COMPLETED before computing metrics
    max_invocation_wait = 120  # 2 minutes
    poll_interval = 10
    
    for i in range(max_invocation_wait // poll_interval):
        time.sleep(poll_interval)
        print(f"\n--- Invocation status check {i+1} ---")
        try:
            run.describe()
        except Exception as e:
            print(f"Status check error: {e}")
    
    # Step 7: Compute metrics
    print("\n[7/7] Computing evaluation metrics...")
    print("Requesting metrics: correctness, answer_relevance, coherence")
    print("\nRequired attributes for these metrics:")
    print("  - correctness: RECORD_ROOT.INPUT, RECORD_ROOT.OUTPUT, RECORD_ROOT.GROUND_TRUTH_OUTPUT")
    print("  - answer_relevance: RECORD_ROOT.INPUT, RECORD_ROOT.OUTPUT")
    print("  - coherence: RECORD_ROOT.OUTPUT")
    
    try:
        # Compute metrics - this triggers async background job
        result = run.compute_metrics([
            "correctness",         # Compares generated SQL to ground truth SQL
            "answer_relevance",    # Does the SQL address the question?
            "coherence",           # Is the output well-structured?
        ])
        print(f"\ncompute_metrics() returned: {result}")
    except Exception as e:
        print(f"\nERROR calling compute_metrics(): {e}")
        import traceback
        traceback.print_exc()
    
    # Wait for metrics computation to complete
    print("\n" + "=" * 60)
    print("METRICS COMPUTATION STARTED")
    print("=" * 60)
    print("\nMetrics are computed asynchronously by Snowflake.")
    print("This can take 2-5 minutes depending on dataset size.")
    print("Status should change to COMPUTATION_IN_PROGRESS then COMPLETED")
    print("\nPolling for completion...")
    
    # Poll for completion - looking for COMPLETED status
    max_wait_minutes = 5
    poll_interval_seconds = 15
    max_polls = (max_wait_minutes * 60) // poll_interval_seconds
    
    for i in range(max_polls):
        time.sleep(poll_interval_seconds)
        print(f"\n--- Metrics status check {i+1}/{max_polls} (waited {(i+1)*poll_interval_seconds}s) ---")
        try:
            run.describe()
        except Exception as e:
            print(f"Status check error: {e}")
    
    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    print("\nView results in Snowsight:")
    print("  AI & ML -> Evaluations -> CORTEX_ANALYST_SEMANTIC_VIEW")
    print(f"\nRun name: sql_eval_{timestamp}")
    print("\nIf metrics are still not visible:")
    print("  1. Refresh the Evaluations page in Snowsight")
    print("  2. Click on the specific run to see metric details")
    print("  3. Check that LLM judge model has access permissions")
    print("\nFinal run status:")
    run.describe()


if __name__ == "__main__":
    main()
