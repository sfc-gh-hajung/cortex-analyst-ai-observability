import streamlit as st
import pandas as pd
from snowflake.snowpark import Session
import os
from dotenv import load_dotenv
from typing import Optional, List
import json
import traceback
from datetime import datetime

load_dotenv()

if 'dataset' not in st.session_state:
    st.session_state.dataset = None
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = 0
if 'semantic_model_type' not in st.session_state:
    st.session_state.semantic_model_type = None
if 'semantic_model_name' not in st.session_state:
    st.session_state.semantic_model_name = None

@st.cache_resource
def get_snowflake_connection():
    try:
        session = Session.get_active_session()
        if session is None:
            raise ValueError("Session is None")
        session.sql("SELECT 1").collect()
        return session
    except Exception:
        try:
            connection_parameters = {
                "account": os.getenv("SNOWFLAKE_ACCOUNT"),
                "user": os.getenv("SNOWFLAKE_USER"),
                "password": os.getenv("SNOWFLAKE_PASSWORD"),
                "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
                "database": os.getenv("SNOWFLAKE_DATABASE", "SNOWFLAKE"),
                "schema": os.getenv("SNOWFLAKE_SCHEMA", "LOCAL"),
                "role": os.getenv("SNOWFLAKE_ROLE", "ACCOUNTADMIN")
            }
            return Session.builder.configs(connection_parameters).create()
        except Exception as e:
            st.error(f"Connection failed: {e}")
            return None

@st.cache_data(ttl=300)
def get_semantic_views(_session) -> pd.DataFrame:
    try:
        _session.sql('SHOW SEMANTIC VIEWS IN ACCOUNT').collect()
        views_df = _session.sql('SELECT * FROM TABLE(RESULT_SCAN(LAST_QUERY_ID()))').to_pandas()
        if not views_df.empty:
            views_df['fully_qualified_name'] = (
                views_df["database_name"] + "." + 
                views_df["schema_name"] + "." + 
                views_df["name"]
            )
        return views_df
    except Exception as e:
        st.warning(f"Could not fetch semantic views: {e}")
        return pd.DataFrame()

def query_analyst_logs_table_function(_session, model_type: str, model_name: str) -> pd.DataFrame:
    query = f"""
    SELECT 
        TIMESTAMP,
        USER_NAME,
        LATEST_QUESTION AS INPUT_QUERY,
        GENERATED_SQL AS OUTPUT_SQL,
        REQUEST_BODY,
        RESPONSE_BODY,
        WARNINGS
    FROM TABLE(
        SNOWFLAKE.LOCAL.CORTEX_ANALYST_REQUESTS(
            '{model_type}',
            '{model_name}'
        )
    )
    ORDER BY TIMESTAMP DESC
    """
    try:
        df = _session.sql(query).to_pandas()
        return df
    except Exception as e:
        st.error(f"Error querying logs via table function: {e}")
        return pd.DataFrame()

def query_analyst_logs_view(_session, limit: int = 1000) -> pd.DataFrame:
    query = f"""
    SELECT 
        TIMESTAMP,
        USER_NAME,
        SEMANTIC_MODEL_NAME,
        LATEST_QUESTION AS INPUT_QUERY,
        GENERATED_SQL AS OUTPUT_SQL,
        REQUEST_BODY,
        RESPONSE_BODY,
        WARNINGS
    FROM SNOWFLAKE.LOCAL.CORTEX_ANALYST_REQUESTS_V
    ORDER BY TIMESTAMP DESC
    LIMIT {limit}
    """
    try:
        df = _session.sql(query).to_pandas()
        return df
    except Exception as e:
        st.error(f"Error querying logs view: {e}")
        return pd.DataFrame()

def process_analyst_logs(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    
    processed = df.copy()
    
    processed = processed[processed['INPUT_QUERY'].notna() & (processed['INPUT_QUERY'] != '')]
    processed = processed[processed['OUTPUT_SQL'].notna() & (processed['OUTPUT_SQL'] != '')]
    
    processed = processed.drop_duplicates(subset=['INPUT_QUERY'])
    
    return processed

def validate_table_name(table_name: str) -> bool:
    parts = table_name.strip().split('.')
    return len(parts) == 3 and all(part.strip() for part in parts)

st.title("Cortex Analyst Evaluation Dataset Builder")
st.caption("Build evaluation datasets from Cortex Analyst request logs")

st.markdown("""
    <style>
    .main .block-container {
        max-width: 95% !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }
    .stTabs [data-baseweb="tab-panel"] {
        padding: 0 !important;
    }
    div[data-testid="stDataFrame"] {
        width: 100% !important;
    }
    </style>
""", unsafe_allow_html=True)

session = get_snowflake_connection()

with st.sidebar:
    st.header("Configuration")
    
    if session:
        st.success("Connected")
        try:
            user_info = session.sql("SELECT CURRENT_USER(), CURRENT_ROLE(), CURRENT_WAREHOUSE()").collect()[0]
            st.caption(f"User: {user_info[0]} | Role: {user_info[1]}")
        except:
            pass
    else:
        st.error("Not connected")
    
    st.divider()
    
    if st.session_state.dataset is not None:
        st.metric("Records in dataset", len(st.session_state.dataset))
    else:
        st.caption("No dataset loaded yet")
    
    st.divider()
    
    if st.button("Reset dataset", help="Clear dataset and start over"):
        st.session_state.dataset = None
        st.rerun()

if session:
    tab_names = ["1. Load Data", "2. Add Records", "3. Review & Edit", "4. Export"]
    
    col_prev, col_tabs, col_next = st.columns([1, 8, 1])
    
    with col_prev:
        if st.session_state.active_tab > 0:
            if st.button("Back", key="nav_prev", use_container_width=True):
                st.session_state.active_tab -= 1
                st.rerun()
    
    with col_tabs:
        selected_tab_name = st.radio(
            "Navigation",
            tab_names,
            index=st.session_state.active_tab,
            horizontal=True,
            label_visibility="collapsed",
            key="tab_selector"
        )
        st.session_state.active_tab = tab_names.index(selected_tab_name)
    
    with col_next:
        if st.session_state.active_tab < len(tab_names) - 1:
            if st.button("Next", key="nav_next", type="primary", use_container_width=True):
                st.session_state.active_tab += 1
                st.rerun()
    
    st.divider()
    
    if st.session_state.active_tab == 0:
        st.header("Load Data")
        st.caption("Load data from Cortex Analyst request logs")
        
        col1, col2 = st.columns(2)
        
        with col1:
            data_source = st.radio(
                "Data source",
                ["From Specific Semantic Model/View", "From All Requests (View)"],
                key="data_source_selector",
                help="Use specific model/view for filtered logs, or view for all requests"
            )
        
        with col2:
            load_mode = st.radio(
                "Load mode",
                ["Replace", "Append"],
                help="Replace: Clear current dataset\nAppend: Add to current dataset",
                key="load_mode_global"
            )
        
        st.divider()
        
        if data_source == "From Specific Semantic Model/View":
            st.subheader("Load from specific semantic model or view")
            
            model_type = st.radio(
                "Semantic model type",
                ["SEMANTIC_VIEW", "FILE_ON_STAGE"],
                horizontal=True,
                key="model_type_selector"
            )
            
            if model_type == "SEMANTIC_VIEW":
                views_df = get_semantic_views(session)
                
                if not views_df.empty:
                    view_options = sorted(views_df['fully_qualified_name'].tolist())
                    model_name = st.selectbox(
                        "Select semantic view",
                        view_options,
                        key="semantic_view_select"
                    )
                else:
                    model_name = st.text_input(
                        "Semantic view name",
                        placeholder="DATABASE.SCHEMA.SEMANTIC_VIEW_NAME",
                        key="semantic_view_input"
                    )
            else:
                model_name = st.text_input(
                    "Stage path to YAML file",
                    placeholder="@my_db.my_schema.my_stage/path/to/model.yaml",
                    key="stage_path_input"
                )
            
            st.session_state.semantic_model_type = model_type
            st.session_state.semantic_model_name = model_name
            
            if st.button("Load from Analyst logs", type="primary", disabled=not model_name):
                with st.spinner("Querying Cortex Analyst logs..."):
                    try:
                        st.info(f"Querying: {model_type} / {model_name}")
                        df = query_analyst_logs_table_function(session, model_type, model_name)
                        st.info(f"Got {len(df)} raw records")
                        
                        if not df.empty:
                            processed_df = process_analyst_logs(df)
                            st.info(f"After processing: {len(processed_df)} records")
                            
                            if not processed_df.empty:
                                result_df = processed_df[['INPUT_QUERY', 'OUTPUT_SQL']].copy()
                                
                                if load_mode == "Replace" or st.session_state.dataset is None:
                                    st.session_state.dataset = result_df
                                    st.success(f"Loaded {len(result_df)} records")
                                else:
                                    st.session_state.dataset = pd.concat([st.session_state.dataset, result_df], ignore_index=True)
                                    st.success(f"Added {len(result_df)} records")
                                
                                st.rerun()
                            else:
                                st.warning("No valid query/SQL pairs found in logs")
                        else:
                            st.warning("No logs found for this semantic model/view")
                            
                    except Exception as e:
                        st.error(f"Error loading logs: {e}")
                        st.error(traceback.format_exc())
        
        else:
            st.subheader("Load from all Cortex Analyst requests")
            st.info("This requires SNOWFLAKE.CORTEX_ANALYST_REQUESTS_ADMIN or SNOWFLAKE.CORTEX_ANALYST_REQUESTS_VIEWER role")
            
            limit = st.number_input("Max records to load", min_value=100, max_value=10000, value=1000, step=100)
            
            if st.button("Load from requests view", type="primary"):
                with st.spinner("Querying Cortex Analyst requests view..."):
                    try:
                        df = query_analyst_logs_view(session, limit)
                        
                        if not df.empty:
                            processed_df = process_analyst_logs(df)
                            
                            if not processed_df.empty:
                                result_df = processed_df[['INPUT_QUERY', 'OUTPUT_SQL']].copy()
                                
                                if 'SEMANTIC_MODEL_NAME' in processed_df.columns:
                                    result_df['SEMANTIC_MODEL_NAME'] = processed_df['SEMANTIC_MODEL_NAME']
                                
                                if load_mode == "Replace" or st.session_state.dataset is None:
                                    st.session_state.dataset = result_df
                                    st.toast(f"Loaded {len(result_df)} records", icon="✅")
                                else:
                                    st.session_state.dataset = pd.concat([st.session_state.dataset, result_df], ignore_index=True)
                                    st.toast(f"Added {len(result_df)} records", icon="✅")
                                
                                st.rerun()
                            else:
                                st.warning("No valid query/SQL pairs found")
                        else:
                            st.warning("No logs found")
                            
                    except Exception as e:
                        st.error(f"Error: {e}")
                        st.error(traceback.format_exc())
        
        st.divider()
        
        if st.session_state.dataset is not None and len(st.session_state.dataset) > 0:
            st.subheader("Current Dataset Preview")
            st.success(f"{len(st.session_state.dataset)} records loaded")
            
            st.dataframe(
                st.session_state.dataset, 
                use_container_width=True, 
                hide_index=True, 
                height=400,
                column_config={
                    "INPUT_QUERY": st.column_config.TextColumn("Input Query", width="medium"),
                    "OUTPUT_SQL": st.column_config.TextColumn("Output SQL", width="large"),
                    "SEMANTIC_MODEL_NAME": st.column_config.TextColumn("Model", width="small")
                }
            )
        else:
            st.info("No records loaded yet. Choose a data source above.")
    
    elif st.session_state.active_tab == 1:
        st.header("Add Evaluation Records")
        st.caption("Manually create evaluation records")
        
        if st.session_state.dataset is not None and len(st.session_state.dataset) > 0:
            st.success(f"Current dataset: {len(st.session_state.dataset)} records")
        
        with st.form("add_record_form", clear_on_submit=True):
            input_query = st.text_area(
                "Input query (natural language question)",
                placeholder="e.g., What were the total sales last month?",
                height=100,
                key="add_input_query"
            )
            
            output_sql = st.text_area(
                "Expected SQL output",
                placeholder="e.g., SELECT SUM(amount) FROM sales WHERE date >= DATEADD('month', -1, CURRENT_DATE())",
                height=150,
                key="add_output_sql"
            )
            
            submit_button = st.form_submit_button("Add record", type="primary")
        
        if submit_button:
            if not input_query or not output_sql:
                st.error("Please fill in both input query and output SQL")
            else:
                new_row = pd.DataFrame([{
                    'INPUT_QUERY': input_query.strip(),
                    'OUTPUT_SQL': output_sql.strip()
                }])
                
                if st.session_state.dataset is None or len(st.session_state.dataset) == 0:
                    st.session_state.dataset = new_row
                else:
                    st.session_state.dataset = pd.concat([st.session_state.dataset, new_row], ignore_index=True)
                
                st.toast(f"Record added! Total: {len(st.session_state.dataset)}", icon="✅")
                st.rerun()
        
        st.divider()
        
        if st.session_state.dataset is not None and len(st.session_state.dataset) > 0:
            st.subheader("Current Dataset")
            st.dataframe(
                st.session_state.dataset, 
                use_container_width=True, 
                hide_index=True, 
                height=300,
                column_config={
                    "INPUT_QUERY": st.column_config.TextColumn("Input Query", width="medium"),
                    "OUTPUT_SQL": st.column_config.TextColumn("Output SQL", width="large")
                }
            )
    
    elif st.session_state.active_tab == 2:
        st.header("Review & Edit Dataset")
        st.caption("Review records and make edits")
        
        if st.session_state.dataset is not None and len(st.session_state.dataset) > 0:
            st.metric("Total records", len(st.session_state.dataset))
            
            st.divider()
            
            st.subheader("Edit Records")
            
            record_options = [
                f"Record {i+1}: {row['INPUT_QUERY'][:50]}..." 
                for i, row in st.session_state.dataset.iterrows()
            ]
            
            record_index = st.selectbox(
                "Select record to edit",
                range(len(st.session_state.dataset)),
                format_func=lambda x: record_options[x],
                key="edit_record_selector"
            )
            
            current_record = st.session_state.dataset.iloc[record_index]
            
            with st.form(f"edit_form_{record_index}"):
                edited_query = st.text_area(
                    "Input query",
                    value=current_record['INPUT_QUERY'],
                    height=100,
                    key=f"edit_query_{record_index}"
                )
                
                edited_sql = st.text_area(
                    "Output SQL",
                    value=current_record['OUTPUT_SQL'],
                    height=150,
                    key=f"edit_sql_{record_index}"
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    save_button = st.form_submit_button("Save changes", type="primary")
                with col2:
                    delete_button = st.form_submit_button("Delete record", type="secondary")
            
            if save_button:
                st.session_state.dataset.at[record_index, 'INPUT_QUERY'] = edited_query.strip()
                st.session_state.dataset.at[record_index, 'OUTPUT_SQL'] = edited_sql.strip()
                st.toast("Record updated!", icon="✅")
                st.rerun()
            
            if delete_button:
                st.session_state.dataset = st.session_state.dataset.drop(record_index).reset_index(drop=True)
                st.toast("Record deleted", icon="🗑️")
                st.rerun()
            
            st.divider()
            st.subheader("All Records")
            st.dataframe(
                st.session_state.dataset, 
                use_container_width=True, 
                hide_index=True, 
                height=300,
                column_config={
                    "INPUT_QUERY": st.column_config.TextColumn("Input Query", width="medium"),
                    "OUTPUT_SQL": st.column_config.TextColumn("Output SQL", width="large")
                }
            )
        else:
            st.warning("No records in dataset. Go to 'Load Data' or 'Add Records' to get started.")
    
    elif st.session_state.active_tab == 3:
        st.header("Export Dataset")
        st.caption("Export your evaluation dataset to Snowflake or download as CSV")
        
        if st.session_state.dataset is not None and len(st.session_state.dataset) > 0:
            st.success(f"Dataset ready with {len(st.session_state.dataset)} records")
            
            st.subheader("Dataset Preview")
            st.dataframe(
                st.session_state.dataset, 
                use_container_width=True, 
                hide_index=True, 
                height=300,
                column_config={
                    "INPUT_QUERY": st.column_config.TextColumn("Input Query", width="medium"),
                    "OUTPUT_SQL": st.column_config.TextColumn("Output SQL", width="large")
                }
            )
            
            st.divider()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Save to Snowflake")
                table_name = st.text_input(
                    "Table name",
                    value="OBSERVABILITY_DB.OBSERVABILITY_SCHEMA.ANALYST_EVAL_DATASET",
                    placeholder="DATABASE.SCHEMA.TABLE_NAME",
                    key="export_table_name"
                )
                
                save_mode = st.radio("Save mode", ["Append", "Overwrite"], horizontal=True, key="export_save_mode")
                
                if st.button("Save to Snowflake", type="primary"):
                    if not table_name.strip():
                        st.warning("Please enter a table name")
                    elif not validate_table_name(table_name):
                        st.error("Invalid table name format. Use DATABASE.SCHEMA.TABLE")
                    else:
                        with st.spinner("Saving to Snowflake..."):
                            try:
                                overwrite = (save_mode == "Overwrite")
                                table_upper = table_name.strip().upper()
                                
                                if overwrite:
                                    session.sql(f"CREATE OR REPLACE TABLE {table_upper} (INPUT_QUERY VARCHAR, OUTPUT_SQL VARCHAR);").collect()
                                else:
                                    session.sql(f"CREATE TABLE IF NOT EXISTS {table_upper} (INPUT_QUERY VARCHAR, OUTPUT_SQL VARCHAR);").collect()
                                
                                export_df = st.session_state.dataset[['INPUT_QUERY', 'OUTPUT_SQL']].copy()
                                
                                write_result = session.write_pandas(
                                    export_df,
                                    table_upper,
                                    auto_create_table=False,
                                    quote_identifiers=False
                                )
                                
                                if isinstance(write_result, tuple):
                                    success = write_result[0]
                                    nrows = write_result[2] if len(write_result) >= 3 else len(export_df)
                                else:
                                    success = write_result
                                    nrows = len(export_df)
                                
                                if success:
                                    count_result = session.sql(f"SELECT COUNT(*) as cnt FROM {table_upper}").collect()
                                    actual_count = count_result[0]['CNT']
                                    st.toast(f"Saved {nrows} records to {table_name} (Total: {actual_count})", icon="✅")
                                else:
                                    st.error("Failed to write records")
                                    
                            except Exception as e:
                                st.error(f"Error: {e}")
                                st.error(traceback.format_exc())
            
            with col2:
                st.subheader("Download CSV")
                st.caption("Download as CSV file for local use")
                
                csv = st.session_state.dataset.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"analyst_eval_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    type="primary"
                )
        else:
            st.warning("No records in dataset. Go to 'Load Data' or 'Add Records' to build your dataset.")

else:
    st.info("Please connect to Snowflake using the sidebar")
    
    with st.expander("Connection requirements"):
        st.markdown("""
        Required environment variables:
        - `SNOWFLAKE_ACCOUNT` - Your Snowflake account identifier
        - `SNOWFLAKE_USER` - Your Snowflake username
        - `SNOWFLAKE_PASSWORD` - Your Snowflake password
        
        Optional environment variables (with defaults):
        - `SNOWFLAKE_WAREHOUSE` (default: COMPUTE_WH)
        - `SNOWFLAKE_DATABASE` (default: SNOWFLAKE)
        - `SNOWFLAKE_SCHEMA` (default: LOCAL)
        - `SNOWFLAKE_ROLE` (default: ACCOUNTADMIN)
        """)

st.divider()
st.caption("Cortex Analyst Evaluation Dataset Builder | Powered by Snowflake")
