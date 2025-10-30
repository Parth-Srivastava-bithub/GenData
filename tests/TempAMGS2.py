import json
import re
import time
from langchain_ollama import OllamaLLM
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph, END
from pydantic import BaseModel
load_dotenv()
from rich import print
import numpy as np
from typing import List, Literal, Optional, Type
from typing import Optional, List, Literal, Dict, Any
from pydantic import BaseModel, Field
import pandas as pd
from AutoEDA import DataRelationshipExtractor # Assuming this local import is correct
from sdv.single_table import TVAESynthesizer
from sdv.metadata import SingleTableMetadata
import warnings
warnings.filterwarnings("ignore")
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Initialize AI models (keeping names as requested)
groq = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"), temperature=1.5)
gemini = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", api_key=os.getenv("GEMINI_API_KEY"))
openrouter = ChatOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model_name="meituan/longcat-flash-chat:free",
    base_url="https://openrouter.ai/api/v1"
)
# Local model connect
phi_json = OllamaLLM(model="phi3:3.8b-mini-4k-instruct-q4_K_M", format="json")
phi_chat = OllamaLLM(model="phi3:3.8b-mini-4k-instruct-q4_K_M")

def json_output(model_obj, schema: Type[Any]):
    return model_obj.with_structured_output(schema)

# --- Pydantic Schemas (State Definition) ---

class ReferralDataFrames(BaseModel):
    """
    Holds the dataframes for analysis and the reports generated from them.
    """
    dataframes: Dict[str, pd.DataFrame] = Field({}, description="Collections of dataframes to get the summary, e.g., {'Real Data': df1}")
    summary: Dict = Field({}, description="The full JSON report from the AutoEDA agent")
    primarySummary: Optional[Any] = Field(None, description="The condensed 'Data Blueprint' for the generator")
    
    features_list_for_outlier_report: Optional[Dict[str, List[str]]] = Field(
        None,
        description="Input map for the report generator, e.g., {'Real Data': ['Insulin', 'Age']}"
    )
    
    contextual_outlier_report: Optional[Dict[str, Dict[str, pd.DataFrame]]] = Field(
        None,
        description="Output report storing the top k outlier rows for each feature."
    )
    stoppage_for_outliers: List = []

    class Config:
        arbitrary_types_allowed = True


class SynthData(BaseModel):
    data: Optional[Any] = Field(None, description="Synthetic dataframe")
    schemaOfData: Optional[Any] = Field(None, description="Schema of needed dataframe")
    temp_df: Optional[Any] = Field(None, description="Temporary Dataframe for validation")
    final_df: Optional[pd.DataFrame] = Field(None, description="Final Dataframe after checking")
    chunk: Optional[Any] = None
    chunk_fingerprints: List[Dict[str, Any]] = Field(default_factory=list)
    current_chunk_fingerprint: Optional[Dict[str, Any]] = Field(None, description="The fingerprint of the chunk currently being validated.")
    synth_data: Optional[Any] = None
    
    # --- REMOVED chunk_generation_status (redundant) ---
    
    diversity_features: List[str] = Field(default_factory=list, description="A list of feature names to track for diversity.")
    seed_length: int = 20
    occurrence_tracker: Dict[str, Dict[Any, int]] = Field(default_factory=dict, description="Nested dict to store counts for diversity features. {feature_name: {value: count}}")
    
    chunk_size : int = 5
    # --- REMOVED max_tries (using global retry) ---

    class Config:
        arbitrary_types_allowed = True

class ValidationClass(BaseModel):
    mode: Literal['approved', 'retry'] = Field('approved', description="Based on the quality of data suggest to append this for synthetic data or retry to generate more factual different data which is not similar")
    advice: Optional[str] = Field("", description="Advice over the given data and its metadata and tell llm if its bad how to and what to make changes")

class MainState(BaseModel):
    referralDataFrames: Optional[ReferralDataFrames] = None
    synthData: Optional[SynthData] = None
    validationClass: Optional[ValidationClass] = None
    
    # --- NEW RETRY STATE FIELDS ---
    retry_count: int = Field(0, description="Global counter for node retries.")
    max_retries: int = Field(5, description="Global max retries for any node.")
    last_node_name: str = Field("", description="The name of the node currently being executed/retried.")
    # --- END NEW FIELDS ---

    class Config:
        arbitrary_types_allowed = True

# --- Graph Nodes ---

def generate_summary(state: MainState):
    # This node doesn't call an LLM, so it doesn't need retries
    extractor = DataRelationshipExtractor()
    df_list = state.referralDataFrames.dataframes
    all_reports = {}

    for name, df in df_list.items():
        print("\n" + "=" * 50)
        print(f"[bold magenta] RUNNING ANALYSIS FOR: {name} [/bold magenta]")
        print("=" * 50)

        report = extractor.run_analysis(df)
        all_reports[name] = report
        print(f"\n[bold cyan] Individual Report for {name}: [/bold cyan]")
        print(json.dumps(report, indent=2))

    state.referralDataFrames.summary = json.dumps(all_reports) 
    return state


# --- REFACTORED NODE (Groq-Only + Retry Logic) ---
def summarize_summary(state: MainState):
    """
    Extracts critical facts from summary JSON using Groq.
    Manages retry state.
    """
    print(f"[magenta]🧠 RUNNING LIGHT DATA BLUEPRINT CONDENSER (Attempt {state.retry_count + 1}/{state.max_retries + 1})...[/magenta]")
    json_summary = state.referralDataFrames.summary

    prompt = f"""
    You are a senior data analyst. Summarize the dataset into a concise "Data Blueprint".
    Only write clear JSON (no markdown, no code fences).
    
    Structure of the JSON should be:
    {{
      "executive_summary": "<one-line summary>",
      "key_statistical_properties": [
        {{"feature": "<col>", "property": "<e.g. skew, kurtosis, zeros_pct>", "value": "<num>", "notes": "<short insight>"}}
      ],
      "key_correlations": [
        {{"feature_a": "<col1>", "feature_b": "<col2>", "correlation": <num>, "strength": "<positive/negative/moderate>"}}
      ]
    }}

    Rules:
    1. Mention only *high skew (>|2|)*, *high kurtosis (>|5|)*, and *zeros_pct > 30%*.
    2. Add only top 2 strong correlations (|corr| > 0.4).
    3. No markdown, backticks, or extra text.
    
    Dataset summary JSON:
    {json_summary}
    """

    # --- Refactored to Groq-Only with Try/Except ---
    try:
        print(f"[cyan]Attempting Groq...[/cyan]")
        
        reply_message = groq.invoke(prompt)
        reply = reply_message.content
        
        match = re.search(r"```json\n([\s\S]*?)\n```", reply)
        if match:
            reply = match.group(1).strip()

        reply_json = json.loads(reply)
        
        # --- SUCCESS ---
        state.referralDataFrames.primarySummary = reply_json
        state.retry_count = 0 # Reset retry counter on success
        print("[green]✅ Blueprint extraction complete![/green]")
        print({'Data Blueprint': reply_json})
        
    except Exception as e:
        # --- FAILURE ---
        print(f"[yellow]Groq failed: {e}[/yellow]")
        state.retry_count += 1 # Increment retry counter on failure
        state.referralDataFrames.primarySummary = {"error": "Summarizer failed", "raw_data": json_summary}

    return state

    
# --- REFACTORED NODE (Groq-Only + Retry Logic) ---
def generate_chunks(state: MainState):
    """
    Generates a chunk of data using Groq.
    Manages retry state.
    """
    print(f"[magenta]GENERATING CHUNKS (Attempt {state.retry_count + 1}/{state.max_retries + 1})...[/magenta]")
    data_schema = state.synthData.schemaOfData
    
    # --- Use Groq-Only ---
    groq_generator = json_output(groq, data_schema)
    
    occurrence_tracker = state.synthData.occurrence_tracker 
    features_to_track = state.synthData.diversity_features
    advice = state.validationClass.advice if state.validationClass.advice else ""
    
    # 1. Create the occurrence string for the prompt
    occurrence_str: str
    if not features_to_track:
        occurrence_str = "No specific features are being tracked for diversity. Focus on overall realism."
    elif not occurrence_tracker: # First run
            occurrence_str = f"This is the very first chunk. Generate a diverse starting set, especially for these features: {features_to_track}"
    else:
        str_parts = []
        for feature in features_to_track:
            past_values = occurrence_tracker.get(feature) # Get dict for this feature
            
            if not past_values:
                str_parts.append(f"- For '{feature}': No values have been generated yet. Create a diverse set.")
            else:
                sorted_values = sorted(past_values.items(), key=lambda item: item[1])
                str_parts.append(f"- For '{feature}': Counts (rare-to-common): {json.dumps(sorted_values)}")
        occurrence_str = "\n".join(str_parts)

    df_describe_storage = {}
    for name, df in state.referralDataFrames.dataframes.items():
        df_describe_storage[name] = df.describe()

    prompt = f"""
    You are a synthetic data generator. Your goal is to create diverse, realistic data rows.

   **Factual Basis (The "Data Blueprint" from real data):**
       ```
       {json.dumps(state.referralDataFrames.primarySummary, indent=2)}
       ```
       and this is original data describe section: {df_describe_storage}
       make sure crate the data which proves these values and stay near
    **Schema:** {data_schema}

    **CONSTRAINT: AVOID REPETITION!**
    We are actively tracking the diversity of the following features: **{features_to_track}**
    
    Here is a report of the values you have *already* generated for those features:
    ```
    {occurrence_str}
    ```

    **Your Task:**
    1.  Generate {state.synthData.chunk_size} new, unique rows.
    2.  **Prioritize Rarity:** For *each* tracked feature, actively generate data for values with *low counts* (or values not in the list).
    3.  **Contextual Coherence (CRITICAL):** The *entire row* must be contextually correct and coherent, *especially* considering the values you choose for the tracked features.
        * (e.g., If tracking 'diabetes_diagnosis', a "Positive" row should have coherent 'glucose_level', 'bmi', 'age', etc.)
    4.  **Follow Last Advice (if any):** {advice}

    Generate {state.synthData.chunk_size} rows now.
    """
    
    # --- Refactored to Groq-Only with Try/Except ---
    try:
        print("[cyan]Attempting Groq...[/cyan]")
        chunk = groq_generator.invoke(prompt)
        
        # --- SUCCESS ---
        state.synthData.chunk = chunk
        state.retry_count = 0 # Reset retry counter
        
    except Exception as e:
        # --- FAILURE ---
        print(f"[yellow]Groq failed to generate chunk: {e}[/yellow]")
        state.synthData.chunk = None # Set chunk to None on failure
        state.retry_count += 1 # Increment retry counter

    return state

# --- REFACTORED NODE (Handles failure from previous node) ---
def get_chunk_metadata(state: MainState):
    """
    Validates the chunk from the previous step.
    If the chunk is invalid, it sets the retry counter and blames 'generate_chunks'.
    """
    # Set node name for logging, though this node *blames* another
    state.last_node_name = "get_chunk_metadata" 
    
    # --- Check for failure from generate_chunks ---
    if not state.synthData or not state.synthData.chunk or not getattr(state.synthData.chunk, 'rows', None):
        print("[get_chunk_metadata] ⚠️ Chunk is missing or invalid (likely from failed generation).")
        state.retry_count = state.retry_count + 1 # Increment retry
        state.last_node_name = "generate_chunks" # !! Blame the *previous* node
        return state # Return to trigger the retry router

    # --- Try to process the chunk ---
    try:
        chunk_dict = [row.model_dump() for row in state.synthData.chunk.rows]
        df = pd.DataFrame(chunk_dict)
        
        fingerprint = {
            "numerical_stats": {},
            "categorical_stats": {}
        }
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if not numeric_cols.empty:
            fingerprint["numerical_stats"] = df[numeric_cols].describe().to_dict()

        categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns
        for col in categorical_cols:
            fingerprint["categorical_stats"][col] = df[col].value_counts(
                normalize=True
            ).to_dict()

        # --- SUCCESS ---
        state.synthData.current_chunk_fingerprint = fingerprint
        state.retry_count = 0 # Reset retry counter on success
        print("--- 1. Generated Fingerprint for New Chunk ---")
        
    except Exception as e:
        # --- FAILURE (This node failed to process a *valid* chunk) ---
        print(f"[get_chunk_metadata] ⚠️ Error processing chunk DataFrame: {e}")
        state.retry_count += 1 # Increment retry
        state.last_node_name = "generate_chunks" # Blame previous node, as chunk was malformed
        
    return state


# --- REFACTORED NODE (Groq-Only + Retry Logic) ---
def validate_chunk(state: MainState):
    """
    Validates the chunk using Groq.
    Manages retry state.
    """
    print(f"[magenta]VALIDATING CHUNK (Attempt {state.retry_count + 1}/{state.max_retries + 1})...[/magenta]")
    chunk = state.synthData.chunk

    prompt = f"""
    You are an analyzer with domain knowledge of this data: {state.referralDataFrames.primarySummary}.
    I have a new chunk of data: {chunk}.
    Are these new data factually correct and do they make sense, or are they misguided?
    Just validate if the data makes sense.
    """
    
    # --- Use Groq-Only ---
    validator_groq = json_output(groq, ValidationClass)

    # --- Refactored to Groq-Only with Try/Except ---
    try:
        print("[cyan]Attempting Groq...[/cyan]")
        reply = validator_groq.invoke(prompt)
        
        # --- SUCCESS ---
        state.validationClass.mode = reply.mode
        state.validationClass.advice = reply.advice
        state.retry_count = 0 # Reset retry counter
        
    except Exception as e:
        # --- FAILURE ---
        print(f"[yellow]Groq failed to validate chunk: {e}[/yellow]")
        # Failsafe: Set to 'retry' to force loop
        state.validationClass.mode = 'retry'
        state.validationClass.advice = 'Validation model failed, forcing retry.'
        state.retry_count += 1 # Increment retry counter

    return state

# --- NEW: Combined router for validation step ---
def route_after_validation(state: MainState) -> Literal[
    'validate_chunk',           # For retry
    'final_node',               # For fail
    'append_chunk',             # For success
    'set_name_generate_chunks', # For rejection
    'generate_synthetic_data'   # For completion
]:
    """
    First, checks for retries.
    If no retry is needed, checks the validation decision.
    """
    
    # 1. Check for retry first (logic from check_for_retry)
    if state.retry_count > 0:
        if state.retry_count > state.max_retries:
            print(f"[red]❌ Node '{state.last_node_name}' failed after {state.retry_count} attempts. Stopping graph.[/red]")
            return "final_node" # Hard fail
        else:
            # Note: We loop back to 'set_name_validate_chunk' to reset the name
            print(f"[yellow]🔁 Retrying node '{state.last_node_name}' (Attempt {state.retry_count + 1})...[/yellow]")
            return 'validate_chunk' # Loop back to validation node

    # 2. If no retry, run the logic from the old after_validate_chunk
    
    # Stop condition 1: Total rows generated
    if state.synthData.final_df is not None and len(state.synthData.final_df) >= state.synthData.seed_length:
            print(f"Stop condition: Reached target seed count (>= {state.synthData.seed_length}).")
            return 'generate_synthetic_data'
    
    # Continue condition 1: Chunk is approved
    if (state.validationClass.mode == 'approved'):
        print("Routing: Chunk approved -> append_chunk")
        return 'append_chunk'
        
    # Continue condition 2: Chunk is rejected, go back to generate
    print("Routing: Chunk rejected -> set_name_generate_chunks")
    return 'set_name_generate_chunks'


def append_chunk(state: MainState):
    # This node doesn't call an LLM, so it doesn't need retries
    
    if state.synthData.current_chunk_fingerprint:
        state.synthData.chunk_fingerprints.append(state.synthData.current_chunk_fingerprint)
    state.synthData.current_chunk_fingerprint = None
    
    # --- Nested function to safely add DataFrame ---
    def add_chunk_in_df(state: MainState):
        try:
            chunk_rows = [row.model_dump() for row in state.synthData.chunk.rows]
            chunk_df = pd.DataFrame(chunk_rows)

            occurrence_tracker = state.synthData.occurrence_tracker
            features_to_track = state.synthData.diversity_features
            
            for feature_to_track in features_to_track:
                
                if feature_to_track not in chunk_df.columns:
                    print(f"[append_chunk] ⚠️ Warning: diversity_feature '{feature_to_track}' not in chunk. Skipping dict update for this feature.")
                    continue 
                
                if feature_to_track not in occurrence_tracker:
                    occurrence_tracker[feature_to_track] = {}
                
                feature_dict = occurrence_tracker[feature_to_track] 
                
                for value in chunk_df[feature_to_track]:
                    value_key = "None" if pd.isna(value) else value
                    feature_dict[value_key] = feature_dict.get(value_key, 0) + 1
                
                print(f"[append_chunk] ✅ Updated occurrence_tracker for '{feature_to_track}'.")
            
            state.synthData.occurrence_tracker = occurrence_tracker

            print(f"[append_chunk] ✅ Updated occurrence_dict with {len(chunk_df)} new rows for features: {state.synthData.diversity_features}.")

            if state.synthData.final_df is None or state.synthData.final_df.empty:
                state.synthData.final_df = chunk_df
            else:
                state.synthData.final_df = pd.concat(
                    [state.synthData.final_df, chunk_df],
                    ignore_index=True
                )

            state.synthData.chunk = None
            print(f"[append_chunk] ✅ Added {len(chunk_df)} rows. Final DF shape: {state.synthData.final_df.shape}")

        except Exception as e:
            print(f"[append_chunk] ❌ Error adding chunk or updating dict: {e}")

        return state
    # --- End of nested function ---
    
    state = add_chunk_in_df(state)
    return state

def generate_synthetic_data(state: MainState):
    # This node is for TVAE (local), not an LLM API call.
    # If it fails, it's a code error, so we don't need the API retry logic.
    print("\n--- 🚀 Entering VAE Synthetic Data Generation Node ---")
    
    golden_seed_df = state.synthData.final_df
    
    if golden_seed_df is None or golden_seed_df.empty:
        print("Error: Cannot train TVAE. `state.synthData.final_df` is empty.")
        return state

    if len(golden_seed_df) < 50: 
        print(f"Warning: Training TVAE on a very small seed dataset ({len(golden_seed_df)} rows).")
        print("Model may have low quality, but proceeding...")
    else:
        print(f"Training TVAE on golden seed data (Shape: {golden_seed_df.shape})...")

    try:
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data=golden_seed_df)

        synthesizer = TVAESynthesizer(metadata)
        synthesizer.fit(golden_seed_df)

        num_to_generate = 500
        print(f"Sampling {num_to_generate} new rows from the trained TVAE model...")
        
        synthetic_data = synthesizer.sample(num_rows=num_to_generate)

        state.synthData.synth_data = synthetic_data
        print(f"✅ Successfully generated and stored {len(synthetic_data)} rows in `synthData.synth_data`.")
    
    except Exception as e:
        print(f"❌ Error during TVAE training or sampling: {e}")
        state.synthData.synth_data = None

    return state

def get_outlier_report(state: MainState):
    # This node is local pandas logic, no retries needed.
    print("[get_outlier_report] 🚀 Starting Contextual Outlier Report generation...")
    
    report_map = state.referralDataFrames.features_list_for_outlier_report
    all_dataframes = state.referralDataFrames.dataframes

    if not report_map:
        print("[get_outlier_report] ⚠️ No features listed for outlier report. Skipping.")
        return state

    def get_top_k(df: pd.DataFrame, feature: str, k: int = 5) -> pd.DataFrame:
        if feature not in df.columns:
            print(f"[get_top_k] ⚠️ Warning: Feature '{feature}' not found in DataFrame. Returning empty DataFrame.")
            return pd.DataFrame(columns=df.columns)
        
        if not pd.api.types.is_numeric_dtype(df[feature]):
            print(f"[get_top_k] ⚠️ Warning: Feature '{feature}' is not numeric. Cannot sort. Returning empty DataFrame.")
            return pd.DataFrame(columns=df.columns)

        sorted_df = df.sort_values(by=feature, ascending=False)
        top_k_rows = sorted_df.head(k)
        return top_k_rows

    outliers_storage: Dict[str, Dict[str, pd.DataFrame]] = {}

    for df_id, feature_list in report_map.items():
        df_to_analyze = all_dataframes.get(df_id)
        
        if df_to_analyze is None:
            print(f"[get_outlier_report] ⚠️ Warning: DataFrame ID '{df_id}' not found in state.dataframes. Skipping.")
            continue

        print(f"[get_outlier_report] 🔎 Analyzing DataFrame: '{df_id}'")
        outliers_storage[df_id] = {}
        
        for feature in feature_list:
            top_k_df = get_top_k(df_to_analyze, feature, k=5)
            
            if not top_k_df.empty:
                print(f"[get_outlier_report]    -> Found and stored top 5 outliers for '{feature}'")
                outliers_storage[df_id][feature] = top_k_df
            else:
                print(f"[get_outlier_report]   -> No outliers found or error for '{feature}'")

    state.referralDataFrames.contextual_outlier_report = outliers_storage
    print(f"[get_outlier_report] ✅ Contextual Outlier Report complete.")
    print({'outlier_report': outliers_storage})

    return state

# --- REFACTORED NODE (Groq-Only + Retry Logic) ---
def put_outliers(state: MainState):
    """
    Generates a chunk of pure, context-aware outliers using Groq-70b.
    Manages retry state.
    """
    print(f"[put_outliers] 🚀 Triggered OUTLIER INJECTION node (Attempt {state.retry_count + 1}/{state.max_retries + 1})...")
    
    outliers_report_map = state.referralDataFrames.contextual_outlier_report
    blueprint = state.referralDataFrames.primarySummary
    schema = state.synthData.schemaOfData
    chunk_size = state.synthData.chunk_size 
    
   
    # --- Use Groq-Only ---
    groq_generator = json_output(groq, schema)

    outliers_json_report = {}
    if outliers_report_map:
        for df_id, feature_dfs in outliers_report_map.items():
            outliers_json_report[df_id] = {}
            for feature, df_outliers in feature_dfs.items():
                outliers_json_report[df_id][feature] = df_outliers.describe().to_json() 

    prompt = f"""
        You are an **Outlier Data Generator AI**, specialized in creating context-aware synthetic outliers.

        🎯 **Objective:**
        Generate {chunk_size} rows of **extreme but realistic outliers** for the Real Data.
        Push values close to the real data’s upper extremes, without exceeding the maximum observed limits.

        **Rules for Generation:**
        1. NEVER exceed the actual max value of any feature from the data report.
        2. Stay within 90–99.5% of that max value to simulate realistic extremity.
        3. Preserve inter-feature relationships — if Insulin is high, Glucose should show correlated rise.
        4. Each generated row must appear *statistically extreme but biologically or contextually plausible.*

        **Factual Basis 1 (Blueprint - Global Summary):**
        ```json
        {json.dumps(blueprint, indent=2)}
        ```

        **Factual Basis 2 (Outlier Context Report - Local Extremes):**
        ```json
        {json.dumps(outliers_json_report, indent=2)}
        ```

        **Schema for Output:** {schema}

        🧠 **Output Instructions:**
        * Return a JSON list of {chunk_size} rows.
        * Each feature must obey the above constraints.
        * Outliers should mimic real-world data skew and magnitude without crossing logical or statistical boundaries.
    """

    # --- Refactored to Groq-Only with Try/Except ---
    try:
        print("[put_outliers] Attempting Groq (Specialized Model)...")
        chunk = groq_generator.invoke(prompt)
        
        # --- SUCCESS ---
        state.synthData.chunk = chunk
        state.retry_count = 0 # Reset retry counter
        print(f"[put_outliers] ✅ Successfully generated {chunk_size} outlier rows.")
        print({'outlier chunk': chunk})
        
    except Exception as e:
        # --- FAILURE ---
        print(f"[yellow]Groq (Specialized Model) failed: {e}[/yellow]")
        state.synthData.chunk = None # Set chunk to None on failure
        state.retry_count += 1 # Increment retry counter

    return state

def condition_for_adding_outliers(state: MainState) -> Literal['set_name_generate_chunks', 'get_outlier_report']:
    if (len(state.referralDataFrames.stoppage_for_outliers) > 0) and (state.referralDataFrames.stoppage_for_outliers[0] < len(state.synthData.final_df)):
        state.referralDataFrames.stoppage_for_outliers.pop(0)
        return 'get_outlier_report'
    # Route to the *setter* node, not the main node
    return 'set_name_generate_chunks'


def final_node(state: MainState):
    print("\n--- 🏁 Reached Final Node ---")
    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)
    if state.synthData.synth_data is not None:
        output_path = os.path.join(output_folder, "final_dataframe.csv")
        state.synthData.synth_data.to_csv(output_path, index=False)
        print(f"Final DataFrame saved to {output_path}")
    else:
        print("Graph finished, but no synthetic data was generated (likely due to an error or empty seed).")
    return state

# --- NEW: Utility Nodes for Setting State ---
# These nodes set the 'last_node_name' before a fallible operation

def set_name_summarize(state: MainState):
    state.last_node_name = "summarize_summary"
    state.retry_count = 0 # Reset counter for the *new* node
    return state

def set_name_generate_chunks(state: MainState):
    state.last_node_name = "generate_chunks"
    state.retry_count = 0 # Reset counter for the *new* node
    return state

def set_name_validate_chunk(state: MainState):
    state.last_node_name = "validate_chunk"
    state.retry_count = 0 # Reset counter for the *new* node
    return state

def set_name_put_outliers(state: MainState):
    state.last_node_name = "put_outliers"
    state.retry_count = 0 # Reset counter for the *new* node
    return state


# --- NEW: Universal Retry Router ---

def check_for_retry(state: MainState) -> Literal["retry", "continue", "fail"]:
    """
    Checks the retry_count to decide the next step.
    - "retry": retry_count > 0 and <= max_retries
    - "continue": retry_count == 0 (success)
    - "fail": retry_count > max_retries (hard failure)
    """
    if state.retry_count > 0:
        if state.retry_count > state.max_retries:
            print(f"[red]❌ Node '{state.last_node_name}' failed after {state.retry_count} attempts. Stopping graph.[/red]")
            return "fail"
        else:
            print(f"[yellow]🔁 Retrying node '{state.last_node_name}' (Attempt {state.retry_count + 1})...[/yellow]")
            return "retry"
    
    # print(f"[green]✅ Node '{state.last_node_name}' succeeded. Continuing...[/green]")
    return "continue"

# --- Graph Definition (Refactored with Retry Loops) ---

graph = StateGraph(MainState)
# --- Graph Definition (Refactored with Retry Loops) ---


# 1. Add all nodes
graph.add_node('generate_summary', generate_summary)
graph.add_node('set_name_summarize', set_name_summarize)
graph.add_node('summarize_summary', summarize_summary)
graph.add_node('set_name_generate_chunks', set_name_generate_chunks)
graph.add_node('generate_chunks', generate_chunks)
graph.add_node('get_chunk_metadata', get_chunk_metadata)
graph.add_node('set_name_validate_chunk', set_name_validate_chunk)
graph.add_node('validate_chunk', validate_chunk)
graph.add_node('append_chunk', append_chunk)
graph.add_node('get_outlier_report', get_outlier_report)
graph.add_node('set_name_put_outliers', set_name_put_outliers)
graph.add_node('put_outliers', put_outliers)
graph.add_node('generate_synthetic_data', generate_synthetic_data)
graph.add_node('final_node', final_node)

# Entry Point
graph.set_entry_point('generate_summary')

# 1. Summarize Loop (LLM Call)
graph.add_edge('generate_summary', 'set_name_summarize')
graph.add_edge('set_name_summarize', 'summarize_summary')
graph.add_conditional_edges(
    'summarize_summary',
    check_for_retry,
    {
        "retry": "summarize_summary",       # Loop back to self
        "continue": "set_name_generate_chunks", # Proceed to next step
        "fail": "final_node"                # Stop graph
    }
)

# 2. Generate Chunk Loop (LLM Call)
graph.add_edge('set_name_generate_chunks', 'generate_chunks')
graph.add_conditional_edges(
    'generate_chunks',
    check_for_retry,
    {
        "retry": "generate_chunks",         # Loop back to self
        "continue": "get_chunk_metadata",   # Proceed to metadata check
        "fail": "final_node"
    }
)

# 3. Metadata Check Loop (Local Logic - Routes back to generate_chunks on error)
# Note: This uses check_for_retry, but if it fails, the error state in get_chunk_metadata
# sets the last_node_name to 'generate_chunks' to force the loop back correctly.
graph.add_conditional_edges(
    'get_chunk_metadata', 
    check_for_retry,      
    {
        "retry": "generate_chunks",         # !! Loop back to GENERATE_CHUNKS !!
        "continue": "set_name_validate_chunk", # Proceed to validation setup
        "fail": "final_node"
    }
)

# 4. Validation Loop (LLM Call)
graph.add_edge('set_name_validate_chunk', 'validate_chunk')
graph.add_conditional_edges(
    'validate_chunk',
    route_after_validation,
    {
        # Routes from retry logic (which use the node setter to reset)
        "validate_chunk": "set_name_validate_chunk", 
        "final_node": "final_node",
        
        # Routes from validation decision logic
        'generate_synthetic_data': 'generate_synthetic_data',
        'append_chunk': 'append_chunk',
        'set_name_generate_chunks': 'set_name_generate_chunks'
    }
)

# 5. Append/Outlier Router (Local Logic)
graph.add_conditional_edges(
    'append_chunk', 
    condition_for_adding_outliers, 
    {
        'set_name_generate_chunks': 'set_name_generate_chunks', # Go to normal generation
        'get_outlier_report': 'get_outlier_report'    # Go to outlier step
    }
)

# 6. Outlier Generation Loop (LLM Call)
graph.add_edge('get_outlier_report', 'set_name_put_outliers')
graph.add_edge('set_name_put_outliers', 'put_outliers')
graph.add_conditional_edges(
    'put_outliers',
    check_for_retry,
    {
        "retry": "put_outliers",            # Loop back to self
        "continue": "append_chunk",         # Proceed to append the outlier chunk
        "fail": "final_node"
    }
)

# 7. Final Path
graph.add_edge('generate_synthetic_data', 'final_node') 
graph.set_finish_point('final_node')
# --- Pydantic Schemas for Data ---

df1 = pd.read_csv("Data/diabetes.csv") # Make sure this path is correct
df2 = pd.read_csv("Data/diabetes_risk_dataset.csv") # Make sure this path is correct

class DiabetesRecord(BaseModel):
    age: int = Field(..., ge=0, le=120, description="Age of the patient in years")
    bmi: float = Field(..., ge=0, description="Body Mass Index")
    glucose_level: float = Field(..., ge=0, description="Fasting glucose level in mg/dL")
    blood_pressure: float = Field(..., ge=0, description="Blood pressure in mmHg")
    insulin: float = Field(..., ge=0, description="Insulin level in µU/mL")
    skin_thickness: float = Field(..., ge=0, description="Skin fold thickness in mm")
    genetic_risk_score: int = Field(..., ge=1, le=10, description="Genetic risk factor, 1 (low) to 10 (high)")
    exercise_frequency: Literal["Sedentary", "Low", "Moderate", "High"] = Field(..., description="Activity level")
    family_history: bool = Field(..., description="Whether diabetes runs in family")
    diabetes_diagnosis: Literal["Positive", "Negative"] = Field(..., description="Final diabetes diagnosis result")

class ChunksRecords(BaseModel):
    rows: Optional[List[DiabetesRecord]] = None
    
# --- Initial State Definition ---
state = MainState(
    referralDataFrames=ReferralDataFrames(
        dataframes={
            "Real Data": df1,
            "Read_Data_2": df2
        },
        summary={},
        primarySummary=None,
        features_list_for_outlier_report={
            "Real Data": ["Insulin", "Age"]
        },
        contextual_outlier_report=None,
        stoppage_for_outliers=[20, 70]
    ),
    synthData=SynthData(
        schemaOfData=ChunksRecords,
        diversity_features=["exercise_frequency", "diabetes_diagnosis"],
        seed_length=100,
        chunk_size=10
    ),
    validationClass=ValidationClass(
        advice="Start by generating a diverse first chunk, paying attention to both 'exercise_frequency' and 'diabetes_diagnosis'."
    ),
    max_retries=4 # You can change the global max retries here
)

# --- Graph Execution ---

final_graph = graph.compile()

print("--- Invoking Graph ---")
final_state = final_graph.invoke(
    state,
    config={"recursion_limit": 250} # Increased recursion limit for retries
)

# --- Final Output ---
if final_state['synthData'].final_df is not None:
    from rich import print
    print("\n[bold green]Real Data (Input df1):[/bold green]")
    print(df1.describe())
    
if final_state['synthData'].synth_data is not None:
    print("\n[bold cyan]Synthetic Data (final_state['synthData'].synth_data):[/bold cyan]")
    print(final_state['synthData'].synth_data.describe())
else:
    print("\n--- No Final Synthetic DataFrame generated ---")

# (Optional: Helper function, not used in graph)
def save_dataframe_to_csv(df, filename):
    if df is not None and not df.empty:
        df.to_csv(filename, index=False)
        print(f"[save_dataframe_to_csv] ✅ DataFrame saved to '{filename}' ({len(df)} rows).")
    else:
        print(f"[save_dataframe_to_csv] ⚠️ No data to save for '{filename}'. DataFrame is None or empty.")