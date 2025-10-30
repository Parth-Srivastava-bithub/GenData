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

# Initialize AI models
groq = ChatGroq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"), temperature=1.5)
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
    
    # 1. INPUT FIELD: A map telling the node which features to get outliers for.
    features_list_for_outlier_report: Optional[Dict[str, List[str]]] = Field(
        None,
        description="Input map for the report generator, e.g., {'Real Data': ['Insulin', 'Age']}"
    )
    
    # 2. OUTPUT FIELD: This is what get_outlier_report will populate.
    #    It stores: {'Real Data': {'Insulin': <df>, 'Age': <df>}}
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
    chunk_generation_status: Literal["success", "failure"] = Field("success", description="Flag set by get_chunk_metadata to check chunk validity.")
# --- MODIFICATIONS ---
# Gemini Here make changes in data
   # The feature we want to actively manage for diversity.
    diversity_features: List[str] = Field(default_factory=list, description="A list of feature names to track for diversity.")
    seed_length: int = 20
   # This dict will store counts for the *values* of the diversity_feature.
   # e.g., if diversity_feature="age", this is {50: 6, 51: 2}
   # e.g., if diversity_feature="exercise_frequency", this is {"Sedentary": 8, "High": 3}
    occurrence_tracker: Dict[str, Dict[Any, int]] = Field(default_factory=dict, description="Nested dict to store counts for diversity features. {feature_name: {value: count}}")# --- END MODIFICATIONS ---
    
    chunk_size : int = 5
    max_tries: int = 5 # This is per-chunk retry

    class Config:
        arbitrary_types_allowed = True

class ValidationClass(BaseModel):
    mode: Literal['approved', 'retry'] = Field('approved', description="Based on the quality of data suggest to append this for synthetic data or retry to generate more factual different data which is not similar")
    advice: Optional[str] = Field("", description="Advice over the given data and its metadata and tell llm if its bad how to and what to make changes")

class MainState(BaseModel):
    referralDataFrames: Optional[ReferralDataFrames] = None
    synthData: Optional[SynthData] = None
    validationClass: Optional[ValidationClass] = None
    # ... existing fields ...
    retry_count: int = 0
    max_retries: int = 3 # Set a global limit
    last_node_name: str = "" # To know where to loop back
    # ... existing fields ...
    class Config:
        arbitrary_types_allowed = True

# --- Graph Nodes ---

def generate_summary(state: MainState):
    extractor = DataRelationshipExtractor()
    # This fix assumes your state is: dataframes={"Real Data": df1}
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

    # Store the full JSON report(s)
    state.referralDataFrames.summary = json.dumps(all_reports) 
    return state


# --- REFACTORED NODE ---
def summarize_summary(state: MainState):
    """
    Simplified version of Data Blueprint Condenser.
    Extracts only the critical numerical facts from summary JSON.
    """
    print("[magenta]🧠 RUNNING LIGHT DATA BLUEPRINT CONDENSER...[/magenta]")
    json_summary = state.referralDataFrames.summary

    # --- Cleaner prompt ---
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

    # --- Lightweight fallback logic ---
    for model_name, model in [("Groq", groq), ("Gemini", gemini), ("OpenRouter", openrouter)]:
        try:
            print(f"[cyan]Attempting {model_name}...[/cyan]")
            
            # 💡 CRITICAL FIX: Extract the text content from the AIMessage object
            reply_message = model.invoke(prompt)
            reply = reply_message.content
            
            # Clean up: sometimes models return JSON in markdown fences (```json...```)
            # This is a safe way to extract pure JSON from a string that might contain fences.
            match = re.search(r"```json\n([\s\S]*?)\n```", reply)
            if match:
                reply = match.group(1).strip()

            reply_json = json.loads(reply)  # Now `reply` is a string, so this will work
            
            state.referralDataFrames.primarySummary = reply_json
            print("[green]✅ Blueprint extraction complete![/green]")
            print({'Data Blueprint': reply_json})
            return state
        except Exception as e:
            print(f"[yellow]{model_name} failed: {e}[/yellow]")
            continue

    # --- Final fallback ---
    print("[red]All models failed. Returning raw summary.[/red]")
    state.referralDataFrames.primarySummary = {"error": "All summarizers failed", "raw_data": json_summary}
    return state

    
def generate_chunks(state: MainState):
    data_schema = state.synthData.schemaOfData
    groq_generator = json_output(groq, data_schema)
    gemini_generator = json_output(gemini, data_schema)
    openrouter_generator = json_output(openrouter, data_schema)
    
# --- NEW GENERIC LOGIC ---
    # This will be {} on first run
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
        # Build a multi-part string
        str_parts = []
        for feature in features_to_track:
            past_values = occurrence_tracker.get(feature) # Get dict for this feature
            
            if not past_values:
                str_parts.append(f"- For '{feature}': No values have been generated yet. Create a diverse set.")
            else:
                # Sort by count (ascending) to show the LLM what's rare
                sorted_values = sorted(past_values.items(), key=lambda item: item[1])
                str_parts.append(f"- For '{feature}': Counts (rare-to-common): {json.dumps(sorted_values)}")
        occurrence_str = "\n".join(str_parts)
# --- END GENERIC LOGIC ---
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
    
    chunk = None
    # ... (rest of the function is unchanged) ...
    # ... (rest of the function is unchanged) ...
    try:
        chunk = groq_generator.invoke(prompt)
    except:
        try:
            chunk = gemini_generator.invoke(prompt)
        except:
            try:
                chunk = openrouter_generator.invoke(prompt)
            except:
                raise ValueError("Failed try again...")
    
    state.synthData.chunk = chunk
    return state
# --- In your Graph Nodes ---

def get_chunk_metadata(state: MainState):
    # --- START MODIFICATION ---
    # This block is now your "try/catch"
    if not state.synthData or not state.synthData.chunk or not getattr(state.synthData.chunk, 'rows', None):
        print("Error: No chunk data or rows found to generate metadata.")
        print("[get_chunk_metadata] ⚠️ Chunk generation failed. Setting status to 'failure' and routing back to 'generate_chunks'.")
        
        # Set the flag so our new conditional edge can catch it
        state.synthData.chunk_generation_status = "failure" 
        
        # We reset the main retry counter because this wasn't a *validation* fail,
        # it was a *generation* fail.
        state.synthData.max_tries = 5 
        
        return state # Return gracefully
    # --- END MODIFICATION ---

    # Convert Pydantic model → dict → DataFrame
    try:
        chunk_dict = [row.model_dump() for row in state.synthData.chunk.rows]
        df = pd.DataFrame(chunk_dict)
    except Exception as e:
        print(f"Error converting chunk to DataFrame: {e}")
        # --- ADD THIS BLOCK TOO ---
        # This also counts as a generation failure
        print("[get_chunk_metadata] ⚠️ Chunk was not valid Pydantic. Setting status to 'failure'.")
        state.synthData.chunk_generation_status = "failure"
        state.synthData.max_tries = 5
        return state
        # --- END ADDITION ---
    
    # --- Metadata Creation Part ---
    fingerprint = {
        "numerical_stats": {},
        "categorical_stats": {}
    }

    # 1. Get Numerical Stats
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if not numeric_cols.empty:
        fingerprint["numerical_stats"] = df[numeric_cols].describe().to_dict()

    # 2. Get Categorical Stats
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns
    for col in categorical_cols:
        fingerprint["categorical_stats"][col] = df[col].value_counts(
            normalize=True
        ).to_dict()

    # --- ADD THIS LINE at the end of the function ---
    state.synthData.chunk_generation_status = "success" # Mark as success!
    state.synthData.current_chunk_fingerprint = fingerprint
    
    print("--- 1. Generated Fingerprint for New Chunk ---")
    
    # This max_tries is for *retries of a single bad chunk*
    state.synthData.max_tries -= 1
    return state

# --- Add this function anywhere with your other nodes ---

def route_after_metadata(state: MainState) -> Literal["validate_chunk", "generate_chunks"]:
    """
    Checks if the chunk generation was successful.
    If 'failure', route back to 'generate_chunks'.
    If 'success', proceed to 'validate_chunk'.
    """
    if state.synthData.chunk_generation_status == "failure":
        return "generate_chunks"
    else:
        return "validate_chunk"


def validate_chunk(state: MainState):
    chunk = state.synthData.chunk

    prompt = f"""
    You are an analyzer with domain knowledge of this data: {state.referralDataFrames.primarySummary}.
    I have a new chunk of data: {chunk}.
    Are these new data factually correct and do they make sense, or are they misguided?
    Just validate if the data makes sense.
    """

    validator_groq = json_output(groq, ValidationClass)
    validator_gemini = json_output(gemini, ValidationClass)
    validator_openrouter = json_output(openrouter, ValidationClass)

    reply = None
    try:
        reply = validator_groq.invoke(prompt)
    except:
        try:
            reply = validator_gemini.invoke(prompt)
        except:
            try:
                reply = validator_openrouter.invoke(prompt)
            except:
                print("Models Failed, auto-approving chunk to avoid crash.")
                # Failsafe: create a default "approved" response
                reply = ValidationClass(mode='approved', advice='Validation models failed, auto-approving.')

    state.validationClass.mode = reply.mode
    state.validationClass.advice = reply.advice
    
    return state

def after_validate_chunk(state: MainState) -> Literal['append_chunk', 'generate_chunks', 'final_node', 'generate_synthetic_data']:
    print(f"Current retry count: {state.synthData.max_tries}")
    
    # Stop condition 1: Total rows generated
    if state.synthData.final_df is not None and len(state.synthData.final_df) >= state.synthData.seed_length:
                # --- CHANGE THIS ---
        # print("Stop condition: Reached target row count (>= 20).")
        # --- TO THIS ---
        print(f"Stop condition: Reached target seed count (>= {state.synthData.seed_length}).")
        return 'generate_synthetic_data'

    

    # Stop condition 2: Too many retries for a *single chunk*
    if state.synthData.max_tries < 0:
        print("Stop condition: Max retries for a single chunk failed. Stopping.")
        return 'final_node'
    
    
    # Continue condition 1: Chunk is approved
    if (state.validationClass.mode == 'approved'):
        print("Routing: Chunk approved -> append_chunk")
        return 'append_chunk'
        
    # Continue condition 2: Chunk is rejected, go back to generate
    print("Routing: Chunk rejected -> generate_chunks")
    return 'generate_chunks'

def append_chunk(state: MainState):
    # Add current fingerprint to history
    if state.synthData.current_chunk_fingerprint:
        state.synthData.chunk_fingerprints.append(state.synthData.current_chunk_fingerprint)
    state.synthData.current_chunk_fingerprint = None
    
    # Reset max_tries for the *next* chunk
    state.synthData.max_tries = 5

    # --- Nested function to safely add DataFrame ---
    def add_chunk_in_df(state: MainState):
        try:
            # Convert chunk rows (list of DiabetesRecord) → list of dicts → DataFrame
            chunk_rows = [row.model_dump() for row in state.synthData.chunk.rows]
            chunk_df = pd.DataFrame(chunk_rows)

# --- NEW GENERIC LOGIC ---
            # Get the main tracker dict
            occurrence_tracker = state.synthData.occurrence_tracker
            # Get the list of features we care about
            features_to_track = state.synthData.diversity_features
            
            # Loop over each feature we're supposed to track
            for feature_to_track in features_to_track:
                
                if feature_to_track not in chunk_df.columns:
                    print(f"[append_chunk] ⚠️ Warning: diversity_feature '{feature_to_track}' not in chunk. Skipping dict update for this feature.")
                    continue # Skip to the next feature in the list
                
                # Ensure the nested dict for this feature exists
                if feature_to_track not in occurrence_tracker:
                    occurrence_tracker[feature_to_track] = {}
                
                # Get the nested dict for this specific feature (e.g., occurrence_tracker["age"])
                feature_dict = occurrence_tracker[feature_to_track] 
                
                # Iterate over the values *for this feature* in the new chunk
                for value in chunk_df[feature_to_track]:
                    value_key = "None" if pd.isna(value) else value
                    
                    # 4. Use .get() for safe incrementing
                    feature_dict[value_key] = feature_dict.get(value_key, 0) + 1
                
                print(f"[append_chunk] ✅ Updated occurrence_tracker for '{feature_to_track}'.")
            
            # 5. Save the updated main tracker back to the state
            state.synthData.occurrence_tracker = occurrence_tracker

            # --- THIS IS THE FIX ---
            # Changed 'state.synthData.diversity_feature' (singular) to 'state.synthData.diversity_features' (plural)
            print(f"[append_chunk] ✅ Updated occurrence_dict with {len(chunk_df)} new rows for features: {state.synthData.diversity_features}.")
            # --- END OF FIX ---
# --- END NEW LOGIC ---

            # Initialize or append to final_df
            if state.synthData.final_df is None or state.synthData.final_df.empty:
                state.synthData.final_df = chunk_df
            else:
                state.synthData.final_df = pd.concat(
                    [state.synthData.final_df, chunk_df],
                    ignore_index=True
                )

            # Reset current chunk
            state.synthData.chunk = None
            print(f"[append_chunk] ✅ Added {len(chunk_df)} rows. Final DF shape: {state.synthData.final_df.shape}")

        except Exception as e:
            print(f"[append_chunk] ❌ Error adding chunk or updating dict: {e}")

        return state
    # --- End of nested function ---
    
    state = add_chunk_in_df(state)
    return state

def generate_synthetic_data(state: MainState):
    print("\n--- 🚀 Entering VAE Synthetic Data Generation Node ---")
    
    golden_seed_df = state.synthData.final_df
    
    # 1. Check if the seed data exists
    if golden_seed_df is None or golden_seed_df.empty:
        print("Error: Cannot train TVAE. `state.synthData.final_df` is empty.")
        return state

    # 2. Warn about small seed size
    if len(golden_seed_df) < 50: # TVAE is a NN, 20 is tiny
        print(f"Warning: Training TVAE on a very small seed dataset ({len(golden_seed_df)} rows).")
        print("Model may have low quality, but proceeding...")
    else:
        print(f"Training TVAE on golden seed data (Shape: {golden_seed_df.shape})...")

    try:
        # 3. Detect metadata
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data=golden_seed_df)

        # 4. Instantiate and fit the model
        synthesizer = TVAESynthesizer(metadata)
        synthesizer.fit(golden_seed_df)

        # 5. Generate (sample) new data
        num_to_generate = 500 # Or you could add this to your state
        print(f"Sampling {num_to_generate} new rows from the trained TVAE model...")
        
        synthetic_data = synthesizer.sample(num_rows=num_to_generate)

        # 6. Store the new synthetic data in the state
        state.synthData.synth_data = synthetic_data
        print(f"✅ Successfully generated and stored {len(synthetic_data)} rows in `synthData.synth_data`.")
    
    except Exception as e:
        print(f"❌ Error during TVAE training or sampling: {e}")
        state.synthData.synth_data = None

    return state

def get_outlier_report(state: MainState):
    """
    Generates a report of the top k outlier rows for specified features.
    This function implements the "Sort and Slice" plan.
    
    It reads from: state.referralDataFrames.features_list_for_outlier_report
    It writes to: state.referralDataFrames.contextual_outlier_report
    """
    print("[get_outlier_report] 🚀 Starting Contextual Outlier Report generation...")
    
    # 1. Get the map of features to analyze, e.g., {"Real Data": ["Insulin", "Age"]}
    report_map = state.referralDataFrames.features_list_for_outlier_report
    
    # 2. Get all available dataframes, e.g., {"Real Data": <df1>}
    all_dataframes = state.referralDataFrames.dataframes

    if not report_map:
        print("[get_outlier_report] ⚠️ No features listed for outlier report. Skipping.")
        return state

    # --- Helper Function ---
    def get_top_k(df: pd.DataFrame, feature: str, k: int = 5) -> pd.DataFrame:
        """
        Sorts the DataFrame by the specified feature (descending)
        and returns the top k rows.
        """
        # Check if feature exists
        if feature not in df.columns:
            print(f"[get_top_k] ⚠️ Warning: Feature '{feature}' not found in DataFrame. Returning empty DataFrame.")
            return pd.DataFrame(columns=df.columns)
        
        # Check if feature is numeric (can't sort 'object' types reliably for outliers)
        if not pd.api.types.is_numeric_dtype(df[feature]):
            print(f"[get_top_k] ⚠️ Warning: Feature '{feature}' is not numeric. Cannot sort. Returning empty DataFrame.")
            return pd.DataFrame(columns=df.columns)

        # Sort values and get top k
        sorted_df = df.sort_values(by=feature, ascending=False)
        top_k_rows = sorted_df.head(k)
        return top_k_rows
    # --- End Helper ---

    # This will be the final output object, structured as you described:
    # { df_id: { col1: <top_5_df>, col2: <top_5_df> } }
    outliers_storage: Dict[str, Dict[str, pd.DataFrame]] = {}

    # Loop through the map, e.g., df_id="Real Data", feature_list=["Insulin", "Age"]
    for df_id, feature_list in report_map.items():
        
        # Get the actual DataFrame from the state
        df_to_analyze = all_dataframes.get(df_id)
        
        if df_to_analyze is None:
            print(f"[get_outlier_report] ⚠️ Warning: DataFrame ID '{df_id}' not found in state.dataframes. Skipping.")
            continue

        print(f"[get_outlier_report] 🔎 Analyzing DataFrame: '{df_id}'")
        # Initialize the inner dict for this df_id
        outliers_storage[df_id] = {}
        
        # Loop through the features for this dataframe, e.g., "Insulin", then "Age"
        for feature in feature_list:
            # Get the top 5 rows
            top_k_df = get_top_k(df_to_analyze, feature, k=5)
            
            if not top_k_df.empty:
                print(f"[get_outlier_report]   -> Found and stored top 5 outliers for '{feature}'")
                # Store the resulting DataFrame
                outliers_storage[df_id][feature] = top_k_df
            else:
                print(f"[get_outlier_report]   -> No outliers found or error for '{feature}'")

    # Save the final report back to the state
    state.referralDataFrames.contextual_outlier_report = outliers_storage
    print(f"[get_outlier_report] ✅ Contextual Outlier Report complete.")
    print({'outlier_report': outliers_storage})




    return state

def put_outliers(state: MainState):
    """
    This is the "Specialized Outlier Injection Node."
    
    It uses the rich "Contextual Outlier Report" to generate a chunk of
    pure, context-aware outliers using a high-intelligence model.
    """
    print("[put_outliers] 🚀 Triggered OUTLIER INJECTION node...")
    
    # 1. Get all the necessary "facts" from the state
    outliers_report_map = state.referralDataFrames.contextual_outlier_report
    blueprint = state.referralDataFrames.primarySummary
    schema = state.synthData.schemaOfData
    # Use chunk_size (e.g., 5 rows) to define the injection amount
    chunk_size = state.synthData.chunk_size 
    
    # Assuming groq, gemini, openrouter, and json_output are available in the scope
    heavy_groq = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"), temperature=1.5)    

    groq_generator = json_output(heavy_groq, schema)
    gemini_generator = json_output(gemini, schema)
    openrouter_generator = json_output(openrouter, schema)

    # 2. Process the Outlier Report DataFrames into a single JSON string
    # We describe the top K rows to get a token-efficient summary of the context
    outliers_json_report = {}
    
    if outliers_report_map:
        for df_id, feature_dfs in outliers_report_map.items():
            outliers_json_report[df_id] = {}
            for feature, df_outliers in feature_dfs.items():
                # Convert the describe output of the top 5 rows to JSON string
                outliers_json_report[df_id][feature] = df_outliers.describe().to_json() 

    # 3. Craft the highly directive prompt
   
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
        ````

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




    # 4. Generate Chunk using specialized models (Fallback is critical here)
    chunk = None
    try:
        print("[put_outliers] Attempting Groq (Specialized Model)...")
        chunk = groq_generator.invoke(prompt)
    except:
        try:
            print("[put_outliers] Groq failed. Trying Gemini...")
            chunk = gemini_generator.invoke(prompt)
        except:
            try:
                print("[put_outliers] Gemini failed. Trying OpenRouter...")
                chunk = openrouter_generator.invoke(prompt)
            except:
                raise ValueError("[put_outliers] Failed to generate outlier chunk with all specialized models.")
    
    # 5. Update State
    state.synthData.chunk = chunk
    print(f"[put_outliers] ✅ Successfully generated {chunk_size} outlier rows.")
    print({'outlier chunk': chunk})
    return state

def condition_for_adding_outliers(state: MainState) -> Literal['generate_chunks', 'get_outlier_report']:
    if (len(state.referralDataFrames.stoppage_for_outliers) > 0) and (state.referralDataFrames.stoppage_for_outliers[0] < len(state.synthData.final_df)):
        state.referralDataFrames.stoppage_for_outliers.pop(0)
        return 'get_outlier_report'
    return 'generate_chunks'


def final_node(state: MainState):
    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)
    if state.synthData.synth_data is not None:
        output_path = os.path.join(output_folder, "final_dataframe.csv")
        state.synthData.synth_data.to_csv(output_path, index=False)
        print(f"Final DataFrame saved to {output_path}")
    return state

# --- Graph Definition ---

graph = StateGraph(MainState)


# 1. Add all nodes
graph.add_node('generate_summary', generate_summary)
graph.add_node('summarize_summary', summarize_summary) 
graph.add_node('generate_chunks', generate_chunks)
graph.add_node('get_chunk_metadata', get_chunk_metadata)
graph.add_node('validate_chunk', validate_chunk)
graph.add_node('append_chunk', append_chunk)
graph.add_node('get_outlier_report', get_outlier_report)
graph.add_node('put_outliers', put_outliers)
graph.add_node('generate_synthetic_data', generate_synthetic_data)
graph.add_node('final_node', final_node)

graph.set_entry_point('generate_summary')
graph.add_edge('generate_summary', 'summarize_summary')
graph.add_edge('summarize_summary', 'generate_chunks')
graph.add_edge('generate_chunks', 'get_chunk_metadata')
# graph.add_edge('get_chunk_metadata', 'validate_chunk') 

# 2. ADD this new conditional edge:
graph.add_conditional_edges(
    'get_chunk_metadata',
    route_after_metadata,
    {
        "validate_chunk": "validate_chunk",
        "generate_chunks": "generate_chunks" # This is the loop-back
    }
)

# Conditional edges
graph.add_conditional_edges(
    'validate_chunk',
    after_validate_chunk,
    {
        'generate_synthetic_data': 'generate_synthetic_data',
        'final_node': 'final_node',
        'append_chunk': 'append_chunk',
        'generate_chunks': 'generate_chunks'
    }
)

graph.add_conditional_edges('append_chunk', condition_for_adding_outliers, {
    'generate_chunks': 'generate_chunks',
    'get_outlier_report': 'get_outlier_report'
})

graph.add_edge('get_outlier_report', 'put_outliers')
graph.add_edge('put_outliers', 'append_chunk')




# --- ADD THIS LINE ---
graph.add_edge('generate_synthetic_data', 'final_node') # Connect success path to the end
# --- END OF FIX ---

graph.set_finish_point('final_node')

# --- Pydantic Schemas for Data ---

df1 = pd.read_csv("Data/diabetes.csv") # Make sure this path is correct

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
    
# --- To track 'age' (as before) ---
# --- To track multiple features ---
state = MainState(
    referralDataFrames=ReferralDataFrames(
        dataframes={
            "Real Data": df1  # Actual dataframe stored here
        },
        summary={},  # Will store AutoEDA full JSON report
        primarySummary=None,  # Condensed blueprint (to be generated later)
        features_list_for_outlier_report={
            "Real Data": ["Insulin", "Age"]  # Example feature tracking
        },
        contextual_outlier_report=None,  # Will be filled by get_outlier_report node
        stoppage_for_outliers=[20, 70]  # Empty list to start tracking stopped features
    ),
    synthData=SynthData(
        schemaOfData=ChunksRecords,
        diversity_features=["exercise_frequency", "diabetes_diagnosis"],
        seed_length=100,   # Controlled seed for reproducibility
        chunk_size=10      # Data chunk size
    ),
    validationClass=ValidationClass(
        advice="Start by generating a diverse first chunk, paying attention to both 'exercise_frequency' and 'diabetes_diagnosis'."
    )
)

# --- To track 'exercise_frequency' instead ---

# --- Graph Execution (The Fix) ---

final_graph = graph.compile()

print("--- Invoking Graph ---")
# 1. This is the fix: Add the 'recursion_limit' config
# 2. Capture the result in 'final_state'
final_state = final_graph.invoke(
    state,
    config={"recursion_limit": 100} # Allow more steps
)

# 3. Print the final state from the returned object

# 4. Check if final_df exists before trying to print .describe()
if final_state['synthData'].final_df is not None:
    from rich import print
    print("\n[bold green]Real Data (Input df1):[/bold green]")
    print(df1.describe())
    print("\n[bold cyan]Synthetic Data (final_state['synthData'].synth_data):[/bold cyan]")
    print(final_state['synthData'].synth_data.describe())
   
    
else:
    print("\n--- No Final DataFrame generated ---")


def save_dataframe_to_csv(df, filename):
    """
    Saves the given DataFrame to a CSV file with the specified filename.
    
    Args:
        df (pd.DataFrame): The DataFrame to be saved.
        filename (str): The filename (including path, if needed) to save the CSV.
    """
    if df is not None and not df.empty:
        df.to_csv(filename, index=False)
        print(f"[save_dataframe_to_csv] ✅ DataFrame saved to '{filename}' ({len(df)} rows).")
    else:
        print(f"[save_dataframe_to_csv] ⚠️ No data to save for '{filename}'. DataFrame is None or empty.")

