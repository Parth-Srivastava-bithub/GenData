import json
import re
import time
from langchain_ollama import OllamaLLM
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI  # replace with correct import
from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph
from pydantic import BaseModel
load_dotenv()
from rich import print
import numpy as np
from typing import List, Literal, Optional, Type
from typing import Optional, List, Literal, Dict, Any
from pydantic import BaseModel, Field
import pandas as pd
from AutoEDA import DataRelationshipExtractor
# Initialize FastAPI

# Initialize AI models
groq = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))
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

class ReferralDataFrames(BaseModel):
    dataframes: List[Any] = Field([], description="Collections of dataframes to get the summary")
    summary: str = Field("", description="Summary from dataframes")
    primarySummary: Optional[Any] = Field(None, description="Summarized import form of summary")

    
class SynthData(BaseModel):
    data: Optional[Any] = Field(None, description="Synthetic dataframe")
    schemaOfData: Optional[Any] = Field(None, description="Schema of needed dataframe")
    temp_df: Optional[Any] = Field(None, description="Temporary Dataframe for validation")
    final_df: Optional[Any] = Field(None, description="Final Dataframe after checking")
    chunk: Optional[Any] = None
    chunk_fingerprints: List[Dict[str, Any]] = Field(default_factory=list)
    current_chunk_fingerprint: Optional[Dict[str, Any]] = Field(None, description="The fingerprint of the chunk currently being validated.")
    chunk_size : int = 5
    max_tries: int = 5

class ValidationClass(BaseModel):
    mode: Literal['approved', 'retry'] = Field('approved', description="Based on the quality of data suggest to append this for synthetic data or retry to generate more factual different data which is not similar")
    advice: Optional[str] = Field(None, description="Advice over the given data and its metadata and tell llm if its bad how to and what to make changes")

class MainState(BaseModel):
    referralDataFrames: Optional[ReferralDataFrames] = None
    synthData: Optional[SynthData] = None
    validationClass: Optional[ValidationClass] = None

def generate_summary(state: MainState):
    json_summary = []
    extractor = DataRelationshipExtractor()
    df_list = state.referralDataFrames.dataframes
# final_report = extractor.run_analysis(sample_df)
# print(json.dumps(final_report, indent=2))
    for i in range(0, len(df_list)):
        report = extractor.run_analysis(df_list[i])
        time.sleep(0.2)
        json_summary.append(report)

    state.referralDataFrames.summary = json_summary
    return state

def summarize_summary(state: MainState):
    json_summary = state.referralDataFrames.summary
   
    class AnalysisSection(BaseModel):
        title: str = Field(..., description="Title of the analysis section")
        summary: str = Field(..., description="Summary text of the analysis")
        key_findings: List[str] = Field(default_factory=list, description="Important points extracted from analysis")

    class EDAReport(BaseModel):
        univariate_analysis: Optional[AnalysisSection] = Field(None, description="Univariate statistics summary")
        multivariate_correlation: Optional[AnalysisSection] = Field(None, description="Correlation or multivariate relationships summary")
        categorical_relationships: Optional[AnalysisSection] = Field(None, description="Relationships between categorical features")

        model_config = {
            "arbitrary_types_allowed": True 
        }

    groq_schema = groq.with_structured_output(EDAReport)
    gemini_schema = gemini.with_structured_output(EDAReport)
    openrouter_schema = openrouter.with_structured_output(EDAReport)

    

    prompt = f"""
        Combine and condense the summaries below into a single JSON object of type `EDAReport`. 
        Keep only relationships relevant to the given data schema: {state.synthData.schemaOfData}.

        Summaries:
        {json_summary}

        Schema:
        class AnalysisSection(BaseModel):
            title: str
            summary: str
            key_findings: List[str]

        class EDAReport(BaseModel):
            univariate_analysis: Optional[AnalysisSection]
            multivariate_correlation: Optional[AnalysisSection]
            categorical_relationships: Optional[AnalysisSection]
        Return the final JSON.
        """

    reply = ""
    time.sleep(0.5)
    try:
        reply = groq_schema.invoke(prompt).model_dump()
    except:
        try:
            reply = gemini_schema.invoke(prompt).model_dump()
        except:
            try:
                reply = openrouter_schema.invoke(prompt).model_dump()
            except:
                print("Models failed")
                reply = json_summary

    state.referralDataFrames.primarySummary = reply
    return state

def generate_chunks(state: MainState):
    data_schema = state.synthData.schemaOfData
    groq_generator = json_output(groq, data_schema)
    gemini_generator = json_output(gemini, data_schema)
    openrouter_generator = json_output(openrouter, data_schema)

    prompt = f"""
    Generate the {state.synthData.chunk_size} rows with the given advice: {state.validationClass.advice} factual information with other similar types of data: {state.referralDataFrames.primarySummary} These are the relationships we found from other similar types of data so generate new chunks of data keeping in mind these relationships
    Schema is {data_schema}
    Generated data should be follow the given information and dont get biased
    """
    chunk = None
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

def get_chunk_metadata(state: MainState):
    if not state.synthData or not state.synthData.chunk:
        raise ValueError("No chunk data found in state.")

    # Convert Pydantic model → dict → DataFrame
    try:
        chunk_dict = [row.model_dump() for row in state.synthData.chunk.rows]
        df = pd.DataFrame(chunk_dict)
    except Exception as e:
        print(f"Error converting chunk to DataFrame: {e}")
        # This is a bad state, we should probably stop or retry
        raise ValueError("Failed to create DataFrame from chunk.")

    
    # --- Metadata Creation Part ---
    
    fingerprint = {
        "numerical_stats": {},
        "categorical_stats": {}
    }

    # 1. Get Numerical Stats
    # Select columns that are numeric (int, float)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if not numeric_cols.empty:
        # .describe() gives: count, mean, std, min, 25%, 50%, 75%, max
        # We convert this to a dict for JSON compatibility
        fingerprint["numerical_stats"] = df[numeric_cols].describe().to_dict()

    # 2. Get Categorical Stats
    # Select columns that are objects, categories, or booleans
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns
    
    for col in categorical_cols:
        fingerprint["categorical_stats"][col] = df[col].value_counts(
            normalize=True
        ).to_dict()

    state.synthData.current_chunk_fingerprint = fingerprint
    
    print("--- 1. Generated Fingerprint for New Chunk ---")
    
    state.synthData.max_tries -= 1
    return state

def validate_chunk(state: MainState):
    chunk = state.synthData.chunk

    prompt = f"""
    You are analyzer having domain of knowledge {state.referralDataFrames.primarySummary} data given to u and u have to analyze that giving data are factually correct or they failed or little misguided, means u just need to validate the data that they makes sense or not, this is the data: {chunk}
    """

    validator_groq = json_output(groq, ValidationClass)
    validator_gemini = json_output(gemini, ValidationClass)
    validator_openrouter = json_output(openrouter, ValidationClass)


    try:
        reply = validator_groq.invoke(prompt)
    except:
        try:
            reply = validator_gemini.invoke(prompt)
        except:
            try:
                reply = validator_openrouter.invoke(prompt)
            except:
                raise ValueError("Models Failed do something...")

            
    state.validationClass.mode = reply.mode
    state.validationClass.advice = reply.advice
    
    return state

def after_validate_chunk(state: MainState) -> Literal['append_chunk', 'generate_chunks', 'final_node']:
    if state.synthData.max_tries < 0 or (state.synthData.final_df is not None and len(state.synthData.final_df) > 20):
        return 'final_node'
    if (state.validationClass.mode == 'approved'):
        return 'validate_chunk_with_metadata'
    return 'generate_chunks'

def validate_chunk_with_metadata(state: MainState):
    if len(state.synthData.chunk_fingerprints) == 0:
        state.synthData.chunk_fingerprints.append(state.synthData.current_chunk_fingerprint)
        state.validationClass.mode = 'approved'
        return state

    chunks_metadata = state.synthData.chunk_fingerprints
    current_chunk_metadata = state.synthData.current_chunk_fingerprint


    for i in range(0, len(chunks_metadata), 5):
        chunks = chunks_metadata[i:i+5]
        prompt = f"""You r analyzer and here are the prev 5 chunks: {chunks} and this is the current chunk: {current_chunk_metadata}, 
        You are have to analyze and check that does this current chunk showing similar property to others cuz we want diverse data yet factual and accurate correct data, cuz this used to trained models
        """
    
        validator_groq = json_output(groq, ValidationClass)
        validator_gemini = json_output(gemini, ValidationClass)
        validator_openrouter = json_output(openrouter, ValidationClass)

        try:
            reply = validator_groq.invoke(prompt)
        except:
            try:
                reply = validator_gemini.invoke(prompt)
            except:
                try:
                    reply = validator_openrouter.invoke(prompt)
                except:
                    raise ValueError("Models Failed do something...")

                
        state.validationClass.mode = reply.mode
        state.validationClass.advice = reply.advice

        if (reply.mode == 'retry'):
            return state
        
    return state

def after_chunks_metadata_validation(state: MainState) -> Literal['append_chunk', 'generate_chunks']:
    if (state.validationClass.mode == 'retry'):
        return 'generate_chunks'
    return 'append_chunk'



def append_chunk(state: MainState):
    # Add current fingerprint to history
    state.synthData.chunk_fingerprints.append(state.synthData.current_chunk_fingerprint)
    state.synthData.current_chunk_fingerprint = None
    state.synthData.max_tries = 5

    def add_chunk_in_df(state: MainState):
        try:
            # Convert chunk rows (list of DiabetesRecord) → list of dicts → DataFrame
            chunk_rows = [row.model_dump() for row in state.synthData.chunk.rows]
            chunk_df = pd.DataFrame(chunk_rows)

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
            print(f"[append_chunk] ❌ Error adding chunk to DataFrame: {e}")

        return state

    state = add_chunk_in_df(state)
    return state





def final_node(state: MainState):
    print("Max Tries but not solved")
    return state




graph = StateGraph(MainState)

graph.add_node('generate_summary', generate_summary)
graph.add_node('summarize_summary', summarize_summary)
graph.add_node('generate_chunks', generate_chunks)
graph.add_node('get_chunk_metadata', get_chunk_metadata)
graph.add_node('validate_chunk', validate_chunk)
graph.add_node('after_validate_chunk', after_validate_chunk)
graph.add_node('validate_chunk_with_metadata', validate_chunk_with_metadata)
graph.add_node('after_chunks_metadata_validation', after_chunks_metadata_validation)
graph.add_node('append_chunk', append_chunk)
graph.add_node('final_node', final_node)


graph.set_entry_point('generate_summary')
graph.add_edge('generate_summary', 'summarize_summary')
graph.add_edge('summarize_summary', 'generate_chunks')
graph.add_edge('generate_chunks', 'get_chunk_metadata')
graph.add_edge('get_chunk_metadata', 'validate_chunk')
graph.add_conditional_edges('validate_chunk', after_validate_chunk)
graph.add_conditional_edges('validate_chunk_with_metadata', after_chunks_metadata_validation)
graph.add_edge('append_chunk', 'generate_chunks')
graph.set_finish_point('final_node')



df1 = pd.read_csv("Data/banking.csv")

from pydantic import BaseModel, Field
from typing import Literal

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
    
state = MainState(
    referralDataFrames=ReferralDataFrames(
        dataframes=[df1]
    ),
    synthData=SynthData(
        schemaOfData=ChunksRecords,
    ),
    validationClass=ValidationClass()  # <- yahi zaroori line hai
)



final_graph = graph.compile()
print(final_graph.invoke(state))

print(state.synthData.final_df.describe())
print(state.synthData.final_df.info())



    
