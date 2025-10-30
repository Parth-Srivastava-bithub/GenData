import json
import os
import re
import itertools
from typing import List, Literal, Optional, Type, Dict, Any, TypedDict

import pandas as pd
from dotenv import load_dotenv
from rich import print
from pydantic import BaseModel, Field # Using Pydantic V2

# --- LangChain Imports ---
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

# Load environment variables (.env file)
load_dotenv()

# --- 1. Pydantic Models for Structured Output ---

class ReportSection(BaseModel):
    """A single section of the analysis report."""
    title: str = Field(description="Title of the analysis section (e.g., 'Univariate Analysis')")
    summary: str = Field(description="A concise summary of the key findings, focusing on numerical relationships and statistical facts.")
    key_findings: List[str] = Field(description="A bulleted list of the most important relationships or anomalies found.")

class FinalReport(BaseModel):
    """The final structured report of all EDA phases."""
    deep_feature_profile: Optional[ReportSection] = Field(default=None, description="Deep statistical profile of individual features (skew, zeros, etc.).") 
    multivariate_correlation: Optional[ReportSection] = Field(default=None, description="Analysis of correlations between numeric features.")
    categorical_relationships: Optional[ReportSection] = Field(default=None, description="Analysis of relationships between categorical features.")

# <-- NEW: Pydantic model for the final comparison summary
class ComparativeReport(BaseModel):
    """The final comparative summary between two or more datasets."""
    title: str = Field(default="Comparative Analysis Summary")
    overall_match_rating: Literal["Poor", "Fair", "Good", "Excellent"] = Field(description="A single rating of how well the 'synthetic' dataset matches the 'real' one.")
    summary: str = Field(description="A high-level summary of how well the datasets match, noting the most critical similarities and differences.")
    key_differences: List[str] = Field(description="Bullet points of the most significant differences (e.g., 'Insulin skew is 10.5 in Real Data but 1.2 in Synthetic').")
    key_similarities: List[str] = Field(description="Bullet points of what the synthetic model captured well (e.g., 'Categorical distributions for Gender are a near-perfect match').")


# --- 2. LangGraph State ---

class GraphState(TypedDict):
    """
    Defines the state of our workflow.
    """
    df: pd.DataFrame
    report: Dict[str, Any]
    current_phase_output: str
    error: Optional[str]

# --- 3. The Main Extractor Class ---

class DataRelationshipExtractor:
    """
    Automates EDA to extract data relationships AND deep statistical properties
    using a LangGraph agent. Now includes comparison capabilities.
    """
    
    def __init__(self):
        """Initializes the class and the LLM models."""
        self.models = self._init_models()
        self.graph = self._build_graph()

    def _init_models(self) -> List[Any]:
        """
        Initializes all the LLM models specified by the user.
        """
        models = []
        
        # --- Groq (Llama 3) ---
        try:
            if os.getenv("GROQ_API_KEY"):
                models.append(
                    ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))
                )
            else:
                print("[yellow]Skipping Groq: GROQ_API_KEY not found.[/yellow]")
        except Exception as e:
            print(f"[red]Failed to load Groq: {e}[/red]")

        # --- Google Gemini ---
        try:
            if os.getenv("GEMINI_API_KEY"):
                models.append(
                    ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", api_key=os.getenv("GEMINI_API_KEY"))
                )
            else:
                print("[yellow]Skipping Gemini: GEMINI_API_KEY not found.[/yellow]")
        except Exception as e:
            print(f"[red]Failed to load Gemini: {e}[/red]")

        # --- OpenRouter ---
        try:
            if os.getenv("OPENROUTER_API_KEY"):
                models.append(
                    ChatOpenAI(
                        api_key=os.getenv("OPENROUTER_API_KEY"),
                        model_name="meituan/longcat-flash-chat:free",
                        base_url="https://openrouter.ai/apir/v1"
                    )
                )
            else:
                print("[yellow]Skipping OpenRouter: OPENROUTER_API_KEY not found.[/yellow]")
        except Exception as e:
            print(f"[red]Failed to load OpenRouter: {e}[/red]")
            
        # --- Local Ollama (Phi-3) ---
        try:
            models.append(
                OllamaLLM(model="phi3:3.8b-mini-4k-instruct-q4_K_M", format="json")
            )
            print("[green]Connected to local Ollama (phi3).[/green]")
        except Exception as e:
            print(f"[red]Failed to connect to local Ollama: {e}[/red]")
            print("[yellow]Please ensure Ollama is running with 'ollama pull phi3:3.8b-mini-4k-instruct-q4_K_M'[/yellow]")

        if not models:
            raise RuntimeError("No LLM models could be loaded! Please check your API keys or Ollama connection.")
            
        return models
    def _call_llm_with_fallback(
        self, 
        prompt_template: ChatPromptTemplate, 
        input_data: Dict[str, str],
        pydantic_schema: Type[BaseModel]
    ) -> Optional[BaseModel]:
        """
        Tries to call each LLM in order until one succeeds.
        Uses structured output to force the LLM to return a Pydantic model.
        """
        for model in self.models:
            try:
                print(f"[cyan]Attempting to use model: {model.__class__.__name__}...[/cyan]")
                chain = prompt_template | model.with_structured_output(pydantic_schema)
                result = chain.invoke(input_data)
                print(f"[green]...Success with {model.__class__.__name__}![/green]")
                return result
            except Exception as e:
                print(f"[yellow]...Failed with {model.__class__.__name__}. Error: {e}[/yellow]")
                continue
        
        print("[red]All LLM models failed![/red]")
        return None

    def _get_deep_profile(self, df: pd.DataFrame) -> str:
        """
        Generates a deep statistical profile for each column, including
        skew, kurtosis, and zero/missing percentages.
        """
        profile_list = []
        for col in df.columns:
            col_data = df[col]
            stats = {
                "column": col,
                "dtype": str(col_data.dtype),
                "missing_pct": col_data.isnull().mean() * 100
            }
            
            if pd.api.types.is_numeric_dtype(col_data):
                stats.update({
                    "mean": col_data.mean(),
                    "std": col_data.std(),
                    "min": col_data.min(),
                    "25%": col_data.quantile(0.25),
                    "50%": col_data.median(),
                    "75%": col_data.quantile(0.75),
                    "max": col_data.max(),
                    "skew": col_data.skew(),
                    "kurtosis": col_data.kurt(),
                    "zeros_pct": (col_data == 0).mean() * 100
                })
            else:
                stats.update({
                    "unique_count": col_data.nunique(),
                    "top_value": col_data.mode().iloc[0] if not col_data.empty and len(col_data.mode()) > 0 else None,
                    "top_freq_pct": (col_data.value_counts(normalize=True).iloc[0] * 100) if not col_data.empty else None
                })
            profile_list.append(stats)
        
        profile_df = pd.DataFrame(profile_list).set_index("column")
        return profile_df.to_string()

    # --- Graph Node Definitions (The "Phases") ---

    def node_deep_profile_stats(self, state: GraphState) -> Dict[str, Any]:
        """Phase 1: Generate deep statistical profile."""
        print("--- Phase 1: Running Deep Feature Profile Analysis ---")
        df = state['df']
        try:
            profile_string = self._get_deep_profile(df)
            return {"current_phase_output": profile_string}
        except Exception as e:
            return {"error": f"Deep Profile analysis failed: {e}"}

    def node_summarize_deep_profile(self, state: GraphState) -> Dict[str, Any]:
        """Summarizes the results of Phase 1."""
        print("--- Phase 1: Summarizing Deep Feature Profile ---")
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a precise data analyst. Summarize this deep data profile into a JSON report ('ReportSection' format). Focus on actionable insights for data generation."),
            ("user", """
            Given the deep profile stats:
            Statistics:
            ```
            {stats}
            ```

            Report on:
            1.  **Numeric Features:** Key stats (mean, std, min, max).
            2.  **Missing Data:** Any columns with `missing_pct` > 0.
            3.  **Zero Values:** Any non-binary numeric columns with high `zeros_pct` (e.g., > 10%)? These might be placeholders for missing data.
            4.  **Skewness:** Any columns with high `skew` (e.g., |skew| > 2)? Note the direction (positive/right or negative/left).
            5.  **Outliers:** Any columns with high `kurtosis` (e.g., > 3)?
            6.  **Categorical Features:** Report `unique_count` and `top_value`.
            
            Keep it compact and structured.
        """),
        ])
        
        section = self._call_llm_with_fallback(
            prompt, 
            {"stats": state['current_phase_output']},
            ReportSection
        )
        
        report = state.get('report', {})
        if section:
            # Save to the new key in the report
            report['deep_feature_profile'] = section.model_dump() # <-- RENAMED & FIXED
        else:
            return {"error": "LLM summarization failed for deep profile stats."}
            
        return {"report": report, "current_phase_output": ""}

    def node_multivariate_stats(self, state: GraphState) -> Dict[str, Any]:
        """Phase 2: Generate correlation matrix."""
        print("--- Phase 2: Running Multivariate (Correlation) Analysis ---")
        df = state['df']
        try:
            corr_matrix = df.corr(numeric_only=True).to_string()
            return {"current_phase_output": corr_matrix}
        except Exception as e:
            return {"error": f"Multivariate analysis failed: {e}"}

    def node_summarize_multivariate(self, state: GraphState) -> Dict[str, Any]:
        """Summarizes the results of Phase 2."""
        print("--- Phase 2: Summarizing Multivariate Analysis ---")
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a precise data analyst. Output a compact JSON report in 'ReportSection' format summarizing key correlations only."),
            ("user", """
            Given this correlation matrix:
            ```
            {stats}
            ```
            Find and report:
                1. Top 3–5 highest positive correlations.
                2. Top 3–5 strongest negative correlations.
                3. Any pairs showing multicollinearity (|corr| > 0.8).
            Keep output minimal and structured.
            """),
        ])
        
        section = self._call_llm_with_fallback(
            prompt, 
            {"stats": state['current_phase_output']},
            ReportSection
        )
        
        report = state['report']
        if section:
            report['multivariate_correlation'] = section.model_dump() # <-- FIXED
        else:
            return {"error": "LLM summarization failed for multivariate stats."}
            
        return {"report": report, "current_phase_output": ""}

    def node_categorical_stats(self, state: GraphState) -> Dict[str, Any]:
        """Phase 3: Generate crosstabs for categorical features."""
        print("--- Phase 3: Running Categorical Relationship Analysis ---")
        df = state['df']
        try:
            categoricals = df.select_dtypes(include=['object', 'category']).columns
            crosstabs_str = ""
            
            if len(categoricals) >= 2:
                for pair in itertools.combinations(categoricals, 2):
                    ct = pd.crosstab(df[pair[0]], df[pair[1]])
                    crosstabs_str += f"\n--- Crosstab for {pair[0]} vs {pair[1]} ---\n"
                    crosstabs_str += ct.to_string()
                    crosstabs__str += "\n"
            
            if not crosstabs_str:
                crosstabs_str = "No categorical pairs found (less than 2 categorical columns)."
                
            return {"current_phase_output": crosstabs_str}
        except Exception as e:
            return {"error": f"Categorical analysis failed: {e}"}

    def node_summarize_categorical(self, state: GraphState) -> Dict[str, Any]:
        """Summarizes the results of Phase 3."""
        print("--- Phase 3: Summarizing Categorical Analysis ---")
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a concise data analyst. Summarize crosstab outputs into a minimal JSON report using 'ReportSection'. Keep only key conditional links and anomalies."),
            ("user", """
            Given these crosstab tables:
            ```
            {stats}
            ```
            Focus on:
            1.  Any strong conditional relationships.
            2.  Any combinations that *never* appear (a count of 0).
            """),
        ])
        
        section = self._call_llm_with_fallback(
            prompt, 
            {"stats": state['current_phase_output']},
            ReportSection
        )
        
        report = state['report']
        if section:
            report['categorical_relationships'] = section.model_dump() # <-- FIXED
        else:
            return {"error": "LLM summarization failed for categorical stats."}
            
        return {"report": report, "current_phase_output": ""}

    def _build_graph(self) -> StateGraph:
        """
        Builds the LangGraph workflow.
        """
        workflow = StateGraph(GraphState)

        workflow.add_node("deep_profile_stats", self.node_deep_profile_stats)
        workflow.add_node("summarize_deep_profile", self.node_summarize_deep_profile)
        workflow.add_node("multivariate_stats", self.node_multivariate_stats)
        workflow.add_node("summarize_multivariate", self.node_summarize_multivariate)
        workflow.add_node("categorical_stats", self.node_categorical_stats)
        workflow.add_node("summarize_categorical", self.node_summarize_categorical)

        workflow.set_entry_point("deep_profile_stats")
        workflow.add_edge("deep_profile_stats", "summarize_deep_profile")
        workflow.add_edge("summarize_deep_profile", "multivariate_stats")
        workflow.add_edge("multivariate_stats", "summarize_multivariate")
        workflow.add_edge("summarize_multivariate", "categorical_stats")
        workflow.add_edge("categorical_stats", "summarize_categorical")
        workflow.add_edge("summarize_categorical", END)

        return workflow.compile()

    def run_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Public method to run the entire analysis workflow on a DataFrame.
        """
        print(f"[bold green]Starting Automated EDA Workflow for DataFrame...[/bold green]")
        
        if len(df) > 2000:
            print(f"DataFrame is large ({len(df)} rows). Sampling 2000 rows for analysis.")
            df_sample = df.sample(2000)
        else:
            df_sample = df.copy()

        initial_state: GraphState = {
            "df": df_sample,
            "report": {},
            "current_phase_output": "",
            "error": None
        }

        final_state = self.graph.invoke(initial_state)

        if final_state.get("error"):
            print(f"[bold red]Workflow failed with error: {final_state['error']}[/bold red]")
            return {"error": final_state['error']}
        
        print(f"[bold green]...Automated EDA Workflow Complete.[/bold green]")
        
        return final_state['report']

    # <-- NEW: Method to generate the final comparative summary
    def generate_comparative_summary(self, reports_dict: Dict[str, Dict]) -> Optional[Dict[str, Any]]:
        """
        Compares two or more generated reports using an LLM.

        Args:
            reports_dict: A dictionary where keys are names (e.g., "Real Data")
                          and values are the report dictionaries from run_analysis.
        
        Returns:
            A dictionary containing the comparative analysis.
        """
        print("\n[bold green]Starting Comparative Analysis...[/bold green]")
        
        # Convert the dictionary of reports into a single string for the prompt
        reports_json_string = json.dumps(reports_dict, indent=2)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert data quality analyst. Your task is to compare the statistical profiles of multiple datasets (e.g., 'Real Data' vs. 'Synthetic Data') and provide a structured JSON comparison in the 'ComparativeReport' format."),
            ("user", """
            Please analyze the following data profiles. The "Real Data" is the ground truth.
            The "Synthetic Data" is an attempt to replicate it.
            
            Pay close attention to mismatches in:
            1.  **Deep Profile:** `skew`, `kurtosis`, and `zeros_pct`. A high `zeros_pct` in the real data (e.g., for 'BloodPressure' or 'BMI') that is missing in the synthetic data is a critical failure. High skew (e.g., 'Insulin') should also be replicated.
            2.  **Correlations:** Are the strong positive/negative correlations from the real data present in the synthetic data?
            
            Give a final "overall_match_rating" based on how well the synthetic data captured these nuances.

            Data Reports:
            ```json
            {reports_json}
            ```
            """),
        ])
        
        comparison = self._call_llm_with_fallback(
            prompt,
            {"reports_json": reports_json_string},
            ComparativeReport
        )
        
        if comparison:
            print("[bold green]...Comparative Analysis Complete![/bold green]")
            return comparison.model_dump() # <-- FIXED
        else:
            print("[bold red]Comparative analysis failed (all LLMs failed).[/bold red]")
            return None


# --- UPDATED: Example Usage for Multiple DataFrames ---
if __name__ == "__main__":
    
    # --- This is where you would load your real data ---
    # Example:
    # df_real = pd.read_csv("path/to/real_data.csv")
    # df_synthetic = pd.read_csv("path/to/synthetic_data.csv")
    
    # For this example, we'll create two dummy dataframes
    # that mimic the "real" vs. "synthetic" problem you had.
    
    # "Real" data (from Pima dataset - has placeholders and skew)
    data_real = {
        'Glucose': [148, 85, 183, 89, 137, 116, 78, 0, 197, 125],
        'BloodPressure': [72, 66, 64, 66, 40, 74, 50, 0, 70, 96],
        'Insulin': [0, 0, 0, 94, 168, 0, 88, 543, 0, 0], # High skew, lots of zeros
        'BMI': [33.6, 26.6, 23.3, 28.1, 43.1, 25.6, 31.0, 35.3, 30.5, 0], # Has zeros
        'Age': [50, 31, 32, 21, 33, 30, 26, 29, 53, 54]
    }
    df_real = pd.DataFrame(data_real)
    
    # "Synthetic" data (mimics a generator that missed the nuances)
    data_synthetic = {
        'Glucose': [130, 90, 150, 100, 120, 105, 85, 110, 180, 130], # Different mean/std
        'BloodPressure': [70, 68, 65, 68, 72, 75, 60, 70, 78, 80], # No zeros
        'Insulin': [50, 60, 45, 80, 100, 55, 70, 120, 60, 50], # No skew, no zeros
        'BMI': [30.1, 27.2, 25.1, 29.0, 35.0, 26.1, 32.0, 33.0, 29.5, 28.8], # No zeros
        'Age': [45, 35, 30, 25, 30, 33, 28, 30, 48, 50] # Different range
    }
    df_synthetic = pd.DataFrame(data_synthetic)

    # 1. Create a dictionary of the dataframes to process
    dataframes_to_process = {
        "Real Data": df_real,
        "Synthetic Data": df_synthetic
    }
    
    all_reports = {}
    
    try:
        # 2. Initialize the extractor ONCE
        extractor = DataRelationshipExtractor()

        # 3. Loop and run analysis on each DataFrame
        for name, df in dataframes_to_process.items():
            print("\n" + "=" * 50)
            print(f"[bold magenta] RUNNING ANALYSIS FOR: {name} [/bold magenta]")
            print("=" * 50)
            
            report = extractor.run_analysis(df)
            all_reports[name] = report
            
            # Print the individual report
            print(f"\n[bold cyan] Individual Report for {name}: [/bold cyan]")
            print(json.dumps(report, indent=2))

        # 4. After all analyses are done, generate the final comparison
        if len(all_reports) > 1:
            print("\n" + "=" * 50)
            print(f"[bold magenta] RUNNING FINAL COMPARATIVE SUMMARY [/bold magenta]")
            print("=" * 50)
            
            comparative_summary = extractor.generate_comparative_summary(all_reports)
            
            if comparative_summary:
                print(f"\n[bold cyan] Final Comparative Report: [/bold cyan]")
                print(json.dumps(comparative_summary, indent=2))
            else:
                print("[bold red]Could not generate comparative summary.[/bold red]")

    except Exception as e:
        print(f"\n[bold red]An error occurred during initialization or execution: {e}[/bold red]")

