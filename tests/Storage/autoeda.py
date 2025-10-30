import json
import os
import re
import itertools
from typing import List, Literal, Optional, Type, Dict, Any, TypedDict

import pandas as pd
from dotenv import load_dotenv
from rich import print
# from pydantic import BaseModel as PydanticBaseModel, Field # This line was unused

# Use langchain_core.pydantic_v1 to avoid v1/v2 conflicts
# from langchain_core.pydantic_v1 import BaseModel, Field # <-- REMOVED
from pydantic import BaseModel, Field # <-- ADDED: Use Pydantic V2

# --- LangChain Imports ---
from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.outputs import PydanticOutputParser # <-- REMOVED
from langchain_ollama.llms import OllamaLLM
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

# Load environment variables (.env file)
load_dotenv()

# --- 1. Pydantic Models for Structured Output ---
# This defines the structure of our final JSON report.

class ReportSection(BaseModel):
    """A single section of the analysis report."""
    title: str = Field(description="Title of the analysis section (e.g., 'Univariate Analysis')")
    summary: str = Field(description="A concise summary of the key findings, focusing on numerical relationships and statistical facts.")
    key_findings: List[str] = Field(description="A bulleted list of the most important relationships or anomalies found.")

class FinalReport(BaseModel):
    """The final structured report of all EDA phases."""
    univariate_analysis: Optional[ReportSection] = Field(default=None, description="Analysis of individual features.")
    multivariate_correlation: Optional[ReportSection] = Field(default=None, description="Analysis of correlations between numeric features.")
    categorical_relationships: Optional[ReportSection] = Field(default=None, description="Analysis of relationships between categorical features.")

# --- 2. LangGraph State ---
# This dictionary defines the "memory" of our graph.
# It's passed between all the phases (nodes).

class GraphState(TypedDict):
    """
    Defines the state of our workflow.
    
    Attributes:
        df: The pandas DataFrame being analyzed.
        report: The accumulating JSON report.
        current_phase_output: A string (e.g., a .describe() table) to be summarized by an LLM.
        error: A string to hold any error messages.
    """
    df: pd.DataFrame
    report: Dict[str, Any]
    current_phase_output: str
    error: Optional[str]

# --- 3. The Main Extractor Class ---

class DataRelationshipExtractor:
    """
    Automates EDA to extract data relationships using a LangGraph agent
    and a fallback chain of LLMs.
    """
    
    def __init__(self):
        """Initializes the class and the LLM models."""
        self.models = self._init_models()
        self.graph = self._build_graph()

    def _init_models(self) -> List[Any]:
        """
        Initializes all the LLM models specified by the user.
        Models that fail to load (e.g., missing API key) are skipped.
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
                # Using a standard, available model
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
                        model_name="meituan/longcat-flash-chat:free", # Using a known free model
                        base_url="https://openrouter.ai/api/v1"
                    )
                )
            else:
                print("[yellow]Skipping OpenRouter: OPENROUTER_API_KEY not found.[/yellow]")
        except Exception as e:
            print(f"[red]Failed to load OpenRouter: {e}[/red]")
            
        # --- Local Ollama (Phi-3) ---
        try:
            # We use the JSON-formatted model for structured output
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
                
                # We chain the prompt, model, and output parser
                # .with_structured_output is the modern way to force JSON
                # based on a Pydantic model.
                
                # Per your feedback, we'll rely *only* on with_structured_output,
                # as all our initialized models (incl. Ollama(format="json")) support this.
                chain = prompt_template | model.with_structured_output(pydantic_schema)
                result = chain.invoke(input_data)

                # REMOVED the 'else' block that used the deprecated PydanticOutputParser

                print(f"[green]...Success with {model.__class__.__name__}![/green]")
                return result
            except Exception as e:
                print(f"[yellow]...Failed with {model.__class__.__name__}. Error: {e}[/yellow]")
                continue
        
        print("[red]All LLM models failed![/red]")
        return None

    # --- Graph Node Definitions (The "Phases") ---

    def node_univariate_stats(self, state: GraphState) -> Dict[str, Any]:
        """Phase 1: Generate univariate statistics."""
        print("--- Phase 1: Running Univariate Analysis ---")
        df = state['df']
        try:
            # .describe(include='all') gets stats for BOTH numeric and categorical
            desc_string = df.describe(include='all').to_string()
            return {"current_phase_output": desc_string}
        except Exception as e:
            return {"error": f"Univariate analysis failed: {e}"}

    def node_summarize_univariate(self, state: GraphState) -> Dict[str, Any]:
        """Summarizes the results of Phase 1."""
        print("--- Phase 1: Summarizing Univariate Analysis ---")
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a precise data analyst. Summarize univariate stats into a short JSON report ('ReportSection' format). Keep only core insights."),
            ("user", """
            Given the summary stats from `data.describe(include='all')`:
            Statistics:
            ```
            {stats}
            ```

            Report:
            1. Mean, min, max for numericals.
            2. Unique count, top value, and freq for categoricals.
            3. Any anomalies (weird min/max, std=0).
            Keep it compact and structured.
        """)
        ,])
        
        section = self._call_llm_with_fallback(
            prompt, 
            {"stats": state['current_phase_output']},
            ReportSection
        )
        
        report = state.get('report', {})
        if section:
            report['univariate_analysis'] = section.dict()
        else:
            return {"error": "LLM summarization failed for univariate stats."}
            
        return {"report": report, "current_phase_output": ""} # Clear output for next step

    def node_multivariate_stats(self, state: GraphState) -> Dict[str, Any]:
        """Phase 2: Generate correlation matrix."""
        print("--- Phase 2: Running Multivariate (Correlation) Analysis ---")
        df = state['df']
        try:
            # This is the "numerical graph" you wanted
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
            3. Variables near 0 correlation.
            4. Any pairs showing multicollinearity (|corr| > 0.8).
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
            report['multivariate_correlation'] = section.dict()
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
            
            # Only run if there are 2 or more categorical features
            if len(categoricals) >= 2:
                # Get all unique pairs
                for pair in itertools.combinations(categoricals, 2):
                    # pd.crosstab is the "numerical bar chart"
                    ct = pd.crosstab(df[pair[0]], df[pair[1]])
                    crosstabs_str += f"\n--- Crosstab for {pair[0]} vs {pair[1]} ---\n"
                    crosstabs_str += ct.to_string()
                    crosstabs_str += "\n"
            
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
        1.  Any strong conditional relationships (e.g., "If Feature_A is 'X', Feature_B is almost always 'Y'").
        2.  Any combinations that *never* appear (a count of 0).
        3.  Any distributions that seem particularly uneven.
        """),
        ])
        
        section = self._call_llm_with_fallback(
            prompt, 
            {"stats": state['current_phase_output']},
            ReportSection
        )
        
        report = state['report']
        if section:
            report['categorical_relationships'] = section.dict()
        else:
            return {"error": "LLM summarization failed for categorical stats."}
            
        return {"report": report, "current_phase_output": ""}

    def _build_graph(self) -> StateGraph:
        """
        Builds the LangGraph workflow.
        """
        workflow = StateGraph(GraphState)

        # --- Add all the nodes (phases) ---
        workflow.add_node("univariate_stats", self.node_univariate_stats)
        workflow.add_node("summarize_univariate", self.node_summarize_univariate)
        workflow.add_node("multivariate_stats", self.node_multivariate_stats)
        workflow.add_node("summarize_multivariate", self.node_summarize_multivariate)
        workflow.add_node("categorical_stats", self.node_categorical_stats)
        workflow.add_node("summarize_categorical", self.node_summarize_categorical)

        # --- Define the edges (the workflow path) ---
        workflow.set_entry_point("univariate_stats")
        workflow.add_edge("univariate_stats", "summarize_univariate")
        workflow.add_edge("summarize_univariate", "multivariate_stats")
        workflow.add_edge("multivariate_stats", "summarize_multivariate")
        workflow.add_edge("summarize_multivariate", "categorical_stats")
        workflow.add_edge("categorical_stats", "summarize_categorical")
        workflow.add_edge("summarize_categorical", END) # End of the graph

        # Compile the graph
        return workflow.compile()

    def run_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Public method to run the entire analysis workflow on a DataFrame.
        
        Args:
            df: The pandas DataFrame to analyze.
            
        Returns:
            A dictionary (JSON) containing the final report.
        """
        print("[bold green]Starting Automated EDA Workflow...[/bold green]")
        
        # We don't want to analyze a huge dataframe, so we sample if it's too large
        if len(df) > 2000:
            print(f"DataFrame is large ({len(df)} rows). Sampling 2000 rows for analysis.")
            df_sample = df.sample(2000)
        else:
            df_sample = df.copy()

        # The initial state to feed into the graph
        initial_state: GraphState = {
            "df": df_sample,
            "report": {},
            "current_phase_output": "",
            "error": None
        }

        # Run the graph
        final_state = self.graph.invoke(initial_state)

        if final_state.get("error"):
            print(f"[bold red]Workflow failed with error: {final_state['error']}[/bold red]")
            return {"error": final_state['error']}
        
        print("[bold green]...Automated EDA Workflow Complete![/bold green]")
        
        # Return the final, compiled report
        return final_state['report']
