import json
import os
from typing import List, Literal, Optional, Type, Dict, Any
import itertools # We'll need this for crosstabs

import pandas as pd
from dotenv import load_dotenv
from rich import print
from pydantic import BaseModel, Field

# --- LangChain Imports ---
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

# --- REMOVED THE EXTRACTOR IMPORT ---
# We no longer want the validator to call the extractor.
# try:
#     from automated_eda_extractor import DataRelationshipExtractor
# except ImportError:
#     ...

# Load environment variables (.env file)
load_dotenv()

# --- 1. Pydantic Models for Validation Output (Unchanged) ---

class ValidationReport(BaseModel):
    """The final structured report comparing target vs. synthetic data."""
    overall_verdict: Literal['High Quality Match', 'Good Match', 'Partial Match', 'Significant Mismatch'] = Field(
        description="The final verdict on data quality."
    )
    overall_score: int = Field(
        description="A quantitative score from 0 (mismatch) to 100 (perfect match) of the synthetic data's quality.",
        ge=0, le=100
    )
    comparative_summary: str = Field(
        description="A high-level summary explaining the verdict and score, highlighting key matches and deviations."
    )
    key_matches: List[str] = Field(
        description="A bulleted list of specific statistical points that matched well (e.g., 'Mean age is aligned', 'Correlation of X/Y is correct')."
    )
    key_mismatches: List[str] = Field(
        description="A bulleted list of specific statistical points that *did not* match (e.g., 'Mean BMI is 10% lower than target', 'Categorical distribution for 'job' is incorrect.')."
    )
    recommendation: Literal['Proceed', 'Proceed with Caution', 'Adjust Prompts and Regenerate'] = Field(
        description="Actionable advice for the data generation process."
    )

# --- 2. The Validator Class (Updated) ---

class SyntheticDataValidator:
    """
    Automates the validation of synthetic data by comparing its
    RAW STATS against a target (original) summary.
    """
    
    def __init__(self):
        """Initializes the LLM models."""
        print("[cyan]Initializing SyntheticDataValidator...[/cyan]")
        self.models = self._init_models()
        # --- REMOVED self.extractor ---
        # self.extractor = DataRelationshipExtractor()
        print("[green]...Validator initialized.[/green]")

    def _init_models(self) -> List[Any]:
        """
        Initializes all the LLM models specified by the user.
        (Code is identical to before, so truncated for brevity)
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
            models.append(
                OllamaLLM(model="phi3:3.8b-mini-4k-instruct-q4_K_M", format="json")
            )
            print("[green]Connected to local Ollama (phi3).[/green]")
        except Exception as e:
            print(f"[red]Failed to connect to local Ollama: {e}[/red]")

        if not models:
            raise RuntimeError("No LLM models could be loaded!")
            
        return models


    def _call_llm_with_fallback(
        self, 
        prompt_template: ChatPromptTemplate, 
        input_data: Dict[str, str],
        pydantic_schema: Type[BaseModel]
    ) -> Optional[BaseModel]:
        """
        Tries to call each LLM in order until one succeeds.
        (Code is identical to before)
        """
        for model in self.models:
            try:
                print(f"[cyan]Attempting to use model for validation: {model.__class__.__name__}...[/cyan]")
                chain = prompt_template | model.with_structured_output(pydantic_schema)
                result = chain.invoke(input_data)
                print(f"[green]...Success with {model.__class__.__name__}![/green]")
                return result
            except Exception as e:
                print(f"[yellow]...Failed with {model.__class__.__name__}. Error: {e}[/yellow]")
                continue
        
        print("[red]All LLM models failed![/red]")
        return None

    def _get_raw_stats_string(self, df: pd.DataFrame) -> str:
        """
        NEW: Helper function to generate all raw stat tables from a dataframe.
        """
        raw_stats_str = ""
        
        # --- Univariate ---
        try:
            raw_stats_str += "--- RAW UNIVARIATE STATS (`.describe()`) ---\n"
            raw_stats_str += df.describe(include='all').to_string()
            raw_stats_str += "\n\n"
        except Exception as e:
            raw_stats_str += f"Error generating describe(): {e}\n\n"
            
        # --- Multivariate ---
        try:
            raw_stats_str += "--- RAW CORRELATION STATS (`.corr()`) ---\n"
            raw_stats_str += df.corr(numeric_only=True).to_string()
            raw_stats_str += "\n\n"
        except Exception as e:
            raw_stats_str += f"Error generating corr(): {e}\n\n"
            
        # --- Categorical ---
        try:
            raw_stats_str += "--- RAW CATEGORICAL STATS (`.crosstab()`) ---\n"
            categoricals = df.select_dtypes(include=['object', 'category']).columns
            crosstabs_str_inner = ""
            if len(categoricals) >= 2:
                for pair in itertools.combinations(categoricals, 2):
                    # Limit to first 5 pairs to avoid token overload
                    if len(crosstabs_str_inner) > 4000:
                         crosstabs_str_inner += "\n... (truncated for brevity) ..."
                         break
                    ct = pd.crosstab(df[pair[0]], df[pair[1]])
                    crosstabs_str_inner += f"\nCrosstab for {pair[0]} vs {pair[1]}:\n"
                    crosstabs_str_inner += ct.to_string()
            
            if not crosstabs_str_inner:
                crosstabs_str_inner = "No categorical pairs found."
            
            raw_stats_str += crosstabs_str_inner
            
        except Exception as e:
            raw_stats_str += f"Error generating crosstabs: {e}\n\n"
            
        return raw_stats_str


    def validate(self, synthetic_df: pd.DataFrame, target_summary: Dict[str, Any]) -> Optional[ValidationReport]:
        """
        Runs the full validation workflow (NEW LOGIC).
        
        1. Generates RAW STATS from the new synthetic_df.
        2. Compares the raw stats to the target_summary (rules) using an LLM.
        3. Returns a structured ValidationReport.
        """
        
        # --- Step 1: Analyze the new synthetic data (NEW) ---
        print("\n--- [blue]Step 1: Generating raw stats from new synthetic data...[/blue] ---")
        try:
            raw_stats_string = self._get_raw_stats_string(synthetic_df)
            target_summary_string = json.dumps(target_summary, indent=2)
        except Exception as e:
            print(f"[bold red]Failed to generate raw stats: {e}[/bold red]")
            return None
        print("--- [green]Step 1: Raw stats generated.[/green] ---")


        # --- Step 2: Prepare for LLM-based comparison (NEW PROMPT) ---
        print("\n--- [blue]Step 2: Preparing for LLM comparative analysis...[/blue] ---")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert Data Quality Analyst and Statistician. Your job is to compare a 'Target Data Summary' (which contains rules and findings from real data) against 'Raw Statistical Tables' (from new synthetic data). You must provide a structured JSON report using the 'ValidationReport' format."),
            ("user", """
Please analyze the two inputs below.

Your goal is to determine if the **Synthetic Data's Raw Stats** are a high-quality statistical match for the **Target Data Summary (Rules)**.

-   **Compare Univariate Stats:** Look at the `univariate_analysis` section in the Target Summary. Do the mean, std, min, max, and 'top' values listed there *match* the numbers in the `RAW UNIVARIATE STATS` table?
-   **Compare Correlations:** Look at the `multivariate_correlation` section. Do the strong positive/negative correlations listed there (e.g., "0.97") *match* the numbers in the `RAW CORRELATION STATS` table?
-   **Compare Categorical Rules:** Look at the `categorical_relationships` section. Do the rules (e.g., "combination X never appears") hold true in the `RAW CATEGORICAL STATS` tables?
-   **Assign a Score:** Give a score from 0-100. **If the raw stats tables perfectly match every single rule and finding in the target summary, the score MUST be 100.** A 99 or 95 is not acceptable if they are identical.
-   **Provide a Verdict:** Is this a 'High Quality Match', 'Good Match', 'Partial Match', or 'Significant Mismatch'?
-   **List specific matches and mismatches.**

---
[TARGET DATA SUMMARY (The Rules)]
```json
{target_summary}
```
---
[SYNTHETIC DATA'S RAW STATS (The Numbers to Check)]
```
{raw_stats}
```
---

Now, provide your full analysis in the `ValidationReport` JSON format. Remember, if the Raw Stats perfectly fulfill the Target Summary, the score is 100.
"""),
        ])

        input_data = {
            "target_summary": target_summary_string,
            "raw_stats": raw_stats_string
        }

        # --- Step 3: Run the comparison ---
        report = self._call_llm_with_fallback(
            prompt, 
            input_data,
            ValidationReport
        )
        
        if report:
            print("--- [green]Step 2: Comparative analysis complete.[/green] ---")
            return report
        else:
            print("[bold red]Failed to generate validation report.[/bold red]")
            return None

