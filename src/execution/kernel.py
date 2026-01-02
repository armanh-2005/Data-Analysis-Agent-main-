import sys
import os
import io
import json
import sqlite3
import traceback
import contextlib
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict, is_dataclass
from pathlib import Path

# Set Matplotlib backend to 'Agg' to prevent GUI errors on servers
matplotlib.use('Agg')

@dataclass
class ExecutionResult:
    """Structure for the output of code execution."""
    success: bool
    output: str              # Complete output text (logs + errors)
    artifacts: List[str]     # List of paths to generated files (images, etc.)
    error_trace: Optional[str] = None  # Stores the traceback if an error occurs

class JupyterKernel:
    """
    Sandboxed Execution Engine.
    This class is responsible for maintaining variable state in memory and executing Python code.
    It acts like a persistent Jupyter kernel.
    """
    
    def __init__(self, db_path: str, artifacts_dir: str):
        self.db_path = db_path
        self.artifacts_dir = artifacts_dir
        
        # Persistent Kernel Memory (Global Scope)
        # All defined variables (df, results, ...) remain here for the lifecycle of this instance.
        self.scope: Dict[str, Any] = {}
        
        # Initial setup (Standard imports)
        self._bootstrap_scope()

    def _bootstrap_scope(self):
        """Injects vital libraries and helper functions into the execution scope."""
        # 1. Standard Libraries
        self.scope.update({
            "pd": pd, 
            "np": np, 
            "sqlite3": sqlite3, 
            "plt": plt, 
            "json": json, 
            "os": os,
            "sys": sys,
            "is_dataclass": is_dataclass, 
            "asdict": asdict,
            "Path": Path
        })

        # 2. Environment Variables
        self.scope["artifacts_dir"] = self.artifacts_dir
        self.scope["RESULTS"] = {}   # To store numerical results/summaries
        self.scope["ARTIFACTS"] = [] # To manually store artifact paths if needed

        # 3. Helper Functions
        # Lazy loading data function (executes only when called)
        from src.db.repository import SQLiteRepository
        
        def fetch_wide_dataframe(questionnaire_id: str) -> pd.DataFrame:
            repo = SQLiteRepository(self.db_path)
            return repo.fetch_wide_dataframe(questionnaire_id)
            
        self.scope["fetch_wide_dataframe"] = fetch_wide_dataframe

        # Import project-specific tools (if they exist)
        try:
            from src.tools import political, stats, viz
            self.scope.update({
                "political": political,
                "stats": stats,
                "viz": viz
            })
        except ImportError:
            pass # No problem if modules are missing

    def execute(self, code: str, cell_separator: str = "# %%") -> ExecutionResult:
        """
        Executes code cell-by-cell.
        If a cell fails, execution stops, preserving the state up to that point.
        """
        # Ensure the artifacts directory exists
        os.makedirs(self.artifacts_dir, exist_ok=True)
        
        # Clear previous plots from Matplotlib memory (does not delete files)
        plt.clf()
        plt.close('all')

        # Split code into cells
        cells = code.split(cell_separator)
        
        full_log = []
        generated_artifacts = []
        has_error = False
        error_msg = None

        # Scan existing files to identify newly created ones later
        existing_files = set(os.listdir(self.artifacts_dir))

        for i, cell_code in enumerate(cells):
            cell_code = cell_code.strip()
            if not cell_code:
                continue

            cell_id = i + 1
            log_capture = io.StringIO()
            
            # Header for log readability
            header = f"\n--- [CELL {cell_id}] ---\n"
            
            try:
                # Capture stdout (print) and stderr
                with contextlib.redirect_stdout(log_capture):
                    with contextlib.redirect_stderr(log_capture):
                        # Execute code within the persistent scope
                        exec(cell_code, {}, self.scope)
                
                # Log successful output
                output_str = log_capture.getvalue()
                if output_str.strip():
                    full_log.append(f"{header}{output_str}")
                else:
                    full_log.append(f"{header}(Executed successfully)")

            except Exception:
                has_error = True
                # Capture full traceback
                error_msg = traceback.format_exc()
                full_log.append(f"{header}❌ ERROR:\n{error_msg}")
                # Stop execution of subsequent cells (Jupyter-like behavior)
                break
            finally:
                log_capture.close()

        # Identify new artifacts (generated images)
        # Method 1: Check for new files in the directory
        current_files = set(os.listdir(self.artifacts_dir))
        new_files = current_files - existing_files
        for f in new_files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.svg')):
                generated_artifacts.append(str(Path(self.artifacts_dir) / f))
        
        # Method 2: Check ARTIFACTS variable in scope (if manually populated by agent)
        if "ARTIFACTS" in self.scope and isinstance(self.scope["ARTIFACTS"], list):
            for path in self.scope["ARTIFACTS"]:
                if path not in generated_artifacts and os.path.exists(path):
                    generated_artifacts.append(path)

        return ExecutionResult(
            success=not has_error,
            output="\n".join(full_log),
            artifacts=generated_artifacts,
            error_trace=error_msg
        )

    def reset_memory(self):
        """Clears kernel memory (for starting a new analysis)."""
        self.scope = {}
        self._bootstrap_scope()
        # Clear Matplotlib memory
        plt.clf()
        plt.close('all')

    def get_variable(self, name: str) -> Any:
        """Safely access variables inside memory (e.g., for debugging or testing)."""
        return self.scope.get(name)
    
    def inject_variable(self, name: str, value: Any):
        """Inject a variable from outside (e.g., questionnaire_id)."""
        self.scope[name] = value