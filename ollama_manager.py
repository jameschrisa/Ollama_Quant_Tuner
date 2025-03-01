#!/usr/bin/env python3
"""
Ollama Manager - A beautiful command-line UI for managing Ollama models and parameters
"""

import argparse
import json
import logging
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import psutil
import requests
import rich
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table
from rich.text import Text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
log = logging.getLogger("ollama_manager")

# Global console object
console = Console()

# Base Ollama API URL
OLLAMA_API_BASE = "http://localhost:11434/api"

@dataclass
class ModelConfig:
    """Class to store Ollama model configuration."""
    model_name: str
    quantization_level: Optional[int] = None
    flash_attention: bool = False
    kv_cache_type: str = "auto"
    context_size: Optional[int] = None


class MemorySnapshot:
    """Class to capture and display memory usage."""
    
    def __init__(self):
        self.timestamp = time.time()
        self.memory_info = self._get_memory_info()
        
    def _get_memory_info(self) -> Dict:
        """Get current memory information."""
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        process = psutil.Process(os.getpid())
        
        return {
            "system_total": memory.total,
            "system_used": memory.used,
            "system_free": memory.free,
            "system_percent": memory.percent,
            "swap_total": swap.total,
            "swap_used": swap.used,
            "swap_free": swap.free,
            "swap_percent": swap.percent,
            "process_rss": process.memory_info().rss,
            "process_vms": process.memory_info().vms,
            "process_percent": process.memory_percent()
        }
    
    def display(self, title: str = "Memory Usage") -> Panel:
        """Create a rich panel with memory information."""
        mem = self.memory_info
        
        memory_table = Table(title=title)
        memory_table.add_column("Type")
        memory_table.add_column("Total")
        memory_table.add_column("Used")
        memory_table.add_column("Free")
        memory_table.add_column("Percent")
        
        memory_table.add_row(
            "System RAM",
            f"{mem['system_total'] / (1024**3):.2f} GB",
            f"{mem['system_used'] / (1024**3):.2f} GB",
            f"{mem['system_free'] / (1024**3):.2f} GB",
            f"{mem['system_percent']}%"
        )
        
        memory_table.add_row(
            "Swap",
            f"{mem['swap_total'] / (1024**3):.2f} GB",
            f"{mem['swap_used'] / (1024**3):.2f} GB",
            f"{mem['swap_free'] / (1024**3):.2f} GB",
            f"{mem['swap_percent']}%"
        )
        
        memory_table.add_row(
            "Process",
            "N/A",
            f"{mem['process_rss'] / (1024**2):.2f} MB",
            "N/A",
            f"{mem['process_percent']:.2f}%"
        )
        
        return Panel(memory_table, title=title, border_style="green")


class OllamaManager:
    """Main class to manage Ollama models and parameters."""
    
    def __init__(self):
        self.before_snapshot = None
        self.after_snapshot = None
        self.model_config = None
        self.available_models = []
        
    def check_ollama_running(self) -> bool:
        """Check if Ollama service is running."""
        try:
            response = requests.get(f"{OLLAMA_API_BASE}/tags", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    def get_available_models(self) -> List[str]:
        """Get list of available models from local Ollama installation."""
        try:
            response = requests.get(f"{OLLAMA_API_BASE}/tags", timeout=10)
            if response.status_code == 200:
                models_data = response.json()
                self.available_models = [model["name"] for model in models_data.get("models", [])]
                return self.available_models
            return []
        except requests.RequestException as e:
            log.error(f"Failed to get available models: {e}")
            return []
    
    def search_ollama_models(self, query: str) -> List[Dict]:
        """Search for models on Ollama's website."""
        try:
            console.print(f"Searching for models matching '[bold]{query}[/bold]'...")
            # This is a workaround since there's no official API for searching
            url = f"https://ollama.com/api/library?q={query}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                models = response.json()
                if models:
                    return models
            return []
        except requests.RequestException as e:
            log.error(f"Failed to search models: {e}")
            return []
    
    def display_model_search_results(self, models: List[Dict]):
        """Display search results in a table."""
        if not models:
            console.print("[yellow]No models found matching your search.[/yellow]")
            return
        
        table = Table(title="Ollama Models Search Results")
        table.add_column("Name", style="cyan")
        table.add_column("Description")
        table.add_column("Size", style="green")
        table.add_column("Quantization")
        
        for model in models:
            name = model.get("name", "Unknown")
            description = model.get("description", "No description")
            # Truncate long descriptions
            if len(description) > 50:
                description = description[:47] + "..."
            size = model.get("size", "Unknown")
            # Format size if available
            if isinstance(size, int):
                size = f"{size / (1024**3):.2f} GB"
            quant = model.get("details", {}).get("quantization", "Unknown")
            
            table.add_row(name, description, size, quant)
        
        console.print(table)
    
    def pull_model(self, model_name: str) -> bool:
        """Pull a model from Ollama's library."""
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold green]Pulling model {model_name}..."),
            console=console
        ) as progress:
            task = progress.add_task("Pulling", total=None)
            
            try:
                # Use subprocess to show output in real-time
                process = subprocess.Popen(
                    ["ollama", "pull", model_name],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                while True:
                    output = process.stdout.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        progress.update(task, description=f"[bold green]Pulling {model_name}: {output.strip()}")
                
                rc = process.poll()
                return rc == 0
            except Exception as e:
                log.error(f"Failed to pull model: {e}")
                return False
    
    def set_model_parameters(self, model_config: ModelConfig) -> bool:
        """Set Ollama model parameters."""
        self.model_config = model_config
        
        # Create modelfile content with parameters
        modelfile_content = f"FROM {model_config.model_name}\n"
        
        if model_config.quantization_level is not None:
            modelfile_content += f"PARAMETER quant_k {model_config.quantization_level}\n"
        
        if model_config.flash_attention:
            modelfile_content += "PARAMETER flash_attn true\n"
        
        if model_config.kv_cache_type != "auto":
            modelfile_content += f"PARAMETER cache_type {model_config.kv_cache_type}\n"
        
        if model_config.context_size is not None:
            modelfile_content += f"PARAMETER context_size {model_config.context_size}\n"
        
        # Create a temporary file for the modelfile
        with open("modelfile.tmp", "w") as f:
            f.write(modelfile_content)
        
        # Create the model with the custom parameters
        custom_name = f"{model_config.model_name}-custom"
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn(f"[bold green]Creating custom model {custom_name}..."),
                console=console
            ) as progress:
                task = progress.add_task("Creating", total=None)
                
                process = subprocess.Popen(
                    ["ollama", "create", custom_name, "-f", "modelfile.tmp"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                while True:
                    output = process.stdout.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        progress.update(task, description=f"[bold green]Creating {custom_name}: {output.strip()}")
                
                rc = process.poll()
                
                # Clean up
                os.remove("modelfile.tmp")
                
                if rc == 0:
                    console.print(f"[bold green]Successfully created model with custom parameters: {custom_name}[/bold green]")
                    return True
                else:
                    error = process.stderr.read()
                    console.print(f"[bold red]Failed to create custom model: {error}[/bold red]")
                    return False
                
        except Exception as e:
            log.error(f"Failed to set model parameters: {e}")
            os.remove("modelfile.tmp")
            return False
    
    def run_model(self, prompt: str) -> bool:
        """Run the model with a prompt and capture memory usage before and after."""
        if not self.model_config:
            console.print("[bold red]No model configuration set. Please configure a model first.[/bold red]")
            return False
        
        custom_name = f"{self.model_config.model_name}-custom"
        
        # Take before snapshot
        self.before_snapshot = MemorySnapshot()
        console.print(self.before_snapshot.display(title="Memory Before Running Model"))
        
        try:
            console.print(f"\n[bold]Running model with prompt: [italic]\"{prompt}\"[/italic][/bold]\n")
            
            # Use subprocess to show output in real-time
            process = subprocess.Popen(
                ["ollama", "run", custom_name, prompt],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Display model output
            console.print(Panel("", title="Model Output", border_style="cyan"))
            
            output = ""
            while True:
                line = process.stdout.readline()
                if line == '' and process.poll() is not None:
                    break
                if line:
                    output += line
                    console.print(line.strip())
            
            # Take after snapshot
            self.after_snapshot = MemorySnapshot()
            console.print(self.after_snapshot.display(title="Memory After Running Model"))
            
            # Display memory diff
            self.display_memory_comparison()
            
            return True
        except Exception as e:
            log.error(f"Failed to run model: {e}")
            return False
    
    def display_memory_comparison(self):
        """Display memory usage comparison before and after running the model."""
        if not self.before_snapshot or not self.after_snapshot:
            console.print("[yellow]No memory snapshots available for comparison.[/yellow]")
            return
        
        before = self.before_snapshot.memory_info
        after = self.after_snapshot.memory_info
        
        diff_table = Table(title="Memory Usage Comparison")
        diff_table.add_column("Metric")
        diff_table.add_column("Before")
        diff_table.add_column("After")
        diff_table.add_column("Difference")
        
        # Calculate differences
        metrics = [
            ("System Used", before["system_used"], after["system_used"], "GB"),
            ("System Free", before["system_free"], after["system_free"], "GB"),
            ("System Usage %", before["system_percent"], after["system_percent"], "%"),
            ("Process RSS", before["process_rss"], after["process_rss"], "MB"),
            ("Process %", before["process_percent"], after["process_percent"], "%")
        ]
        
        for name, before_val, after_val, unit:
            if name.endswith("%"):
                # Percentage values
                diff = after_val - before_val
                diff_text = f"{diff:+.2f}{unit}"
                
                diff_table.add_row(
                    name,
                    f"{before_val:.2f}{unit}",
                    f"{after_val:.2f}{unit}",
                    diff_text
                )
            elif unit == "GB":
                # GB values
                before_gb = before_val / (1024**3)
                after_gb = after_val / (1024**3)
                diff_gb = after_gb - before_gb
                diff_text = f"{diff_gb:+.2f} {unit}"
                
                diff_table.add_row(
                    name,
                    f"{before_gb:.2f} {unit}",
                    f"{after_gb:.2f} {unit}",
                    diff_text
                )
            elif unit == "MB":
                # MB values
                before_mb = before_val / (1024**2)
                after_mb = after_val / (1024**2)
                diff_mb = after_mb - before_mb
                diff_text = f"{diff_mb:+.2f} {unit}"
                
                diff_table.add_row(
                    name,
                    f"{before_mb:.2f} {unit}",
                    f"{after_mb:.2f} {unit}",
                    diff_text
                )
        
        console.print(Panel(diff_table, title="Memory Impact Analysis", border_style="yellow"))


def get_system_info() -> Dict:
    """Get information about the system."""
    system_info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "memory_total": psutil.virtual_memory().total / (1024**3),  # GB
        "cpu_count": psutil.cpu_count(logical=False),
        "cpu_count_logical": psutil.cpu_count(logical=True)
    }
    
    # Try to get GPU information if available
    try:
        if platform.system() == "Linux":
            # Try nvidia-smi for NVIDIA GPUs
            gpu_info = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"]).decode("utf-8").strip()
            system_info["gpu"] = gpu_info
        elif platform.system() == "Darwin":  # macOS
            # For Apple Silicon, get Metal device info
            metal_info = subprocess.check_output(["system_profiler", "SPDisplaysDataType"]).decode("utf-8")
            if "Apple M" in metal_info:
                system_info["gpu"] = "Apple Silicon (integrated)"
    except (subprocess.SubprocessError, FileNotFoundError):
        system_info["gpu"] = "No GPU information available"
    
    return system_info


def display_system_info():
    """Display system information."""
    info = get_system_info()
    
    table = Table(title="System Information")
    table.add_column("Component", style="cyan")
    table.add_column("Details", style="green")
    
    table.add_row("Platform", info["platform"])
    table.add_row("Processor", info["processor"])
    table.add_row("Physical CPU Cores", str(info["cpu_count"]))
    table.add_row("Logical CPU Cores", str(info["cpu_count_logical"]))
    table.add_row("Total Memory", f"{info['memory_total']:.2f} GB")
    table.add_row("GPU", info["gpu"])
    table.add_row("Python Version", info["python_version"])
    
    console.print(Panel(table, border_style="blue"))


def recommend_parameters(system_info: Dict) -> Dict:
    """Recommend Ollama parameters based on system specifications."""
    recommendations = {}
    
    # Determine if the system has a discrete GPU
    has_gpu = "nvidia" in system_info.get("gpu", "").lower() or "amd" in system_info.get("gpu", "").lower()
    
    # Memory-based recommendations
    memory_gb = system_info["memory_total"]
    
    if memory_gb < 8:
        recommendations["model_size"] = "Small models only (7B or less)"
        recommendations["quantization"] = "q4_0 or q4_k_m recommended"
        recommendations["context_size"] = "2048 or lower"
        recommendations["flash_attention"] = False
    elif memory_gb < 16:
        recommendations["model_size"] = "Up to 13B models"
        recommendations["quantization"] = "q5_k_m recommended for balance"
        recommendations["context_size"] = "4096 or lower"
        recommendations["flash_attention"] = has_gpu
    elif memory_gb < 32:
        recommendations["model_size"] = "Up to 33B models possible"
        recommendations["quantization"] = "q6_k recommended for quality"
        recommendations["context_size"] = "8192 possible"
        recommendations["flash_attention"] = has_gpu
    else:
        recommendations["model_size"] = "Large models (70B+) possible"
        recommendations["quantization"] = "q8_0 for highest quality if memory allows"
        recommendations["context_size"] = "16384 or higher possible"
        recommendations["flash_attention"] = has_gpu
    
    # KV cache recommendations
    if has_gpu:
        recommendations["kv_cache"] = "f16 or auto"
    else:
        recommendations["kv_cache"] = "auto"
    
    return recommendations


def display_recommendations(recommendations: Dict):
    """Display parameter recommendations."""
    table = Table(title="Recommended Ollama Parameters")
    table.add_column("Parameter", style="cyan")
    table.add_column("Recommendation", style="green")
    
    for param, value in recommendations.items():
        table.add_row(param.replace("_", " ").title(), str(value))
    
    console.print(Panel(table, border_style="magenta"))


def interactive_menu(manager: OllamaManager):
    """Interactive menu for the application."""
    while True:
        console.clear()
        console.print(Panel.fit("[bold cyan]Ollama Manager[/bold cyan] - Interactive Mode", border_style="cyan"))
        
        # Main menu options
        options = [
            "Check System Information",
            "View Local Models",
            "Search Ollama Library",
            "Pull a Model",
            "Configure and Run a Model",
            "Exit"
        ]
        
        # Check if Ollama is running
        if not manager.check_ollama_running():
            console.print("[bold red]Warning: Ollama service is not running![/bold red]")
            console.print("Please start Ollama service before proceeding.\n")
        
        # Display menu options
        for i, option in enumerate(options):
            console.print(f"[bold]{i+1}.[/bold] {option}")
        
        choice = Prompt.ask("\nEnter your choice", choices=[str(i+1) for i in range(len(options))])
        
        if choice == "1":  # System Information
            console.clear()
            console.print(Panel("[bold]System Information[/bold]", border_style="blue"))
            
            system_info = get_system_info()
            display_system_info()
            
            recommendations = recommend_parameters(system_info)
            display_recommendations(recommendations)
            
            console.print("\nThis information can help you choose appropriate models and parameters.")
            Prompt.ask("Press Enter to continue")
            
        elif choice == "2":  # View Local Models
            console.clear()
            console.print(Panel("[bold]Local Ollama Models[/bold]", border_style="blue"))
            
            if not manager.check_ollama_running():
                console.print("[bold red]Ollama service is not running. Cannot fetch local models.[/bold red]")
                Prompt.ask("Press Enter to continue")
                continue
            
            models = manager.get_available_models()
            
            if models:
                table = Table(title="Available Local Models")
                table.add_column("Model Name", style="cyan")
                
                for model in models:
                    table.add_row(model)
                
                console.print(table)
            else:
                console.print("[yellow]No local models found. Use option 4 to pull models.[/yellow]")
            
            Prompt.ask("Press Enter to continue")
            
        elif choice == "3":  # Search Ollama Library
            console.clear()
            console.print(Panel("[bold]Search Ollama Model Library[/bold]", border_style="blue"))
            
            query = Prompt.ask("Enter search term (e.g., llama, mistral, tiny)")
            
            results = manager.search_ollama_models(query)
            manager.display_model_search_results(results)
            
            Prompt.ask("Press Enter to continue")
            
        elif choice == "4":  # Pull a Model
            console.clear()
            console.print(Panel("[bold]Pull a Model from Ollama Library[/bold]", border_style="blue"))
            
            if not manager.check_ollama_running():
                console.print("[bold red]Ollama service is not running. Cannot pull models.[/bold red]")
                Prompt.ask("Press Enter to continue")
                continue
            
            model_name = Prompt.ask("Enter model name to pull (e.g., llama2:7b, mistral:7b-instruct)")
            
            if Confirm.ask(f"Pull model '{model_name}'?", default=True):
                success = manager.pull_model(model_name)
                
                if success:
                    console.print(f"[bold green]Successfully pulled model: {model_name}[/bold green]")
                else:
                    console.print(f"[bold red]Failed to pull model: {model_name}[/bold red]")
            
            Prompt.ask("Press Enter to continue")
            
        elif choice == "5":  # Configure and Run a Model
            console.clear()
            console.print(Panel("[bold]Configure and Run Model[/bold]", border_style="blue"))
            
            if not manager.check_ollama_running():
                console.print("[bold red]Ollama service is not running. Cannot configure models.[/bold red]")
                Prompt.ask("Press Enter to continue")
                continue
            
            # Get available models
            models = manager.get_available_models()
            
            if not models:
                console.print("[yellow]No local models found. Use option 4 to pull models first.[/yellow]")
                Prompt.ask("Press Enter to continue")
                continue
            
            # Display available models
            table = Table(title="Available Models")
            table.add_column("Index", style="cyan")
            table.add_column("Model Name", style="green")
            
            for i, model in enumerate(models):
                table.add_row(str(i+1), model)
            
            console.print(table)
            
            # Select model
            model_idx = IntPrompt.ask(
                "Select model index",
                min_value=1,
                max_value=len(models)
            )
            model_name = models[model_idx-1]
            
            # Configure parameters
            console.print(f"\n[bold]Configuring parameters for: [cyan]{model_name}[/cyan][/bold]")
            
            # Get system info for recommendations
            system_info = get_system_info()
            recommendations = recommend_parameters(system_info)
            display_recommendations(recommendations)
            
            # Quantization levels with explanations
            quant_options = {
                "1": "Default (no change)",
                "2": "Q4_0 (4-bit, smallest size, lowest quality)",
                "3": "Q4_K_M (4-bit, small size, better quality)",
                "4": "Q5_K_M (5-bit, balanced size and quality)",
                "5": "Q6_K (6-bit, good quality, larger size)",
                "6": "Q8_0 (8-bit, best quality, largest size)"
            }
            
            # Display quantization options
            console.print("\n[bold]Quantization Options:[/bold]")
            for key, desc in quant_options.items():
                console.print(f"{key}. {desc}")
            
            quant_choice = Prompt.ask("Select quantization", choices=list(quant_options.keys()))
            
            # Map choices to actual K levels
            quant_mapping = {
                "1": None,  # Default
                "2": 1,     # Q4_0
                "3": 3,     # Q4_K_M
                "4": 5,     # Q5_K_M
                "5": 6,     # Q6_K
                "6": 8      # Q8_0
            }
            
            quantization_level = quant_mapping[quant_choice]
            
            # Flash attention
            flash_attention = Confirm.ask(
                "Enable Flash Attention? (Recommended for GPUs)",
                default=recommendations["flash_attention"]
            )
            
            # KV cache type
            kv_cache_options = ["auto", "f16", "f32"]
            console.print("\n[bold]KV Cache Options:[/bold]")
            console.print("1. auto (Let Ollama decide)")
            console.print("2. f16 (Half precision, faster but less accurate)")
            console.print("3. f32 (Full precision, more accurate but uses more memory)")
            
            kv_choice = Prompt.ask("Select KV cache type", choices=["1", "2", "3"])
            kv_cache_type = kv_cache_options[int(kv_choice) - 1]
            
            # Context size
            use_custom_context = Confirm.ask("Set custom context size?", default=False)
            context_size = None
            if use_custom_context:
                context_size = IntPrompt.ask(
                    "Enter context size (power of 2 recommended, e.g., 2048, 4096, 8192)",
                    default=4096
                )
            
            # Create model config
            model_config = ModelConfig(
                model_name=model_name,
                quantization_level=quantization_level,
                flash_attention=flash_attention,
                kv_cache_type=kv_cache_type,
                context_size=context_size
            )
            
            # Set parameters
            console.print("\n[bold]Setting model parameters...[/bold]")
            success = manager.set_model_parameters(model_config)
            
            if not success:
                console.print("[bold red]Failed to set model parameters. Returning to menu.[/bold red]")
                Prompt.ask("Press Enter to continue")
                continue
            
            # Run the model
            prompt = Prompt.ask("\nEnter a prompt to test the model")
            manager.run_model(prompt)
            
            Prompt.ask("Press Enter to continue")
            
        elif choice == "6":  # Exit
            console.print("[bold green]Thank you for using Ollama Manager![/bold green]")
            return


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description="Ollama Manager - Configure and test Ollama models")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    args = parser.parse_args()
    
    console.print(Panel.fit("[bold cyan]Ollama Manager[/bold cyan]", border_style="cyan"))
    
    manager = OllamaManager()
    
    # Always run in interactive mode for now
    interactive_menu(manager)


if __name__ == "__main__":
    main()
