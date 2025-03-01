"""
Memory visualization utilities for Ollama Manager
"""

import math
import sys
from typing import Dict, List, Tuple

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.layout import Layout
from rich.text import Text

# Initialize console
console = Console()


def format_bytes(size_bytes: int) -> str:
    """
    Format bytes to human-readable string with appropriate unit
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted string with appropriate unit (KB, MB, GB)
    """
    if size_bytes == 0:
        return "0 B"
    
    size_name = ("B", "KB", "MB", "GB", "TB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    
    return f"{s} {size_name[i]}"


def create_memory_progress_bars(memory_data: Dict, title: str = "Memory Usage") -> Panel:
    """
    Create a panel with visual progress bars for memory usage
    
    Args:
        memory_data: Dictionary containing memory information
        title: Title for the panel
        
    Returns:
        Rich Panel containing memory progress bars
    """
    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
        TextColumn("[green]{task.fields[used]}"),
        TextColumn("of"),
        TextColumn("[cyan]{task.fields[total]}")
    )
    
    # System RAM usage
    ram_task = progress.add_task(
        "System RAM",
        total=memory_data["system_total"],
        completed=memory_data["system_used"],
        used=format_bytes(memory_data["system_used"]),
        total=format_bytes(memory_data["system_total"])
    )
    
    # Swap usage
    swap_task = progress.add_task(
        "Swap Memory",
        total=memory_data["swap_total"],
        completed=memory_data["swap_used"],
        used=format_bytes(memory_data["swap_used"]),
        total=format_bytes(memory_data["swap_total"])
    )
    
    # Return as a panel
    return Panel(progress, title=title, border_style="green")


def create_memory_diff_visualization(before: Dict, after: Dict) -> Panel:
    """
    Create a visual representation of memory usage differences
    
    Args:
        before: Memory information before model execution
        after: Memory information after model execution
        
    Returns:
        Rich Panel with visualization of memory differences
    """
    # Calculate differences
    system_diff = after["system_used"] - before["system_used"]
    process_diff = after["process_rss"] - before["process_rss"]
    
    # Create table for summary
    summary_table = Table(title="Memory Usage Changes")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Change", style="green")
    summary_table.add_column("Impact", style="yellow")
    
    # Add system RAM change
    system_diff_str = format_bytes(system_diff)
    if system_diff > 0:
        system_diff_str = f"+{system_diff_str}"
        system_impact = "Increased usage"
    else:
        system_impact = "Decreased usage"
    
    summary_table.add_row(
        "System RAM",
        system_diff_str,
        system_impact
    )
    
    # Add process memory change
    process_diff_str = format_bytes(process_diff)
    if process_diff > 0:
        process_diff_str = f"+{process_diff_str}"
        process_impact = "Model required additional memory"
    else:
        process_impact = "Memory was freed"
    
    summary_table.add_row(
        "Process Memory",
        process_diff_str,
        process_impact
    )
    
    # Create progress bars for before/after comparison
    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=60),
        TextColumn("{task.percentage:>3.1f}%"),
    )
    
    # Add before/after tasks
    progress.add_task(
        "Before Model Run",
        total=100,
        completed=before["system_percent"]
    )
    
    progress.add_task(
        "After Model Run",
        total=100,
        completed=after["system_percent"]
    )
    
    # Create layout
    layout = Layout()
    layout.split_column(
        Layout(summary_table, name="summary"),
        Layout(Text("\nSystem RAM Usage Comparison", style="bold magenta"), name="title"),
        Layout(progress, name="progress")
    )
    
    return Panel(layout, title="Memory Impact Visualization", border_style="yellow")


def generate_memory_recommendation(before: Dict, after: Dict, model_config: Dict) -> str:
    """
    Generate recommendations based on memory usage patterns
    
    Args:
        before: Memory information before model execution
        after: Memory information after model execution
        model_config: Current model configuration
        
    Returns:
        Recommendation text
    """
    # Calculate memory impact
    memory_used_gb = (after["system_used"] - before["system_used"]) / (1024**3)
    memory_percent_change = after["system_percent"] - before["system_percent"]
    
    # Initialize recommendations
    recommendations = []
    
    # Check if we're using too much memory
    if after["system_percent"] > 90:
        recommendations.append("⚠️ System memory usage is very high (>90%). Consider using a smaller model or higher quantization.")
    
    # Check memory impact
    if memory_used_gb > 4:
        recommendations.append(f"🔍 Model is using {memory_used_gb:.2f} GB of memory, which is substantial.")
        
        # Recommend higher quantization if using lower quantization
        if model_config.get("quantization_level") is None or model_config.get("quantization_level") > 4:
            recommendations.append("📝 Try using Q4_K_M quantization to reduce memory usage.")
    
    # Check if Flash Attention should be used
    if memory_percent_change > 15 and not model_config.get("flash_attention", False):
        recommendations.append("💡 Consider enabling Flash Attention if you have a compatible GPU to improve memory efficiency.")
    
    # Check if context size can be reduced
    if model_config.get("context_size", 0) > 8192 and memory_used_gb > 2:
        recommendations.append(f"📏 Your context size of {model_config['context_size']} may be unnecessarily large for your use case.")
        recommendations.append("   Consider reducing context size to 4096 or 8192 if you don't need to process very long documents.")
    
    # If few recommendations were made
    if len(recommendations) == 0:
        recommendations.append("✅ Memory usage looks good with the current configuration!")
        
        # If we're using very little memory
        if after["system_percent"] < 50 and memory_used_gb < 2:
            recommendations.append("🔍 You may be able to use a larger model or lower quantization for better quality.")
    
    return "\n".join(recommendations)
