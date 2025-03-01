#!/usr/bin/env python3
"""
Ollama Manager - Main entry point
"""

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from config_manager import ConfigManager
from logger import get_default_log_file, setup_logger
from ollama_manager import OllamaManager, interactive_menu
from utils import check_ollama_installed, get_ollama_version


def main():
    """Main entry point for the application."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Ollama Manager - A beautiful terminal interface for managing Ollama models and parameters"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set logging level"
    )
    parser.add_argument(
        "--log-file",
        help="Path to log file (default: ~/.local/share/ollama_manager/logs/ollama_manager.log)"
    )
    parser.add_argument(
        "--config-dir",
        help="Path to configuration directory"
    )
    parser.add_argument(
        "--model",
        help="Run with specific model (skips model selection)"
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version information and exit"
    )
    
    args = parser.parse_args()
    
    # Initialize console
    console = Console()
    
    # Show version if requested
    if args.version:
        console.print("[bold]Ollama Manager[/bold] version 1.0.0")
        ollama_version = get_ollama_version()
        if ollama_version:
            console.print(f"Ollama version: {ollama_version}")
        else:
            console.print("[yellow]Ollama not found. Please install Ollama first.[/yellow]")
        return 0
    
    # Set up logging
    log_file = args.log_file or get_default_log_file()
    logger = setup_logger(args.log_level, log_file)
    
    # Load configuration
    config_manager = ConfigManager(args.config_dir)
    
    # Check if Ollama is installed
    if not check_ollama_installed():
        console.print(Panel(
            "[bold red]Ollama is not installed or not in PATH[/bold red]\n\n"
            "Please install Ollama from [link]https://ollama.com/download[/link] "
            "and make sure it's available in your PATH before running Ollama Manager.",
            title="Ollama Not Found",
            border_style="red"
        ))
        return 1
    
    # Display welcome message
    console.print(Panel(
        "[bold cyan]Welcome to Ollama Manager![/bold cyan]\n\n"
        "A beautiful terminal interface for managing Ollama models and parameters.",
        title="Ollama Manager v1.0.0",
        border_style="cyan"
    ))
    
    # Initialize manager
    manager = OllamaManager(config_manager)
    
    # Run interactive menu
    try:
        interactive_menu(manager)
        return 0
    except KeyboardInterrupt:
        console.print("\n[bold green]Exiting Ollama Manager. Goodbye![/bold green]")
        return 0
    except Exception as e:
        logger.exception("Unexpected error")
        console.print(f"[bold red]An error occurred: {str(e)}[/bold red]")
        return 1


if __name__ == "__main__":
    sys.exit(main())
