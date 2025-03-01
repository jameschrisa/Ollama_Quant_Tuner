"""
Logging configuration for Ollama Manager
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler


def setup_logger(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    app_name: str = "ollama_manager"
) -> logging.Logger:
    """
    Set up application logger
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        app_name: Application name for logger
        
    Returns:
        Configured logger
    """
    # Convert string log level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create logger
    logger = logging.getLogger(app_name)
    logger.setLevel(numeric_level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Console handler with Rich formatting
    console_handler = RichHandler(
        rich_tracebacks=True,
        markup=True,
        show_time=True,
        show_path=False
    )
    console_handler.setLevel(numeric_level)
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler if log file specified
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_default_log_file() -> str:
    """
    Get default log file path based on platform
    
    Returns:
        Path to default log file
    """
    home_dir = Path.home()
    
    if sys.platform.startswith("win"):
        log_dir = home_dir / "AppData" / "Local" / "OllamaManager" / "logs"
    elif sys.platform.startswith("darwin"):  # macOS
        log_dir = home_dir / "Library" / "Logs" / "OllamaManager"
    else:  # Linux and others
        log_dir = home_dir / ".local" / "share" / "ollama_manager" / "logs"
    
    log_dir.mkdir(parents=True, exist_ok=True)
    return str(log_dir / "ollama_manager.log")
