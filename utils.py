"""
Utility functions for Ollama Manager
"""

import json
import logging
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import psutil

# Configure logging
logger = logging.getLogger("ollama_utils")


@dataclass
class SystemResources:
    """Class to store system resource information."""
    cpu_count: int
    cpu_count_logical: int
    memory_total_gb: float
    platform_name: str
    processor: str
    gpu_info: str
    python_version: str
    has_nvidia_gpu: bool
    has_amd_gpu: bool
    has_apple_silicon: bool


def get_system_resources() -> SystemResources:
    """
    Get detailed information about system resources
    
    Returns:
        SystemResources object with system information
    """
    # Get basic system info
    cpu_count = psutil.cpu_count(logical=False) or 1
    cpu_count_logical = psutil.cpu_count(logical=True) or 1
    memory_total_gb = psutil.virtual_memory().total / (1024**3)
    platform_name = platform.platform()
    processor = platform.processor()
    python_version = platform.python_version()
    
    # Initialize GPU flags
    has_nvidia_gpu = False
    has_amd_gpu = False
    has_apple_silicon = False
    gpu_info = "No GPU detected"
    
    # Try to get GPU information
    try:
        if platform.system() == "Linux":
            # Try nvidia-smi for NVIDIA GPUs
            try:
                nvidia_output = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                    stderr=subprocess.DEVNULL
                ).decode("utf-8").strip()
                
                if nvidia_output:
                    gpu_info = f"NVIDIA: {nvidia_output}"
                    has_nvidia_gpu = True
            except (subprocess.SubprocessError, FileNotFoundError):
                # Try lspci for AMD GPUs
                try:
                    lspci_output = subprocess.check_output(
                        ["lspci", "-v"],
                        stderr=subprocess.DEVNULL
                    ).decode("utf-8").strip()
                    
                    if "AMD" in lspci_output and "VGA" in lspci_output:
                        gpu_info = "AMD GPU detected"
                        has_amd_gpu = True
                except (subprocess.SubprocessError, FileNotFoundError):
                    pass
                
        elif platform.system() == "Darwin":  # macOS
            # Check for Apple Silicon
            try:
                sysctl_output = subprocess.check_output(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    stderr=subprocess.DEVNULL
                ).decode("utf-8").strip()
                
                if "Apple" in sysctl_output:
                    gpu_info = "Apple Silicon (integrated GPU)"
                    has_apple_silicon = True
            except (subprocess.SubprocessError, FileNotFoundError):
                # Check system_profiler as fallback
                try:
                    sp_output = subprocess.check_output(
                        ["system_profiler", "SPDisplaysDataType"],
                        stderr=subprocess.DEVNULL
                    ).decode("utf-8").strip()
                    
                    if "Apple M" in sp_output:
                        gpu_info = "Apple Silicon (integrated GPU)"
                        has_apple_silicon = True
                except (subprocess.SubprocessError, FileNotFoundError):
                    pass
                
        elif platform.system() == "Windows":
            # Try to get GPU info from Windows
            try:
                wmic_output = subprocess.check_output(
                    ["wmic", "path", "win32_VideoController", "get", "name"],
                    stderr=subprocess.DEVNULL
                ).decode("utf-8").strip()
                
                if "NVIDIA" in wmic_output:
                    gpu_info = f"NVIDIA GPU: {wmic_output.splitlines()[1].strip()}"
                    has_nvidia_gpu = True
                elif "AMD" in wmic_output:
                    gpu_info = f"AMD GPU: {wmic_output.splitlines()[1].strip()}"
                    has_amd_gpu = True
            except (subprocess.SubprocessError, FileNotFoundError):
                pass
    except Exception as e:
        logger.error(f"Error getting GPU info: {e}")
    
    return SystemResources(
        cpu_count=cpu_count,
        cpu_count_logical=cpu_count_logical,
        memory_total_gb=memory_total_gb,
        platform_name=platform_name,
        processor=processor,
        gpu_info=gpu_info,
        python_version=python_version,
        has_nvidia_gpu=has_nvidia_gpu,
        has_amd_gpu=has_amd_gpu,
        has_apple_silicon=has_apple_silicon
    )


def recommend_ollama_parameters(system: SystemResources) -> Dict:
    """
    Generate recommended Ollama parameters based on system resources
    
    Args:
        system: SystemResources object with system information
        
    Returns:
        Dictionary of recommended parameters
    """
    recommendations = {}
    
    # Determine if system has GPU support
    has_gpu = system.has_nvidia_gpu or system.has_amd_gpu or system.has_apple_silicon
    
    # Model size recommendations based on memory
    if system.memory_total_gb < 8:
        recommendations["model_size"] = "Small models only (7B or less)"
        recommendations["recommended_models"] = [
            "phi:latest",
            "tinyllama:latest",
            "gemma:2b",
            "llama2:7b-q4_0"  # 4-bit quantized 7B model
        ]
    elif system.memory_total_gb < 16:
        recommendations["model_size"] = "Up to 13B models"
        recommendations["recommended_models"] = [
            "llama2:13b",
            "mistral:7b-instruct-v0.2",
            "gemma:7b-instruct",
            "orca-mini:13b"
        ]
    elif system.memory_total_gb < 32:
        recommendations["model_size"] = "Up to 33B models possible"
        recommendations["recommended_models"] = [
            "llama2:13b-chat",
            "llama3:8b-instruct",
            "wizard-vicuna:33b-q4_0",  # 4-bit quantized 33B model
            "mixtral:8x7b-instruct-v0.1"
        ]
    else:
        recommendations["model_size"] = "Large models (70B+) possible"
        recommendations["recommended_models"] = [
            "llama3:70b-instruct-q4_0",  # 4-bit quantized 70B model
            "mixtral:8x7b-instruct-v0.1",
            "llama2:70b-chat",
            "nous-hermes:70b-llama2-q4_K_M"  # Better 4-bit quantization
        ]
    
    # Quantization recommendations
    if system.memory_total_gb < 8:
        recommendations["quantization"] = "q4_0 or q4_k_m recommended"
        recommendations["quant_level"] = 1  # Q4_0
    elif system.memory_total_gb < 16:
        recommendations["quantization"] = "q5_k_m recommended for balance"
        recommendations["quant_level"] = 5  # Q5_K_M
    elif system.memory_total_gb < 32:
        recommendations["quantization"] = "q6_k recommended for quality"
        recommendations["quant_level"] = 6  # Q6_K
    else:
        recommendations["quantization"] = "q8_0 for highest quality if memory allows"
        recommendations["quant_level"] = 8  # Q8_0
    
    # Context size recommendations
    if system.memory_total_gb < 8:
        recommendations["context_size"] = 2048
    elif system.memory_total_gb < 16:
        recommendations["context_size"] = 4096
    elif system.memory_total_gb < 32:
        recommendations["context_size"] = 8192
    else:
        recommendations["context_size"] = 16384
    
    # KV cache and flash attention recommendations
    if has_gpu:
        recommendations["kv_cache"] = "f16"
        recommendations["flash_attention"] = True
    else:
        recommendations["kv_cache"] = "auto"
        recommendations["flash_attention"] = False
    
    # CPU thread recommendations
    if system.cpu_count_logical > 8:
        recommendations["num_cpu_threads"] = min(system.cpu_count_logical - 2, 16)
    else:
        recommendations["num_cpu_threads"] = max(system.cpu_count_logical - 1, 1)
    
    return recommendations


def get_quantization_options() -> Dict[str, Dict]:
    """
    Get information about different quantization options
    
    Returns:
        Dictionary of quantization options with details
    """
    return {
        "q4_0": {
            "k_level": 1,
            "description": "4-bit per weight, no K-means clustering",
            "benefits": "Smallest size, fastest loading",
            "drawbacks": "Lower quality compared to other options",
            "recommended_for": "Very limited memory systems",
            "size_reduction": "~75% reduction from F16"
        },
        "q4_k_m": {
            "k_level": 3,
            "description": "4-bit per weight with K-means clustering",
            "benefits": "Better quality than Q4_0 with similar size",
            "drawbacks": "Still has quality degradation",
            "recommended_for": "Low memory systems needing better quality",
            "size_reduction": "~75% reduction from F16"
        },
        "q5_k_m": {
            "k_level": 5,
            "description": "5-bit per weight with K-means clustering",
            "benefits": "Good balance of quality and size",
            "drawbacks": "Larger than 4-bit options",
            "recommended_for": "Balanced systems",
            "size_reduction": "~69% reduction from F16"
        },
        "q6_k": {
            "k_level": 6,
            "description": "6-bit per weight with K-means clustering",
            "benefits": "Good quality with moderate compression",
            "drawbacks": "Larger size than lower bit options",
            "recommended_for": "Systems with more memory prioritizing quality",
            "size_reduction": "~63% reduction from F16"
        },
        "q8_0": {
            "k_level": 8,
            "description": "8-bit per weight, no K-means clustering",
            "benefits": "Highest quality among quantized models",
            "drawbacks": "Largest file size among quantized options",
            "recommended_for": "Systems with ample memory prioritizing quality",
            "size_reduction": "~50% reduction from F16"
        }
    }


def check_ollama_installed() -> bool:
    """
    Check if Ollama is installed on the system
    
    Returns:
        True if Ollama is installed, False otherwise
    """
    try:
        subprocess.run(
            ["ollama", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False
        )
        return True
    except FileNotFoundError:
        return False


def get_ollama_version() -> Optional[str]:
    """
    Get the installed Ollama version
    
    Returns:
        Version string or None if not found
    """
    try:
        result = subprocess.run(
            ["ollama", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )
        
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except FileNotFoundError:
        return None


def check_param_compatibility(model_name: str, quantization_level: int, 
                              flash_attention: bool, kv_cache_type: str,
                              context_size: int, system: SystemResources) -> List[str]:
    """
    Check if the selected parameters are compatible with the system
    
    Args:
        model_name: Name of the model
        quantization_level: Quantization level (1-8)
        flash_attention: Whether flash attention is enabled
        kv_cache_type: KV cache type (auto, f16, f32)
        context_size: Context window size
        system: SystemResources object
        
    Returns:
        List of warnings (empty if all compatible)
    """
    warnings = []
    
    # Extract model size from name if possible
    model_size = None
    if "7b" in model_name.lower():
        model_size = 7
    elif "13b" in model_name.lower():
        model_size = 13
    elif "33b" in model_name.lower():
        model_size = 33
    elif "70b" in model_name.lower():
        model_size = 70
    
    # Check for sufficient memory
    if model_size is not None:
        estimated_memory = 0
        
        # Base memory requirements based on model size (very rough estimates)
        if quantization_level == 8:  # Q8_0
            estimated_memory = model_size * 0.5  # ~0.5GB per billion parameters
        elif quantization_level >= 6:  # Q6_K
            estimated_memory = model_size * 0.4  # ~0.4GB per billion parameters
        elif quantization_level >= 5:  # Q5_K_M
            estimated_memory = model_size * 0.35  # ~0.35GB per billion parameters
        else:  # Q4_0, Q4_K_M
            estimated_memory = model_size * 0.25  # ~0.25GB per billion parameters
        
        # Add context window impact
        context_modifier = context_size / 4096
        estimated_memory *= (1 + (context_modifier - 1) * 0.3)  # 30% more memory for 2x context
        
        # Check if we have enough memory
        if estimated_memory > system.memory_total_gb * 0.7:  # 70% of total memory
            warnings.append(f"⚠️ Estimated memory usage ({estimated_memory:.1f}GB) may exceed available system memory.")
            
            if quantization_level > 4:
                warnings.append("   Consider using a higher quantization level (Q4_K_M) to reduce memory usage.")
            
            if context_size > 4096:
                warnings.append("   Consider reducing context size to save memory.")
    
    # Check flash attention compatibility
    if flash_attention and not (system.has_nvidia_gpu or system.has_amd_gpu or system.has_apple_silicon):
        warnings.append("⚠️ Flash Attention is enabled but no compatible GPU was detected.")
        warnings.append("   This may not improve performance and could cause issues.")
    
    # KV cache checks
    if kv_cache_type == "f16" and not (system.has_nvidia_gpu or system.has_amd_gpu or system.has_apple_silicon):
        warnings.append("⚠️ F16 KV cache works best with GPU acceleration, but no compatible GPU was detected.")
    
    return warnings


def suggest_alternative_models(memory_gb: float) -> List[Dict]:
    """
    Suggest alternative models based on memory constraints
    
    Args:
        memory_gb: Available memory in GB
        
    Returns:
        List of model suggestions with details
    """
    suggestions = []
    
    if memory_gb < 4:
        suggestions.append({
            "name": "phi:2.7b-q4_0",
            "description": "Microsoft's 2.7B parameter model with 4-bit quantization",
            "memory_required": "~1.5GB",
            "use_case": "General text tasks with minimal memory usage"
        })
        suggestions.append({
            "name": "tinyllama:1.1b-chat-v1.0-q4_0",
            "description": "Extremely small 1.1B parameter model with 4-bit quantization",
            "memory_required": "~1GB",
            "use_case": "Basic chat and text completion with minimal memory"
        })
    elif memory_gb < 8:
        suggestions.append({
            "name": "llama2:7b-chat-q4_0",
            "description": "Meta's 7B chat model with 4-bit quantization",
            "memory_required": "~3.5GB",
            "use_case": "General chat and text completion"
        })
        suggestions.append({
            "name": "mistral:7b-instruct-v0.2-q4_0",
            "description": "Mistral 7B instruct model with 4-bit quantization",
            "memory_required": "~3.5GB",
            "use_case": "Instruction following and chat"
        })
    elif memory_gb < 16:
        suggestions.append({
            "name": "llama3:8b-instruct-q5_K_M",
            "description": "Meta's 8B model with 5-bit quantization",
            "memory_required": "~6GB",
            "use_case": "High-quality instruction following"
        })
        suggestions.append({
            "name": "orca-mini:13b-q5_K_M",
            "description": "Orca Mini 13B model with 5-bit quantization",
            "memory_required": "~8GB",
            "use_case": "Advanced reasoning and knowledge tasks"
        })
    else:
        suggestions.append({
            "name": "mixtral:8x7b-instruct-v0.1-q6_K",
            "description": "Mixtral 8x7B MoE model with 6-bit quantization",
            "memory_required": "~12GB",
            "use_case": "State-of-the-art instruction following with multiple experts"
        })
        suggestions.append({
            "name": "llama3:70b-instruct-q4_K_M",
            "description": "Meta's 70B model with 4-bit quantization",
            "memory_required": "~35GB",
            "use_case": "Highest quality reasoning and knowledge tasks"
        })
    
    return suggestions
