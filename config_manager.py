"""
Configuration manager for Ollama Manager
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Set up logging
logger = logging.getLogger("config_manager")


class ConfigManager:
    """
    Manages configuration settings for Ollama Manager
    """
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize the configuration manager
        
        Args:
            config_dir: Optional custom configuration directory
        """
        # Determine config directory
        if config_dir:
            self.config_dir = Path(config_dir)
        else:
            # Default to user's home directory
            home_dir = Path.home()
            self.config_dir = home_dir / ".ollama_manager"
        
        # Ensure config directory exists
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Config file path
        self.config_file = self.config_dir / "config.json"
        
        # Initial configuration
        self.config = self._load_config()
    
    def _load_config(self) -> Dict:
        """
        Load configuration from file or create default
        
        Returns:
            Configuration dictionary
        """
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Error loading config: {e}")
                # Backup corrupted config
                if self.config_file.exists():
                    backup_path = self.config_file.with_suffix(".json.bak")
                    self.config_file.rename(backup_path)
                    logger.info(f"Backed up corrupted config to {backup_path}")
        
        # Default configuration
        return {
            "api": {
                "base_url": "http://localhost:11434/api",
                "timeout": 30
            },
            "ui": {
                "theme": "dark",
                "show_memory_visualization": True,
                "compact_mode": False,
                "log_level": "INFO"
            },
            "defaults": {
                "quantization_level": 4,  # Q4_K_M
                "flash_attention": True,
                "kv_cache_type": "auto",
                "context_size": 4096,
                "max_tokens": 2048
            },
            "recent_models": [],
            "user_presets": {}
        }
    
    def save_config(self) -> bool:
        """
        Save configuration to file
        
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            return True
        except IOError as e:
            logger.error(f"Error saving config: {e}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value
        
        Args:
            key: Configuration key (can use dot notation like 'api.timeout')
            default: Default value if key doesn't exist
            
        Returns:
            Configuration value or default
        """
        parts = key.split('.')
        
        # Navigate through nested dictionary
        value = self.config
        try:
            for part in parts:
                value = value[part]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> bool:
        """
        Set a configuration value
        
        Args:
            key: Configuration key (can use dot notation like 'api.timeout')
            value: Value to set
            
        Returns:
            True if set successfully, False otherwise
        """
        parts = key.split('.')
        
        # Navigate to the correct level
        config = self.config
        for i, part in enumerate(parts[:-1]):
            if part not in config or not isinstance(config[part], dict):
                config[part] = {}
            config = config[part]
        
        # Set the value
        config[parts[-1]] = value
        
        return self.save_config()
    
    def add_recent_model(self, model_name: str) -> None:
        """
        Add a model to the recent models list
        
        Args:
            model_name: Name of the model
        """
        recent = self.config.get("recent_models", [])
        
        # Remove if already exists (to move to front)
        if model_name in recent:
            recent.remove(model_name)
        
        # Add to front of list
        recent.insert(0, model_name)
        
        # Keep only last 10
        self.config["recent_models"] = recent[:10]
        self.save_config()
    
    def get_recent_models(self) -> List[str]:
        """
        Get list of recently used models
        
        Returns:
            List of model names
        """
        return self.config.get("recent_models", [])
    
    def save_model_preset(self, name: str, model_config: Dict) -> bool:
        """
        Save a model configuration preset
        
        Args:
            name: Preset name
            model_config: Model configuration dictionary
            
        Returns:
            True if saved successfully, False otherwise
        """
        presets = self.config.get("user_presets", {})
        presets[name] = model_config
        self.config["user_presets"] = presets
        return self.save_config()
    
    def get_model_preset(self, name: str) -> Optional[Dict]:
        """
        Get a model configuration preset
        
        Args:
            name: Preset name
            
        Returns:
            Preset configuration or None if not found
        """
        presets = self.config.get("user_presets", {})
        return presets.get(name)
    
    def list_presets(self) -> List[str]:
        """
        Get list of available presets
        
        Returns:
            List of preset names
        """
        return list(self.config.get("user_presets", {}).keys())
    
    def delete_preset(self, name: str) -> bool:
        """
        Delete a model preset
        
        Args:
            name: Preset name
            
        Returns:
            True if deleted successfully, False otherwise
        """
        presets = self.config.get("user_presets", {})
        if name in presets:
            del presets[name]
            self.config["user_presets"] = presets
            return self.save_config()
        return False
    
    def reset_to_defaults(self) -> bool:
        """
        Reset configuration to defaults
        
        Returns:
            True if reset successfully, False otherwise
        """
        self.config = self._load_config()
        return self.save_config()
