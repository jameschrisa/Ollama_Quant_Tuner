"""
Ollama API Client for Ollama Manager
"""

import json
import logging
import time
from typing import Dict, List, Optional, Tuple, Union

import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ollama_api")


class OllamaAPIClient:
    """
    Client for interacting with the Ollama API
    """
    
    def __init__(self, base_url: str = "http://localhost:11434/api"):
        """
        Initialize the Ollama API client
        
        Args:
            base_url: Base URL for the Ollama API
        """
        self.base_url = base_url
        self.timeout = 10  # Default timeout in seconds
    
    def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Tuple[bool, Union[Dict, str]]:
        """
        Make an HTTP request to the Ollama API
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            data: Request data for POST requests
            
        Returns:
            Tuple of (success, response_data)
        """
        url = f"{self.base_url}/{endpoint}"
        
        try:
            if method == "GET":
                response = requests.get(url, timeout=self.timeout)
            elif method == "POST":
                response = requests.post(url, json=data, timeout=self.timeout)
            else:
                return False, f"Unsupported HTTP method: {method}"
            
            # Check if the request was successful
            if response.status_code == 200:
                try:
                    return True, response.json()
                except json.JSONDecodeError:
                    return True, response.text
            else:
                return False, f"API request failed with status code: {response.status_code}, response: {response.text}"
                
        except requests.RequestException as e:
            return False, f"Request failed: {str(e)}"
    
    def check_service(self) -> bool:
        """
        Check if the Ollama service is running
        
        Returns:
            True if the service is running, False otherwise
        """
        success, _ = self._make_request("GET", "tags")
        return success
    
    def list_models(self) -> Tuple[bool, List[Dict]]:
        """
        Get a list of available models
        
        Returns:
            Tuple of (success, model_list)
        """
        success, response = self._make_request("GET", "tags")
        
        if success:
            try:
                return True, response.get("models", [])
            except (AttributeError, KeyError):
                return False, []
        else:
            return False, []
    
    def pull_model(self, model_name: str, progress_callback=None) -> Tuple[bool, str]:
        """
        Pull a model from Ollama library
        
        Args:
            model_name: Name of the model to pull
            progress_callback: Optional callback function to report progress
            
        Returns:
            Tuple of (success, message)
        """
        endpoint = "pull"
        data = {"name": model_name}
        
        try:
            url = f"{self.base_url}/{endpoint}"
            with requests.post(url, json=data, stream=True) as response:
                if response.status_code != 200:
                    return False, f"Failed to pull model: {response.text}"
                
                total_downloaded = 0
                for line in response.iter_lines():
                    if line:
                        # Parse progress from response
                        try:
                            progress = json.loads(line)
                            if progress.get('status') and progress_callback:
                                progress_callback(progress)
                            if progress.get('completed', False):
                                return True, "Model pulled successfully"
                        except json.JSONDecodeError:
                            pass
                        
                return True, "Model pulled successfully"
        except requests.RequestException as e:
            return False, f"Failed to pull model: {str(e)}"
    
    def generate(self, model: str, prompt: str, options: Optional[Dict] = None) -> Tuple[bool, Dict]:
        """
        Generate a response from a model
        
        Args:
            model: Name of the model to use
            prompt: Text prompt to send to the model
            options: Optional parameters for generation
            
        Returns:
            Tuple of (success, response)
        """
        endpoint = "generate"
        data = {
            "model": model,
            "prompt": prompt
        }
        
        if options:
            data.update(options)
        
        success, response = self._make_request("POST", endpoint, data)
        return success, response
    
    def get_model_details(self, model_name: str) -> Tuple[bool, Dict]:
        """
        Get detailed information about a model
        
        Args:
            model_name: Name of the model
            
        Returns:
            Tuple of (success, model_details)
        """
        success, models = self.list_models()
        
        if success:
            for model in models:
                if model.get("name") == model_name:
                    return True, model
            
            return False, {"error": f"Model '{model_name}' not found"}
        else:
            return False, {"error": "Failed to list models"}
    
    def create_model(self, model_name: str, modelfile_content: str) -> Tuple[bool, str]:
        """
        Create a new model from a modelfile
        
        Args:
            model_name: Name for the new model
            modelfile_content: Content of the modelfile
            
        Returns:
            Tuple of (success, message)
        """
        endpoint = "create"
        data = {
            "name": model_name,
            "modelfile": modelfile_content
        }
        
        success, response = self._make_request("POST", endpoint, data)
        
        if success:
            return True, f"Model '{model_name}' created successfully"
        else:
            return False, f"Failed to create model: {response}"
    
    def delete_model(self, model_name: str) -> Tuple[bool, str]:
        """
        Delete a model
        
        Args:
            model_name: Name of the model to delete
            
        Returns:
            Tuple of (success, message)
        """
        endpoint = "delete"
        data = {
            "name": model_name
        }
        
        success, response = self._make_request("DELETE", endpoint, data)
        
        if success:
            return True, f"Model '{model_name}' deleted successfully"
        else:
            return False, f"Failed to delete model: {response}"
    
    def get_embedding(self, model: str, prompt: str) -> Tuple[bool, Dict]:
        """
        Get embeddings for a prompt
        
        Args:
            model: Name of the model to use
            prompt: Text to get embeddings for
            
        Returns:
            Tuple of (success, embeddings)
        """
        endpoint = "embeddings"
        data = {
            "model": model,
            "prompt": prompt
        }
        
        success, response = self._make_request("POST", endpoint, data)
        return success, response
    
    def search_models(self, query: str) -> List[Dict]:
        """
        Search for models on Ollama's website
        
        Args:
            query: Search query
            
        Returns:
            List of matching models
        """
        try:
            # This is a workaround as there's no official API for searching
            url = f"https://ollama.com/api/library?q={query}"
            response = requests.get(url, timeout=self.timeout)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to search models, status code: {response.status_code}")
                return []
        except requests.RequestException as e:
            logger.error(f"Failed to search models: {str(e)}")
            return []
