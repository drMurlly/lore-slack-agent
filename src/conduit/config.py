"""Conduit configuration loading and management."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import yaml
from pathlib import Path


@dataclass
class ServerConfig:
    """Configuration for an MCP server."""
    name: str
    command: str
    args: List[str] = field(default_factory=list)
    env: Optional[Dict[str, str]] = None


@dataclass
class Config:
    """Conduit configuration."""
    model: str = "llama3.2"
    ollama_api_base: Optional[str] = None
    servers: List[ServerConfig] = field(default_factory=list)


def load_config(path: str) -> Config:
    """Load configuration from a YAML file.
    
    Args:
        path: Path to the YAML config file
        
    Returns:
        Config object with validated settings
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML is malformed
        ValueError: If required fields are missing or invalid
    """
    config_path = Path(path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(config_path, 'r') as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Invalid YAML in config file: {e}")
    
    if data is None:
        data = {}
    
    # Validate and extract model
    model = data.get('model', 'llama3.2')
    if not isinstance(model, str) or not model.strip():
        raise ValueError("Model must be a non-empty string")
    
    # Validate and extract ollama_api_base
    ollama_api_base = data.get('ollama_api_base')
    if ollama_api_base is not None and not isinstance(ollama_api_base, str):
        raise ValueError("ollama_api_base must be a string")
    
    # Validate and extract servers
    servers = []
    if 'servers' in data:
        if not isinstance(data['servers'], list):
            raise ValueError("servers must be a list")
        
        for i, server_data in enumerate(data['servers']):
            if not isinstance(server_data, dict):
                raise ValueError(f"Server {i} must be a dictionary")
            
            if 'name' not in server_data:
                raise ValueError(f"Server {i} missing required field 'name'")
            
            if 'command' not in server_data:
                raise ValueError(f"Server {i} missing required field 'command'")
            
            if not isinstance(server_data['name'], str) or not server_data['name'].strip():
                raise ValueError(f"Server {i} name must be a non-empty string")
            
            if not isinstance(server_data['command'], str) or not server_data['command'].strip():
                raise ValueError(f"Server {i} command must be a non-empty string")
            
            args = server_data.get('args', [])
            if not isinstance(args, list):
                raise ValueError(f"Server {i} args must be a list")
            
            env = server_data.get('env')
            if env is not None and not isinstance(env, dict):
                raise ValueError(f"Server {i} env must be a dictionary")
            
            servers.append(ServerConfig(
                name=server_data['name'],
                command=server_data['command'],
                args=args,
                env=env
            ))
    
    return Config(
        model=model,
        ollama_api_base=ollama_api_base,
        servers=servers
    )
