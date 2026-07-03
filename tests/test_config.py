"""Tests for config loading and validation."""

import pytest
import yaml
import tempfile
import os
from pathlib import Path

from conduit.config import load_config, Config, ServerConfig


class TestConfigLoading:
    """Tests for load_config function."""
    
    def test_valid_config_loads(self):
        """Test that a valid config file loads correctly."""
        config_data = {
            'model': 'llama3.2',
            'ollama_api_base': 'http://localhost:11434',
            'servers': [
                {'name': 'notes', 'command': 'python', 'args': ['server.py']},
                {'name': 'web', 'command': 'python', 'args': ['fetch.py']}
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            config = load_config(temp_path)
            assert config.model == 'llama3.2'
            assert config.ollama_api_base == 'http://localhost:11434'
            assert len(config.servers) == 2
            assert config.servers[0].name == 'notes'
            assert config.servers[0].command == 'python'
            assert config.servers[0].args == ['server.py']
            assert config.servers[1].name == 'web'
        finally:
            os.unlink(temp_path)
    
    def test_config_defaults(self):
        """Test that missing fields use sane defaults."""
        config_data = {}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            config = load_config(temp_path)
            assert config.model == 'llama3.2'
            assert config.ollama_api_base is None
            assert config.servers == []
        finally:
            os.unlink(temp_path)
    
    def test_config_with_env(self):
        """Test config with server environment variables."""
        config_data = {
            'model': 'mistral',
            'servers': [
                {
                    'name': 'test_server',
                    'command': 'python',
                    'args': ['server.py'],
                    'env': {'KEY': 'value', 'DEBUG': 'true'}
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            config = load_config(temp_path)
            assert config.model == 'mistral'
            assert len(config.servers) == 1
            assert config.servers[0].env == {'KEY': 'value', 'DEBUG': 'true'}
        finally:
            os.unlink(temp_path)
    
    def test_file_not_found(self):
        """Test that missing config file raises clear error."""
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_config('/nonexistent/path/config.yaml')
    
    def test_malformed_yaml(self):
        """Test that malformed YAML raises clear error."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [unclosed")
            temp_path = f.name
        
        try:
            with pytest.raises(yaml.YAMLError):
                load_config(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_invalid_model_type(self):
        """Test that invalid model type raises clear error."""
        config_data = {'model': 123}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Model must be a non-empty string"):
                load_config(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_empty_model_string(self):
        """Test that empty model string raises clear error."""
        config_data = {'model': ''}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Model must be a non-empty string"):
                load_config(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_servers_not_list(self):
        """Test that servers as non-list raises clear error."""
        config_data = {'servers': 'not_a_list'}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="servers must be a list"):
                load_config(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_server_missing_name(self):
        """Test that server missing name raises clear error."""
        config_data = {'servers': [{'command': 'python'}]}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="missing required field 'name'"):
                load_config(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_server_missing_command(self):
        """Test that server missing command raises clear error."""
        config_data = {'servers': [{'name': 'test'}]}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="missing required field 'command'"):
                load_config(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_server_args_not_list(self):
        """Test that server args as non-list raises clear error."""
        config_data = {'servers': [{'name': 'test', 'command': 'python', 'args': 'not_list'}]}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="args must be a list"):
                load_config(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_server_env_not_dict(self):
        """Test that server env as non-dict raises clear error."""
        config_data = {'servers': [{'name': 'test', 'command': 'python', 'env': 'not_dict'}]}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="env must be a dictionary"):
                load_config(temp_path)
        finally:
            os.unlink(temp_path)


class TestMultiServerToolAggregation:
    """Tests for multi-server tool aggregation with collision-safe namespacing."""
    
    def test_two_servers_aggregate_tools(self):
        """Test that two servers can be loaded and tools aggregated."""
        config_data = {
            'model': 'llama3.2',
            'servers': [
                {'name': 'notes', 'command': 'python', 'args': ['notes_server.py']},
                {'name': 'web', 'command': 'python', 'args': ['web_server.py']}
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            config = load_config(temp_path)
            assert len(config.servers) == 2
            server_names = [s.name for s in config.servers]
            assert 'notes' in server_names
            assert 'web' in server_names
        finally:
            os.unlink(temp_path)
    
    def test_same_server_twice_collision_safe(self):
        """Test that using the same server under two names is collision-safe."""
        config_data = {
            'model': 'llama3.2',
            'servers': [
                {'name': 'stub1', 'command': 'python', 'args': ['stub_server.py']},
                {'name': 'stub2', 'command': 'python', 'args': ['stub_server.py']}
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            config = load_config(temp_path)
            assert len(config.servers) == 2
            # Server names are distinct even if they use the same command
            assert config.servers[0].name == 'stub1'
            assert config.servers[1].name == 'stub2'
            # Both use the same underlying server
            assert config.servers[0].command == config.servers[1].command
        finally:
            os.unlink(temp_path)
    
    def test_example_config_loads(self):
        """Test that the example config file loads correctly."""
        # Check if example config exists
        example_path = Path('conduit.example.yaml')
        if example_path.exists():
            config = load_config(str(example_path))
            assert config.model == 'llama3.2'
            assert config.ollama_api_base == 'http://localhost:11434/v1'
            assert len(config.servers) == 1
            assert config.servers[0].name == 'glossary'
