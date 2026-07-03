#!/usr/bin/env python3
"""Web Fetch MCP Server - provides URL fetching tools for Conduit agent."""

import json
import sys
import urllib.request
import urllib.error
from typing import Any, Optional
from html.parser import HTMLParser


class HTMLTextExtractor(HTMLParser):
    """Extract plain text from HTML."""
    
    def __init__(self):
        super().__init__()
        self.text_parts: list[str] = []
        self._in_script = False
        self._in_style = False
    
    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]):
        if tag.lower() == "script":
            self._in_script = True
        elif tag.lower() == "style":
            self._in_style = True
    
    def handle_endtag(self, tag: str):
        if tag.lower() == "script":
            self._in_script = False
        elif tag.lower() == "style":
            self._in_style = False
    
    def handle_data(self, data: str):
        if not self._in_script and not self._in_style:
            self.text_parts.append(data.strip())
    
    def get_text(self) -> str:
        return " ".join(self.text_parts)


def fetch_url(url: str, max_length: int = 2000) -> str:
    """Fetch a URL and return its text content.
    
    Args:
        url: The URL to fetch.
        max_length: Maximum length of returned text.
    
    Returns:
        JSON string with the fetched content or error.
    """
    try:
        # Validate URL
        if not url.startswith(("http://", "https://")):
            return json.dumps({"error": "URL must start with http:// or https://"})
        
        # Fetch the URL
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Conduit-MCP-Agent/1.0"}
        )
        
        with urllib.request.urlopen(req, timeout=10) as response:
            # Check content type
            content_type = response.headers.get("Content-Type", "")
            
            if "text/html" in content_type:
                # Parse HTML and extract text
                html_content = response.read().decode("utf-8", errors="ignore")
                parser = HTMLTextExtractor()
                parser.feed(html_content)
                text = parser.get_text()
            else:
                # Return raw text
                text = response.read().decode("utf-8", errors="ignore")
            
            # Truncate if necessary
            if len(text) > max_length:
                text = text[:max_length] + "... [truncated]"
            
            return json.dumps({
                "url": url,
                "content": text,
                "length": len(text),
                "status": "success"
            })
    
    except urllib.error.HTTPError as e:
        return json.dumps({
            "url": url,
            "error": f"HTTP {e.code}: {e.reason}",
            "status": "error"
        })
    except urllib.error.URLError as e:
        return json.dumps({
            "url": url,
            "error": f"URL error: {e.reason}",
            "status": "error"
        })
    except Exception as e:
        return json.dumps({
            "url": url,
            "error": str(e),
            "status": "error"
        })


def run_server():
    """Run the MCP server via stdio."""
    
    def handle_request(method: str, params: dict, request_id: Any) -> dict:
        """Handle an MCP request."""
        if method == "fetch_url":
            url = params.get("url", "")
            max_length = params.get("max_length", 2000)
            result = fetch_url(url, max_length)
            return {"result": result}
        else:
            return {"error": f"Unknown method: {method}"}
    
    # Read lines from stdin, process as JSON-RPC
    for line in sys.stdin:
        try:
            request = json.loads(line.strip())
            method = request.get("method", "")
            params = request.get("params", {})
            request_id = request.get("id")
            
            response = handle_request(method, params, request_id)
            response["id"] = request_id
            response["jsonrpc"] = "2.0"
            
            print(json.dumps(response), flush=True)
        except json.JSONDecodeError:
            error_response = {
                "id": None,
                "jsonrpc": "2.0",
                "error": {"code": -32700, "message": "Parse error"}
            }
            print(json.dumps(error_response), flush=True)
        except Exception as e:
            error_response = {
                "id": None,
                "jsonrpc": "2.0",
                "error": {"code": -32603, "message": str(e)}
            }
            print(json.dumps(error_response), flush=True)


if __name__ == "__main__":
    run_server()
