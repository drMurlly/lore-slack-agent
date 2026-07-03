#!/usr/bin/env python3
"""Notes MCP Server - provides note-taking tools for Conduit agent."""

import json
import sys
from typing import Any, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Note:
    """A single note entry."""
    id: int
    content: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class NotesStore:
    """In-memory store for notes."""
    
    def __init__(self):
        self._notes: list[Note] = []
        self._next_id = 1
    
    def add_note(self, content: str) -> dict[str, Any]:
        """Add a new note and return its details."""
        note = Note(id=self._next_id, content=content)
        self._notes.append(note)
        self._next_id += 1
        return {
            "id": note.id,
            "content": note.content,
            "created_at": note.created_at,
            "status": "created"
        }
    
    def list_notes(self) -> list[dict[str, Any]]:
        """List all notes."""
        return [
            {
                "id": note.id,
                "content": note.content,
                "created_at": note.created_at
            }
            for note in self._notes
        ]
    
    def get_note(self, note_id: int) -> Optional[dict[str, Any]]:
        """Get a specific note by ID."""
        for note in self._notes:
            if note.id == note_id:
                return {
                    "id": note.id,
                    "content": note.content,
                    "created_at": note.created_at
                }
        return None


# Global store instance
_store = NotesStore()


def add_note(content: str) -> str:
    """Add a new note.
    
    Args:
        content: The text content of the note.
    
    Returns:
        JSON string with the created note details.
    """
    result = _store.add_note(content)
    return json.dumps(result)


def list_notes() -> str:
    """List all notes.
    
    Returns:
        JSON string with list of all notes.
    """
    notes = _store.list_notes()
    return json.dumps({"notes": notes, "count": len(notes)})


def get_note(note_id: int) -> str:
    """Get a specific note by ID.
    
    Args:
        note_id: The ID of the note to retrieve.
    
    Returns:
        JSON string with the note details or error.
    """
    note = _store.get_note(note_id)
    if note:
        return json.dumps(note)
    return json.dumps({"error": f"Note with id {note_id} not found"})


def run_server():
    """Run the MCP server via stdio."""
    
    def handle_request(method: str, params: dict, request_id: Any) -> dict:
        """Handle an MCP request."""
        if method == "add_note":
            content = params.get("content", "")
            result = add_note(content)
            return {"result": result}
        elif method == "list_notes":
            result = list_notes()
            return {"result": result}
        elif method == "get_note":
            note_id = params.get("note_id", 0)
            result = get_note(note_id)
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
