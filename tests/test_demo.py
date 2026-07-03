"""Tests for the standalone demo script."""
import subprocess
import sys
import json
import pathlib

ROOT = pathlib.Path(__file__).parents[1]


def test_demo_runs_and_writes_output(tmp_path, monkeypatch):
    """Test that run_demo.py runs successfully and writes output."""
    monkeypatch.chdir(tmp_path)
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_demo.py")],
        capture_output=True,
        text=True,
        timeout=30
    )
    assert r.returncode == 0, f"Demo failed:\nstdout: {r.stdout}\nstderr: {r.stderr}"
    
    # Check output file exists and has correct structure
    out_path = tmp_path / "demo_output.json"
    assert out_path.exists(), "demo_output.json was not created"
    
    out = json.loads(out_path.read_text())
    assert "answer" in out, "Output missing 'answer' field"
    assert "citations" in out, "Output missing 'citations' field"
    assert len(out["citations"]) >= 2, f"Expected at least 2 citations, got {len(out['citations'])}"
    
    # The money-shot: answer must contain both old and new values
    answer_lower = out["answer"].lower()
    assert "$20" in out["answer"] or "20" in answer_lower, "Answer should mention $20"
    assert "$10" in out["answer"] or "10" in answer_lower, "Answer should mention $10"


def test_readme_has_mermaid_and_lore():
    """Test that README.md has the architecture diagram and describes Lore."""
    readme = (ROOT / "README.md").read_text()
    
    # Check for mermaid diagram
    assert "mermaid" in readme.lower(), "README should contain mermaid diagram"
    assert "flowchart" in readme, "README should have flowchart"
    
    # Check for architecture section
    assert "architecture" in readme.lower(), "README should have Architecture section"
    
    # Check for contradiction/timeline resolver mention
    assert "contradiction" in readme.lower() or "timeline" in readme.lower(), \
        "README should mention contradiction or timeline resolution"
    
    # Check that it describes Lore (deep research) not just Conduit
    assert "lore" in readme.lower() or "deep research" in readme.lower(), \
        "README should describe Lore/deep research functionality"
