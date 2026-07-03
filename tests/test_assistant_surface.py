"""Tests for the assistant surface streaming trace functionality."""
import unittest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

from conduit.assistant_surface import SlackClient, AssistantContext, ResearchAssistant
from conduit.citations import Answer, Citation


class TestResearchAssistant(unittest.TestCase):
    """Tests for the ResearchAssistant class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_client = MagicMock()
        self.context = AssistantContext(
            channel="C123456",
            thread_ts="1234567890.123456"
        )
        self.assistant = ResearchAssistant(
            client=self.mock_client,
            context=self.context
        )
    
    def test_set_status_calls_correct_method(self):
        """Test that set_status calls client.assistant_threads_setStatus with correct args."""
        self.assistant.set_status("Thinking...")
        
        self.mock_client.assistant_threads_setStatus.assert_called_once_with(
            channel_id="C123456",
            thread_ts="1234567890.123456",
            status="Thinking..."
        )
    
    def test_emit_trace_accumulates_lines(self):
        """Test that emit_trace appends to trace log."""
        self.assistant.emit_trace("decompose", "3 sub-queries")
        self.assistant.emit_trace("search", "#pricing")
        self.assistant.emit_trace("synthesis", "5 evidence items")
        
        self.assertEqual(len(self.assistant.trace_log), 3)
        self.assertEqual(self.assistant.trace_log[0], "decompose: 3 sub-queries")
        self.assertEqual(self.assistant.trace_log[1], "search: #pricing")
        self.assertEqual(self.assistant.trace_log[2], "synthesis: 5 evidence items")
    
    def test_trace_log_returns_copy(self):
        """Test that trace_log returns a copy, not the original list."""
        self.assistant.emit_trace("test", "value")
        trace_copy = self.assistant.trace_log
        
        # Modify the returned list
        trace_copy.append("modified")
        
        # Original should be unchanged
        self.assertEqual(len(self.assistant.trace_log), 1)
        self.assertNotIn("modified", self.assistant.trace_log)
    
    def test_post_result_calls_chat_postMessage_once(self):
        """Test that post_result calls chat_postMessage exactly once."""
        answer = Answer(
            text="The answer is 42.",
            citations=[]
        )
        
        self.assistant.post_result(answer, "https://canvas.example.com/report")
        
        self.mock_client.chat_postMessage.assert_called_once()
    
    def test_post_result_includes_context_block(self):
        """Test that post_result includes context block with trace lines."""
        self.assistant.emit_trace("decompose", "2 sub-queries")
        self.assistant.emit_trace("search", "#engineering")
        
        answer = Answer(
            text="The answer is 42.",
            citations=[]
        )
        
        self.assistant.post_result(answer, "https://canvas.example.com/report")
        
        call_args = self.mock_client.chat_postMessage.call_args
        blocks = call_args.kwargs["blocks"]
        
        # Should have at least one context block
        context_blocks = [b for b in blocks if b.get("type") == "section"]
        self.assertGreaterEqual(len(context_blocks), 1)
        
        # First block should contain trace info
        self.assertIn("Research Trace:", blocks[0]["text"]["text"])
        self.assertIn("decompose: 2 sub-queries", blocks[0]["text"]["text"])
    
    def test_post_result_includes_citation_blocks(self):
        """Test that post_result includes blocks for each citation."""
        answer = Answer(
            text="The price changed [1] and then [2].",
            citations=[
                Citation(
                    index=1,
                    permalink="https://slack.com/archives/C123/p123",
                    channel="pricing",
                    quote="We changed the price to $20"
                ),
                Citation(
                    index=2,
                    permalink="https://slack.com/archives/C456/p456",
                    channel="engineering",
                    quote="Updated the pricing service"
                )
            ]
        )
        
        self.assistant.post_result(answer, "https://canvas.example.com/report")
        
        call_args = self.mock_client.chat_postMessage.call_args
        blocks = call_args.kwargs["blocks"]
        
        # Should have citation blocks
        citation_blocks = [
            b for b in blocks 
            if b.get("type") == "section" and "#pricing" in b.get("text", {}).get("text", "")
        ]
        self.assertEqual(len(citation_blocks), 1)
        
        # Check for both citations
        block_texts = [b["text"]["text"] for b in blocks if b.get("type") == "section"]
        self.assertTrue(any("#pricing" in t for t in block_texts))
        self.assertTrue(any("#engineering" in t for t in block_texts))
    
    def test_post_result_includes_final_blocks(self):
        """Test that post_result includes final_block output."""
        answer = Answer(
            text="The answer is 42.",
            citations=[]
        )
        
        self.assistant.post_result(answer, "https://canvas.example.com/report")
        
        call_args = self.mock_client.chat_postMessage.call_args
        blocks = call_args.kwargs["blocks"]
        
        # Should have at least one action block from final_block
        action_blocks = [b for b in blocks if b.get("type") == "actions"]
        self.assertGreaterEqual(len(action_blocks), 1)


class TestResearchAssistantIntegration(unittest.TestCase):
    """Integration tests for ResearchAssistant with research.run."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_client = MagicMock()
        self.context = AssistantContext(
            channel="C123456",
            thread_ts="1234567890.123456"
        )
        self.assistant = ResearchAssistant(
            client=self.mock_client,
            context=self.context
        )
    
    def test_run_calls_set_status_multiple_times(self):
        """Test that run() calls set_status at least 3 times when assistant is provided."""
        from conduit.research import run
        from conduit.agent import LLMClient
        
        # Create a fake RTS that returns some evidence
        class FakeRTS:
            def search(self, query, limit=10):
                from conduit.rts_client import SearchHit
                return [
                    SearchHit(
                        text="We decided on Python",
                        channel="engineering",
                        ts="1234567890.123456",
                        permalink="https://slack.com/archives/C123/p123",
                        score=0.9,
                        author="alice"
                    )
                ]
        
        # Create a fake LLM that returns sub-queries
        class FakeLLM(LLMClient):
            def chat(self, messages, tools=None):
                if "Decompose" in messages[0]["content"]:
                    return {"content": "What language?\nWhat framework?"}
                return {"content": "The team uses Python."}
        
        result = run(
            question="What tech stack are we using?",
            rts=FakeRTS(),
            llm=FakeLLM(),
            assistant=self.assistant
        )
        
        # set_status should be called at least 3 times (decompose, search, synthesis)
        self.assertGreaterEqual(
            self.mock_client.assistant_threads_setStatus.call_count,
            3
        )
    
    def test_run_trace_log_starts_with_decompose(self):
        """Test that trace_log first entry starts with decompose count."""
        from conduit.research import run
        from conduit.agent import LLMClient
        
        class FakeRTS:
            def search(self, query, limit=10):
                from conduit.rts_client import SearchHit
                return [
                    SearchHit(
                        text="We decided on Python",
                        channel="engineering",
                        ts="1234567890.123456",
                        permalink="https://slack.com/archives/C123/p123",
                        score=0.9,
                        author="alice"
                    )
                ]
        
        class FakeLLM(LLMClient):
            def chat(self, messages, tools=None):
                if "Decompose" in messages[0]["content"]:
                    return {"content": "What language?\nWhat framework?"}
                return {"content": "The team uses Python."}
        
        result = run(
            question="What tech stack are we using?",
            rts=FakeRTS(),
            llm=FakeLLM(),
            assistant=self.assistant
        )
        
        # First trace entry should start with decompose
        self.assertTrue(
            self.assistant.trace_log[0].startswith("decompose:")
        )
        self.assertIn("sub-queries", self.assistant.trace_log[0])
    
    def test_run_without_assistant_works_normally(self):
        """Test that run() works normally when assistant is None."""
        from conduit.research import run
        from conduit.agent import LLMClient
        
        class FakeRTS:
            def search(self, query, limit=10):
                from conduit.rts_client import SearchHit
                return [
                    SearchHit(
                        text="We decided on Python",
                        channel="engineering",
                        ts="1234567890.123456",
                        permalink="https://slack.com/archives/C123/p123",
                        score=0.9,
                        author="alice"
                    )
                ]
        
        class FakeLLM(LLMClient):
            def chat(self, messages, tools=None):
                if "Decompose" in messages[0]["content"]:
                    return {"content": "What language?"}
                return {"content": "The team uses Python."}
        
        result = run(
            question="What tech stack are we using?",
            rts=FakeRTS(),
            llm=FakeLLM(),
            assistant=None
        )
        
        # Should return a valid ResearchResult
        self.assertEqual(result.question, "What tech stack are we using?")
        self.assertGreater(len(result.evidence), 0)
        
        # No calls to mock client
        self.mock_client.assistant_threads_setStatus.assert_not_called()
        self.mock_client.chat_postMessage.assert_not_called()


class TestSuggestedPrompts(unittest.TestCase):
    """M17: suggested starter prompts for the assistant thread."""

    def test_suggested_prompts_returns_list_of_dicts(self):
        from conduit.assistant_surface import suggested_prompts
        prompts = suggested_prompts("C12345")
        self.assertIsInstance(prompts, list)
        for p in prompts:
            self.assertIn("title", p)
            self.assertIn("message", p)

    def test_suggested_prompts_nonempty(self):
        from conduit.assistant_surface import suggested_prompts
        self.assertGreaterEqual(len(suggested_prompts()), 3)

    def test_suggested_prompts_all_nonempty_strings(self):
        from conduit.assistant_surface import suggested_prompts
        for p in suggested_prompts():
            self.assertIsInstance(p["title"], str)
            self.assertIsInstance(p["message"], str)
            self.assertTrue(p["title"].strip())
            self.assertTrue(p["message"].strip())

    def test_one_prompt_is_newcomer_framed(self):
        """For-Good evidence on screen: at least one prompt is newcomer/onboarding framed."""
        from conduit.assistant_surface import suggested_prompts
        joined = " ".join(p["title"].lower() for p in suggested_prompts())
        self.assertTrue("new here" in joined or "onboard" in joined or "story behind" in joined)

    def test_thread_started_sets_prompts(self):
        """assistant_thread_started wires set_suggested_prompts with a non-empty list."""
        import conduit.slack_app as slack_app
        set_prompts = MagicMock()
        say = MagicMock()
        slack_app.assistant_thread_started(
            payload={"channel_id": "C1"}, set_suggested_prompts=set_prompts, say=say)
        set_prompts.assert_called_once()
        kwargs = set_prompts.call_args.kwargs
        self.assertTrue(len(kwargs["prompts"]) >= 3)
        say.assert_called_once()


class TestStreamingTrace(unittest.TestCase):
    """P2-1: live streaming trace — one message posted then edited in place."""

    def setUp(self):
        self.client = MagicMock()
        self.client.chat_postMessage.return_value = {"ts": "1700000000.000100"}
        self.ctx = AssistantContext(channel="C1", thread_ts="1700000000.000000")
        self.assistant = ResearchAssistant(self.client, self.ctx, stream=True)

    def test_first_step_posts_then_updates(self):
        self.assistant.emit_trace("decompose", "2 sub-queries")
        self.assistant.emit_trace("search", "#pricing → 4 hits")
        self.assistant.emit_trace("synthesis", "5 evidence items")
        # exactly one post, then updates in place
        self.client.chat_postMessage.assert_called_once()
        self.assertEqual(self.client.chat_update.call_count, 2)
        # the updated message accumulates trace blocks (header + 3 steps)
        last_blocks = self.client.chat_update.call_args.kwargs["blocks"]
        self.assertGreaterEqual(len(last_blocks), 4)
        dumped = str(last_blocks)
        self.assertIn("Decompose", dumped)
        self.assertIn("Synthesis", dumped)

    def test_update_targets_same_message_ts(self):
        self.assistant.emit_trace("decompose", "x")
        self.assistant.emit_trace("search", "y")
        self.assertEqual(self.client.chat_update.call_args.kwargs["ts"], "1700000000.000100")

    def test_non_streaming_makes_no_calls(self):
        buffered = ResearchAssistant(self.client, self.ctx)  # stream defaults False
        buffered.emit_trace("decompose", "x")
        self.client.chat_postMessage.assert_not_called()
        self.client.chat_update.assert_not_called()
        self.assertEqual(buffered.trace_log, ["decompose: x"])
