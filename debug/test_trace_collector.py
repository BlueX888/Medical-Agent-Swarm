import unittest

from core.llm_client import LLMResponse

from .trace_collector import DebugTraceCollector


class DebugTraceCollectorTest(unittest.TestCase):
    def test_event_sequence_metadata_and_redaction(self):
        collector = DebugTraceCollector(
            question="test",
            context={"api_key": "secret", "age": "40"},
            metadata={"Authorization": "Bearer secret", "source": "test"},
        )

        first = collector.record_event(
            "llm_call",
            input={"api_key": "secret", "messages": ["hello"]},
            metadata={"secret_key": "secret", "model": "test-model"},
        )
        second = collector.record_event("skill_call", output={"ok": True})

        run = collector.get_run().to_dict()
        events = [event.to_dict() for event in collector.get_events()]

        self.assertEqual(first.sequence, 1)
        self.assertEqual(second.sequence, 2)
        self.assertEqual(events[0]["metadata"]["model"], "test-model")
        self.assertEqual(run["context"]["api_key"], "[redacted]")
        self.assertEqual(run["metadata"]["Authorization"], "[redacted]")
        self.assertEqual(events[0]["input"]["api_key"], "[redacted]")
        self.assertEqual(events[0]["metadata"]["secret_key"], "[redacted]")

    def test_long_values_are_truncated_with_marker(self):
        collector = DebugTraceCollector(question="test")
        collector.record_event("raw", output={"text": "x" * 20050})
        event = collector.get_events()[0].to_dict()

        self.assertIn("[truncated", event["output"]["text"])


class LLMResponseTest(unittest.TestCase):
    def test_usage_fields_are_optional(self):
        response = LLMResponse(content="ok", tool_calls=[], finish_reason="stop")

        self.assertEqual(response.content, "ok")
        self.assertEqual(response.tool_calls, [])
        self.assertIsNone(response.usage)
        self.assertIsNone(response.model)
        self.assertIsNone(response.response_id)


if __name__ == "__main__":
    unittest.main()
