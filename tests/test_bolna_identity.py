"""
Each Bolna caller is their own participant.

Before this, every Bolna request carried source="bolna_call", and the endpoint
used f"bolna_{source}" as the user id: every caller shared one participant, one
conversation history and one study pseudonym. Bolna fills system variables into
the custom function's params, so the agent config now sends from_number and
execution_id and the endpoint keys on them.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simple_rag_server import bolna_caller_id  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class CallerId(unittest.TestCase):
    def test_two_callers_are_two_participants(self):
        a = bolna_caller_id({"source": "bolna_call", "from_number": "+919812345678", "execution_id": "e1"})
        b = bolna_caller_id({"source": "bolna_call", "from_number": "+919800000000", "execution_id": "e2"})
        self.assertNotEqual(a, b)

    def test_the_same_caller_keeps_one_participant_across_calls(self):
        first = bolna_caller_id({"from_number": "+919812345678", "execution_id": "e1"})
        second = bolna_caller_id({"from_number": "+919812345678", "execution_id": "e2"})
        self.assertEqual(first, second)

    def test_call_id_when_the_number_is_missing(self):
        self.assertEqual(bolna_caller_id({"execution_id": "e1"}), "session:bolna:e1")
        self.assertEqual(bolna_caller_id({"call_sid": "CA1"}), "session:bolna:CA1")

    def test_unfilled_templates_are_not_identities(self):
        caller = bolna_caller_id({"from_number": "%(from_number)s", "execution_id": "e9"})
        self.assertEqual(caller, "session:bolna:e9")

    def test_no_identity_is_never_shared(self):
        a = bolna_caller_id({"source": "bolna_call"})
        b = bolna_caller_id({"source": "bolna_call"})
        self.assertTrue(a.startswith("session:bolna:anon_"))
        self.assertNotEqual(a, b)

    def test_pseudonyms_differ_in_the_study_log(self):
        from study_logging import pseudonymise

        a = bolna_caller_id({"from_number": "+919812345678"})
        b = bolna_caller_id({"from_number": "+919800000000"})
        self.assertNotEqual(pseudonymise(a), pseudonymise(b))


class Wiring(unittest.TestCase):
    def test_the_endpoint_uses_the_caller_id_everywhere(self):
        source = open(os.path.join(ROOT, "simple_rag_server.py"), encoding="utf-8").read()
        start = source.index('@app.post("/api/bolna/query"')
        route = source[start:source.index("@app.", start + 10)]
        self.assertNotIn('bolna_{source}', route)
        self.assertEqual(route.count("user_id=caller_id"), 3)

    def test_the_agent_config_sends_the_system_variables(self):
        from bolna_integration.config import get_palli_sahayak_agent_config

        config = get_palli_sahayak_agent_config(server_url="https://example.org", language="hi")
        text = repr(config)
        self.assertIn("%(from_number)s", text)
        self.assertIn("%(execution_id)s", text)


if __name__ == "__main__":
    unittest.main()
