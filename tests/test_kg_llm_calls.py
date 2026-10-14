"""Knowledge-graph LLM calls go through the shared client and degrade without it."""
import asyncio
import json

import pytest

from llm import LLMResponse, LLMSettings, get_llm_client, set_llm_client
from knowledge_graph.cypher_generator import CypherGenerator
from knowledge_graph.entity_extractor import EntityExtractor


class FakeClient:
    def __init__(self, text="", ok=True):
        self.settings = LLMSettings(api_key="test-key")
        self.calls = []
        self._text = text
        self._ok = ok

    async def generate(self, prompt, **kw):
        self.calls.append({"prompt": prompt, **kw})
        if self._ok:
            return LLMResponse(self._text, True, "gemini", self.settings.model)
        return LLMResponse("", False, "none", self.settings.model, error="gemini: down")


@pytest.fixture
def install():
    previous = get_llm_client()

    def _install(client):
        set_llm_client(client)
        return client

    yield _install
    set_llm_client(previous)


LLM_JSON = json.dumps({
    "entities": [{"name": "Haloperidol", "type": "Medication"}],
    "relationships": [{
        "source": "Haloperidol", "source_type": "Medication",
        "target": "Nausea", "target_type": "Symptom", "relationship": "TREATS",
    }],
})

TEXT = "Morphine is used for breakthrough pain. Haloperidol helps nausea."


def test_extractor_uses_shared_client(install):
    fake = install(FakeClient(LLM_JSON))
    entities, relationships = asyncio.run(EntityExtractor().extract(TEXT))
    names = {e.name.lower() for e in entities}
    assert "haloperidol" in names
    assert len(relationships) == 1
    assert fake.calls[0]["purpose"] == "kg_entity_extraction"
    assert fake.calls[0]["json_mode"] is True


def test_extractor_provider_none_never_calls_llm(install):
    fake = install(FakeClient(LLM_JSON))
    entities, relationships = asyncio.run(
        EntityExtractor(llm_provider="none", use_patterns=True).extract(TEXT)
    )
    assert fake.calls == []
    assert relationships == []
    assert any("morphine" in e.name.lower() for e in entities)


def test_extractor_use_llm_false_never_calls_llm(install):
    fake = install(FakeClient(LLM_JSON))
    asyncio.run(EntityExtractor().extract(TEXT, use_llm=False))
    assert fake.calls == []


def test_extractor_degrades_to_patterns_when_llm_fails(install):
    install(FakeClient(ok=False))
    entities, relationships = asyncio.run(EntityExtractor().extract(TEXT))
    assert relationships == []
    assert any("morphine" in e.name.lower() for e in entities)


UNMATCHED = "Who looks after night-time agitation?"


def test_cypher_template_match_skips_llm(install):
    fake = install(FakeClient("MATCH (n) RETURN n LIMIT 5"))
    q = asyncio.run(CypherGenerator().generate("What medications treat pain?"))
    assert q.template_used == "medication_for_symptom"
    assert fake.calls == []


def test_cypher_llm_generated(install):
    fake = install(FakeClient("```cypher\nMATCH (s:Symptom) RETURN s.name AS name LIMIT 5\n```"))
    q = asyncio.run(CypherGenerator().generate(UNMATCHED))
    assert q.template_used == "llm_generated"
    assert "```" not in q.query
    assert fake.calls[0]["purpose"] == "kg_cypher_generation"


def test_cypher_llm_write_query_rejected(install):
    install(FakeClient("MATCH (n) DETACH DELETE n"))
    q = asyncio.run(CypherGenerator().generate(UNMATCHED))
    assert q.template_used == "llm_fallback"


def test_cypher_degrades_to_default_search_when_llm_fails(install):
    install(FakeClient(ok=False))
    q = asyncio.run(CypherGenerator().generate(UNMATCHED))
    assert q.template_used == "default_search"


def test_cypher_use_llm_false_never_calls_llm(install):
    fake = install(FakeClient("MATCH (n) RETURN n LIMIT 5"))
    q = asyncio.run(CypherGenerator(use_llm=False).generate(UNMATCHED))
    assert q.template_used == "default_search"
    assert fake.calls == []
