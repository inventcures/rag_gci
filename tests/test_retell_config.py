"""
Tests for Retell.AI Configuration Module

Tests configuration, voice settings, and agent config generation.
"""

import os
import sys
import pytest
from unittest.mock import patch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestRetellConfig:
    """Test Retell configuration module."""

    def test_import_config_module(self):
        """Test that config module can be imported."""
        from retell_integration.config import (
            RETELL_API_BASE,
            RETELL_SYSTEM_PROMPT,
            RetellAgentConfig,
            get_palli_sahayak_retell_config,
        )
        assert RETELL_API_BASE is not None
        assert RETELL_SYSTEM_PROMPT is not None

    def test_cartesia_voice_ids(self):
        """The Cartesia voices the table names are all known, as UUIDs."""
        from supported_languages import CARTESIA_VOICES, PROVIDER_RETELL, SUPPORTED_LANGUAGES

        for lang in SUPPORTED_LANGUAGES:
            voice = lang.voice(PROVIDER_RETELL)
            if voice is not None:
                assert len(CARTESIA_VOICES[voice.name]) == 36  # UUID format

    def test_language_configs(self):
        """Hindi and English agents listen in hi-IN and en-IN (the table's languages)."""
        from retell_integration.config import get_palli_sahayak_retell_config

        assert get_palli_sahayak_retell_config("wss://x", language="hi").language == "hi-IN"
        assert get_palli_sahayak_retell_config("wss://x", language="en").language == "en-IN"

    def test_system_prompt_content(self):
        """Test system prompt contains required elements."""
        from retell_integration.config import RETELL_SYSTEM_PROMPT

        # Should mention palliative care
        assert "palliative" in RETELL_SYSTEM_PROMPT.lower() or "palli" in RETELL_SYSTEM_PROMPT.lower()

        # Should be substantial
        assert len(RETELL_SYSTEM_PROMPT) > 100

    def test_retell_agent_config_dataclass(self):
        """Test RetellAgentConfig dataclass."""
        from retell_integration.config import RetellAgentConfig
        from supported_languages import CARTESIA_VOICES

        config = RetellAgentConfig(
            agent_name="Test Agent",
            llm_websocket_url="wss://example.com/ws",
            voice_id=CARTESIA_VOICES["Hindi Narrator Woman"],
            webhook_url="https://example.com/webhook"
        )

        assert config.llm_websocket_url == "wss://example.com/ws"
        assert config.webhook_url == "https://example.com/webhook"
        assert config.voice_id is not None
        assert config.language == "hi-IN"  # Default language

    def test_retell_agent_config_to_dict(self):
        """Test RetellAgentConfig.to_dict() method."""
        from retell_integration.config import RetellAgentConfig
        from supported_languages import CARTESIA_VOICES

        config = RetellAgentConfig(
            agent_name="Test Agent",
            llm_websocket_url="wss://example.com/ws",
            voice_id=CARTESIA_VOICES["Hindi Narrator Woman"],
            webhook_url="https://example.com/webhook"
        )

        result = config.to_dict()

        assert isinstance(result, dict)
        assert result["agent_name"] == "Test Agent"
        assert "llm_websocket_url" in result
        assert "voice_id" in result

    def test_get_palli_sahayak_config(self):
        """Test Palli Sahayak agent configuration helper."""
        from retell_integration.config import get_palli_sahayak_retell_config

        config = get_palli_sahayak_retell_config(
            llm_websocket_url="wss://example.com/ws/retell/llm",
            webhook_url="https://example.com/api/retell/webhook",
            language="hi"
        )

        assert "Palli Sahayak" in config.agent_name
        assert config.language == "hi-IN"
        assert "retell" in config.llm_websocket_url

    def test_get_palli_sahayak_config_english(self):
        """Test configuration for English language."""
        from retell_integration.config import get_palli_sahayak_retell_config

        config = get_palli_sahayak_retell_config(
            llm_websocket_url="wss://example.com/ws",
            webhook_url="https://example.com/webhook",
            language="en"
        )

        assert config.language == "en-IN"
        # Voice should be different from Hindi
        assert config.voice_id is not None


class TestVobizConfig:
    """Test Vobiz.ai telephony configuration."""

    def test_import_vobiz_module(self):
        """Test that Vobiz config module can be imported."""
        from retell_integration.vobiz_config import (
            VobizConfig,
            get_vobiz_config,
        )
        assert VobizConfig is not None
        assert get_vobiz_config is not None

    def test_vobiz_config_dataclass(self):
        """Test VobizConfig dataclass."""
        from retell_integration.vobiz_config import VobizConfig

        config = VobizConfig(
            api_key="test_key",
            did_number="+919876543210",
            sip_domain="sip.vobiz.ai",
            sip_username="testuser",
            sip_password="testpass"
        )

        assert config.did_number == "+919876543210"
        assert config.sip_domain == "sip.vobiz.ai"
        assert config.sip_port == 5060  # Default

    def test_vobiz_config_is_configured(self):
        """Test is_configured() method."""
        from retell_integration.vobiz_config import VobizConfig

        # Not configured (no DID)
        config1 = VobizConfig()
        assert not config1.is_configured()

        # Configured (has DID)
        config2 = VobizConfig(did_number="+919876543210")
        assert config2.is_configured()

    def test_vobiz_config_is_fully_configured(self):
        """Test is_fully_configured() method."""
        from retell_integration.vobiz_config import VobizConfig

        # Not fully configured
        config1 = VobizConfig(did_number="+919876543210")
        assert not config1.is_fully_configured()

        # Fully configured
        config2 = VobizConfig(
            api_key="key",
            did_number="+919876543210",
            sip_domain="sip.vobiz.ai",
            sip_username="user",
            sip_password="pass"
        )
        assert config2.is_fully_configured()

    def test_vobiz_get_sip_uri(self):
        """Test SIP URI generation."""
        from retell_integration.vobiz_config import VobizConfig

        config = VobizConfig(
            sip_domain="sip.vobiz.ai",
            sip_username="testuser",
            sip_port=5060
        )

        uri = config.get_sip_uri()
        assert uri == "sip:testuser@sip.vobiz.ai:5060"

    def test_vobiz_to_dict(self):
        """Test to_dict() excludes secrets."""
        from retell_integration.vobiz_config import VobizConfig

        config = VobizConfig(
            api_key="secret_key",
            api_secret="secret",
            did_number="+919876543210",
            sip_password="secret_pass"
        )

        result = config.to_dict()

        # Should NOT contain secrets
        assert "api_key" not in result
        assert "api_secret" not in result
        assert "sip_password" not in result

        # Should contain public info
        assert result["did_number"] == "+919876543210"
        assert "configured" in result

    @patch.dict(os.environ, {"VOBIZ_DID_NUMBER": "+911234567890"})
    def test_vobiz_config_from_environment(self):
        """Test loading config from environment variables."""
        from retell_integration.vobiz_config import get_vobiz_config

        config = get_vobiz_config()
        assert config.did_number == "+911234567890"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
