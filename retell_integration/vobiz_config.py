"""
Vobiz.ai Telephony Configuration for Indian PSTN Integration

Vobiz.ai provides Indian DID numbers and SIP trunking for Retell.AI integration,
allowing callers to reach the Palli Sahayak helpline via regular phone (+91).

No internet required for callers - they use standard cellular/landline phones.

Reference: https://vobiz.ai/
"""

import os
import logging
from typing import Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class VobizConfig:
    """
    Configuration for Vobiz.ai Indian telephony integration.

    Vobiz.ai provides:
    - Indian DID numbers (+91 prefixed)
    - SIP trunking to connect with Retell.AI
    - Low latency for real-time voice
    - Compliance with Indian telecom regulations (TRAI)

    Flow:
    1. Caller dials +91 Vobiz number from any phone
    2. Vobiz receives call and routes via SIP
    3. Retell receives call and connects to agent
    4. Agent uses Custom LLM WebSocket for responses
    """

    # Vobiz.ai API credentials
    api_key: str = ""
    api_secret: str = ""

    # Indian DID number for inbound calls
    did_number: str = ""  # Format: +919876543210

    # SIP configuration for Retell integration
    sip_domain: str = ""
    sip_username: str = ""
    sip_password: str = ""
    sip_port: int = 5060

    # Retell SIP trunk settings (obtained after Retell setup)
    retell_sip_trunk_id: str = ""
    retell_inbound_number_id: str = ""

    # Advanced settings
    codec: str = "PCMU"  # G.711 u-law, best for telephony
    dtmf_mode: str = "rfc2833"

    def __post_init__(self):
        """Load from environment if not provided."""
        if not self.api_key:
            self.api_key = os.getenv("VOBIZ_API_KEY", "")
        if not self.api_secret:
            self.api_secret = os.getenv("VOBIZ_API_SECRET", "")
        if not self.did_number:
            self.did_number = os.getenv("VOBIZ_DID_NUMBER", "")
        if not self.sip_domain:
            self.sip_domain = os.getenv("VOBIZ_SIP_DOMAIN", "")
        if not self.sip_username:
            self.sip_username = os.getenv("VOBIZ_SIP_USERNAME", "")
        if not self.sip_password:
            self.sip_password = os.getenv("VOBIZ_SIP_PASSWORD", "")
        if not self.retell_sip_trunk_id:
            self.retell_sip_trunk_id = os.getenv("RETELL_SIP_TRUNK_ID", "")
        if not self.retell_inbound_number_id:
            self.retell_inbound_number_id = os.getenv("RETELL_INBOUND_NUMBER_ID", "")

    def is_configured(self) -> bool:
        """Check if Vobiz is minimally configured."""
        return bool(self.did_number)

    def is_fully_configured(self) -> bool:
        """Check if Vobiz is fully configured with SIP."""
        return bool(
            self.api_key and
            self.did_number and
            self.sip_domain and
            self.sip_username and
            self.sip_password
        )

    def get_sip_uri(self) -> str:
        """Get SIP URI for Retell integration."""
        if self.sip_domain and self.sip_username:
            return f"sip:{self.sip_username}@{self.sip_domain}:{self.sip_port}"
        return ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding secrets)."""
        return {
            "did_number": self.did_number,
            "sip_domain": self.sip_domain,
            "sip_port": self.sip_port,
            "configured": self.is_configured(),
            "fully_configured": self.is_fully_configured(),
            "has_sip_trunk": bool(self.retell_sip_trunk_id),
            "codec": self.codec
        }

    def get_retell_sip_trunk_config(self) -> Dict[str, Any]:
        """
        Get configuration for Retell SIP trunk setup.

        This is used when configuring Retell to receive calls
        from Vobiz.ai via SIP.

        Reference: https://docs.retellai.com/deploy/custom-telephony

        Returns:
            Dictionary for Retell create-sip-trunk API
        """
        config = {
            "inbound_sip_trunk_settings": {
                "termination_uri": self.get_sip_uri()
            }
        }

        if self.sip_username and self.sip_password:
            config["inbound_sip_trunk_settings"]["authentication"] = {
                "type": "credentials",
                "username": self.sip_username,
                "password": self.sip_password
            }

        return config

    def get_retell_phone_number_config(self, agent_id: str) -> Dict[str, Any]:
        """
        Get configuration for importing Vobiz number to Retell.

        Args:
            agent_id: Retell agent ID to assign to this number

        Returns:
            Dictionary for Retell import-phone-number API
        """
        return {
            "phone_number": self.did_number,
            "inbound_agent_id": agent_id,
            "sip_trunk_id": self.retell_sip_trunk_id
        }


def get_vobiz_config() -> VobizConfig:
    """
    Get Vobiz configuration from environment.

    Environment variables:
    - VOBIZ_API_KEY: Vobiz.ai API key
    - VOBIZ_API_SECRET: Vobiz.ai API secret
    - VOBIZ_DID_NUMBER: Indian DID number (+91...)
    - VOBIZ_SIP_DOMAIN: SIP server domain
    - VOBIZ_SIP_USERNAME: SIP authentication username
    - VOBIZ_SIP_PASSWORD: SIP authentication password
    - RETELL_SIP_TRUNK_ID: Retell SIP trunk ID (after setup)
    - RETELL_INBOUND_NUMBER_ID: Retell inbound number ID (after import)

    Returns:
        VobizConfig instance
    """
    return VobizConfig()


# Instructions for setting up Vobiz.ai with Retell
VOBIZ_SETUP_INSTRUCTIONS = """
# Vobiz.ai + Retell.AI Integration Setup Guide

## Overview

This guide explains how to connect a Vobiz.ai Indian phone number (+91)
to Retell.AI for the Palli Sahayak palliative care voice helpline.

## Prerequisites

1. Vobiz.ai account with API access
2. Retell.AI account with Custom Telephony enabled
3. Server with Custom LLM WebSocket endpoint deployed

## Step 1: Get Indian DID from Vobiz.ai

1. Log into Vobiz.ai dashboard (https://dashboard.vobiz.ai)
2. Navigate to Numbers > Buy Number
3. Select India (+91) and choose your preferred city/region
4. Purchase the DID number (e.g., +919876543210)
5. Note down:
   - DID Number
   - SIP Domain (e.g., sip.vobiz.ai)
   - SIP Username
   - SIP Password

## Step 2: Deploy Custom LLM Server

1. Deploy your server with the WebSocket endpoint:
   - wss://your-domain.com/ws/retell/llm

2. Ensure the server is accessible from the internet
   - Use ngrok for development: `ngrok http 8000`
   - Use proper domain for production

3. Test the endpoint is reachable

## Step 3: Create Retell Agent with Custom LLM

```python
from retell_integration import RetellClient, get_palli_sahayak_retell_config

client = RetellClient()
config = get_palli_sahayak_retell_config(
    llm_websocket_url="wss://your-domain.com/ws/retell/llm",
    webhook_url="https://your-domain.com/api/retell/webhook",
    language="hi"
)
result = await client.create_agent(config)
agent_id = result.agent_id
print(f"Created agent: {agent_id}")
```

## Step 4: Create SIP Trunk in Retell

1. Go to Retell Dashboard > Settings > Custom Telephony
2. Click "Add SIP Trunk"
3. Configure:
   - Name: "Vobiz India"
   - Termination URI: sip:username@sip.vobiz.ai:5060
   - Authentication: Username/Password from Vobiz
4. Save and note the SIP Trunk ID

## Step 5: Import Phone Number to Retell

1. In Retell Dashboard > Phone Numbers
2. Click "Import Number"
3. Enter your Vobiz +91 number
4. Select the SIP trunk created in Step 4
5. Assign the agent created in Step 3

## Step 6: Configure Vobiz Routing

1. In Vobiz Dashboard > Numbers > Your DID
2. Configure inbound routing:
   - Destination Type: SIP
   - SIP URI: (Retell's SIP endpoint - from SIP trunk details)
3. Save configuration

## Step 7: Set Environment Variables

Add to your .env file:

```bash
# Vobiz.ai Configuration
VOBIZ_API_KEY=your_vobiz_api_key
VOBIZ_API_SECRET=your_vobiz_secret
VOBIZ_DID_NUMBER=+919876543210
VOBIZ_SIP_DOMAIN=sip.vobiz.ai
VOBIZ_SIP_USERNAME=your_sip_username
VOBIZ_SIP_PASSWORD=your_sip_password

# Retell Configuration (after setup)
RETELL_API_KEY=your_retell_api_key
RETELL_AGENT_ID=agent_id_from_step_3
RETELL_SIP_TRUNK_ID=trunk_id_from_step_4
RETELL_INBOUND_NUMBER_ID=number_id_from_step_5
```

## Step 8: Test the Integration

1. Start your server:
   ```bash
   python simple_rag_server.py -p r
   ```

2. Call the +91 number from any phone

3. Verify:
   - Call connects to Retell
   - Welcome message plays
   - Your responses are processed via Custom LLM
   - RAG queries return knowledge base content

## Troubleshooting

### Call Not Connecting
- Check Vobiz dashboard for call logs
- Verify SIP trunk configuration in Retell
- Ensure SIP credentials are correct

### No Audio
- Check codec compatibility (G.711 recommended)
- Verify network firewall allows SIP/RTP traffic

### Custom LLM Not Responding
- Check WebSocket endpoint is accessible
- Review server logs for connection attempts

### High Latency
- Use a server closer to India
- Consider Vobiz PoP locations
"""
