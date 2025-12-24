"""
Bolna API Client for Palli Sahayak Voice AI Helpline

This module provides the BolnaClient class for interacting with the Bolna.ai API.
It handles agent creation, call management, and webhook configuration.

Documentation: https://www.bolna.ai/docs/introduction
"""

import os
import logging
import aiohttp
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

BOLNA_API_BASE = "https://api.bolna.dev/v1"


@dataclass
class BolnaCallResult:
    """Result of a Bolna API call."""
    success: bool
    call_id: Optional[str] = None
    agent_id: Optional[str] = None
    error: Optional[str] = None
    data: Optional[Dict[str, Any]] = None


class BolnaClient:
    """
    Client for Bolna.ai API.

    Handles:
    - Agent creation and management
    - Call initiation (outbound)
    - Phone number management
    - Webhook configuration

    Usage:
        client = BolnaClient()

        # Create agent
        result = await client.create_agent(agent_config)

        # Initiate call
        result = await client.initiate_call(
            agent_id="agent-123",
            phone_number="+919876543210"
        )
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Bolna client.

        Args:
            api_key: Bolna API key. If not provided, reads from BOLNA_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("BOLNA_API_KEY")
        if not self.api_key:
            logger.warning("Bolna API key not configured - set BOLNA_API_KEY env var")

        self.base_url = BOLNA_API_BASE
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def is_available(self) -> bool:
        """Check if Bolna client is configured with API key."""
        return bool(self.api_key)

    async def create_agent(self, config: Dict[str, Any]) -> BolnaCallResult:
        """
        Create a new Bolna agent.

        Args:
            config: Agent configuration dictionary (see config.py for structure)

        Returns:
            BolnaCallResult with agent_id if successful
        """
        if not self.is_available():
            return BolnaCallResult(success=False, error="Bolna API key not configured")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/agents",
                    headers=self.headers,
                    json=config
                ) as response:
                    data = await response.json()

                    if response.status in (200, 201):
                        logger.info(f"Created Bolna agent: {data.get('agent_id')}")
                        return BolnaCallResult(
                            success=True,
                            agent_id=data.get("agent_id"),
                            data=data
                        )
                    else:
                        error_msg = data.get("message", f"HTTP {response.status}")
                        logger.error(f"Failed to create agent: {error_msg}")
                        return BolnaCallResult(success=False, error=error_msg)

        except aiohttp.ClientError as e:
            logger.error(f"Network error creating Bolna agent: {e}")
            return BolnaCallResult(success=False, error=f"Network error: {e}")
        except Exception as e:
            logger.error(f"Failed to create Bolna agent: {e}")
            return BolnaCallResult(success=False, error=str(e))

    async def get_agent(self, agent_id: str) -> BolnaCallResult:
        """
        Get agent details.

        Args:
            agent_id: ID of the agent to retrieve

        Returns:
            BolnaCallResult with agent data
        """
        if not self.is_available():
            return BolnaCallResult(success=False, error="Bolna API key not configured")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/agents/{agent_id}",
                    headers=self.headers
                ) as response:
                    data = await response.json()

                    if response.status == 200:
                        return BolnaCallResult(
                            success=True,
                            agent_id=agent_id,
                            data=data
                        )
                    else:
                        return BolnaCallResult(
                            success=False,
                            error=data.get("message", f"HTTP {response.status}")
                        )

        except Exception as e:
            logger.error(f"Failed to get Bolna agent: {e}")
            return BolnaCallResult(success=False, error=str(e))

    async def update_agent(self, agent_id: str, config: Dict[str, Any]) -> BolnaCallResult:
        """
        Update an existing agent.

        Args:
            agent_id: ID of the agent to update
            config: Updated configuration

        Returns:
            BolnaCallResult
        """
        if not self.is_available():
            return BolnaCallResult(success=False, error="Bolna API key not configured")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.put(
                    f"{self.base_url}/agents/{agent_id}",
                    headers=self.headers,
                    json=config
                ) as response:
                    data = await response.json()

                    if response.status == 200:
                        logger.info(f"Updated Bolna agent: {agent_id}")
                        return BolnaCallResult(
                            success=True,
                            agent_id=agent_id,
                            data=data
                        )
                    else:
                        return BolnaCallResult(
                            success=False,
                            error=data.get("message", f"HTTP {response.status}")
                        )

        except Exception as e:
            logger.error(f"Failed to update Bolna agent: {e}")
            return BolnaCallResult(success=False, error=str(e))

    async def delete_agent(self, agent_id: str) -> BolnaCallResult:
        """
        Delete an agent.

        Args:
            agent_id: ID of the agent to delete

        Returns:
            BolnaCallResult
        """
        if not self.is_available():
            return BolnaCallResult(success=False, error="Bolna API key not configured")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.delete(
                    f"{self.base_url}/agents/{agent_id}",
                    headers=self.headers
                ) as response:
                    if response.status in (200, 204):
                        logger.info(f"Deleted Bolna agent: {agent_id}")
                        return BolnaCallResult(success=True, agent_id=agent_id)
                    else:
                        data = await response.json()
                        return BolnaCallResult(
                            success=False,
                            error=data.get("message", f"HTTP {response.status}")
                        )

        except Exception as e:
            logger.error(f"Failed to delete Bolna agent: {e}")
            return BolnaCallResult(success=False, error=str(e))

    async def list_agents(self) -> BolnaCallResult:
        """
        List all agents.

        Returns:
            BolnaCallResult with list of agents in data field
        """
        if not self.is_available():
            return BolnaCallResult(success=False, error="Bolna API key not configured")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/agents",
                    headers=self.headers
                ) as response:
                    data = await response.json()

                    if response.status == 200:
                        return BolnaCallResult(success=True, data=data)
                    else:
                        return BolnaCallResult(
                            success=False,
                            error=data.get("message", f"HTTP {response.status}")
                        )

        except Exception as e:
            logger.error(f"Failed to list Bolna agents: {e}")
            return BolnaCallResult(success=False, error=str(e))

    async def initiate_call(
        self,
        agent_id: str,
        phone_number: str,
        user_data: Optional[Dict[str, Any]] = None
    ) -> BolnaCallResult:
        """
        Initiate an outbound call.

        Args:
            agent_id: ID of the agent to use
            phone_number: Phone number to call (E.164 format, e.g., +919876543210)
            user_data: Optional user context data to pass to the agent

        Returns:
            BolnaCallResult with call_id if successful
        """
        if not self.is_available():
            return BolnaCallResult(success=False, error="Bolna API key not configured")

        try:
            payload = {
                "agent_id": agent_id,
                "recipient_phone_number": phone_number
            }

            if user_data:
                payload["user_data"] = user_data

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/calls",
                    headers=self.headers,
                    json=payload
                ) as response:
                    data = await response.json()

                    if response.status in (200, 201):
                        call_id = data.get("call_id") or data.get("id")
                        logger.info(f"Initiated call {call_id} to {phone_number}")
                        return BolnaCallResult(
                            success=True,
                            call_id=call_id,
                            agent_id=agent_id,
                            data=data
                        )
                    else:
                        error_msg = data.get("message", f"HTTP {response.status}")
                        logger.error(f"Failed to initiate call: {error_msg}")
                        return BolnaCallResult(success=False, error=error_msg)

        except Exception as e:
            logger.error(f"Failed to initiate Bolna call: {e}")
            return BolnaCallResult(success=False, error=str(e))

    async def get_call_status(self, call_id: str) -> BolnaCallResult:
        """
        Get status of a call.

        Args:
            call_id: ID of the call

        Returns:
            BolnaCallResult with call status in data field
        """
        if not self.is_available():
            return BolnaCallResult(success=False, error="Bolna API key not configured")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/calls/{call_id}",
                    headers=self.headers
                ) as response:
                    data = await response.json()

                    if response.status == 200:
                        return BolnaCallResult(
                            success=True,
                            call_id=call_id,
                            data=data
                        )
                    else:
                        return BolnaCallResult(
                            success=False,
                            error=data.get("message", f"HTTP {response.status}")
                        )

        except Exception as e:
            logger.error(f"Failed to get call status: {e}")
            return BolnaCallResult(success=False, error=str(e))

    async def end_call(self, call_id: str) -> BolnaCallResult:
        """
        End an active call.

        Args:
            call_id: ID of the call to end

        Returns:
            BolnaCallResult
        """
        if not self.is_available():
            return BolnaCallResult(success=False, error="Bolna API key not configured")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/calls/{call_id}/end",
                    headers=self.headers
                ) as response:
                    if response.status in (200, 204):
                        logger.info(f"Ended call: {call_id}")
                        return BolnaCallResult(success=True, call_id=call_id)
                    else:
                        data = await response.json()
                        return BolnaCallResult(
                            success=False,
                            error=data.get("message", f"HTTP {response.status}")
                        )

        except Exception as e:
            logger.error(f"Failed to end call: {e}")
            return BolnaCallResult(success=False, error=str(e))

    async def list_phone_numbers(self) -> BolnaCallResult:
        """
        List available phone numbers.

        Returns:
            BolnaCallResult with phone numbers in data field
        """
        if not self.is_available():
            return BolnaCallResult(success=False, error="Bolna API key not configured")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/phone-numbers",
                    headers=self.headers
                ) as response:
                    data = await response.json()

                    if response.status == 200:
                        return BolnaCallResult(success=True, data=data)
                    else:
                        return BolnaCallResult(
                            success=False,
                            error=data.get("message", f"HTTP {response.status}")
                        )

        except Exception as e:
            logger.error(f"Failed to list phone numbers: {e}")
            return BolnaCallResult(success=False, error=str(e))

    async def configure_webhook(
        self,
        agent_id: str,
        webhook_url: str,
        events: Optional[List[str]] = None
    ) -> BolnaCallResult:
        """
        Configure webhook for agent events.

        Args:
            agent_id: Agent ID
            webhook_url: URL to receive webhooks
            events: List of events to subscribe to (default: all)

        Returns:
            BolnaCallResult
        """
        if not self.is_available():
            return BolnaCallResult(success=False, error="Bolna API key not configured")

        if events is None:
            events = ["call_started", "call_ended", "extraction_completed", "transcription"]

        try:
            payload = {
                "webhook_url": webhook_url,
                "events": events
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/agents/{agent_id}/webhooks",
                    headers=self.headers,
                    json=payload
                ) as response:
                    data = await response.json()

                    if response.status in (200, 201):
                        logger.info(f"Configured webhook for agent {agent_id}")
                        return BolnaCallResult(
                            success=True,
                            agent_id=agent_id,
                            data=data
                        )
                    else:
                        return BolnaCallResult(
                            success=False,
                            error=data.get("message", f"HTTP {response.status}")
                        )

        except Exception as e:
            logger.error(f"Failed to configure webhook: {e}")
            return BolnaCallResult(success=False, error=str(e))

    async def health_check(self) -> bool:
        """
        Check if Bolna API is accessible.

        Returns:
            True if API is healthy, False otherwise
        """
        if not self.is_available():
            return False

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/health",
                    headers=self.headers,
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    return response.status == 200
        except Exception:
            return False
