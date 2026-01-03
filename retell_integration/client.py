"""
Retell API Client for Palli Sahayak Voice AI Helpline

This module provides the RetellClient class for interacting with the Retell.AI API.
It handles agent creation, call management, and phone number configuration.

Documentation: https://docs.retellai.com/api-references
"""

import os
import logging
import aiohttp
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from .config import RETELL_API_BASE, RetellAgentConfig

logger = logging.getLogger(__name__)


@dataclass
class RetellCallResult:
    """Result of a Retell API call."""
    success: bool
    call_id: Optional[str] = None
    agent_id: Optional[str] = None
    error: Optional[str] = None
    data: Optional[Dict[str, Any]] = None


class RetellClient:
    """
    Client for Retell.AI API.

    Handles:
    - Agent creation and management
    - Call initiation (outbound)
    - Phone number management
    - LLM configuration

    Usage:
        client = RetellClient()

        # Create agent with custom LLM
        config = get_palli_sahayak_retell_config(...)
        result = await client.create_agent(config)

        # Initiate outbound call
        result = await client.create_phone_call(
            from_number="+1234567890",
            to_number="+919876543210",
            agent_id="agent-123"
        )
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Retell client.

        Args:
            api_key: Retell API key. If not provided, reads from RETELL_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("RETELL_API_KEY")
        if not self.api_key:
            logger.warning("Retell API key not configured - set RETELL_API_KEY env var")

        self.base_url = RETELL_API_BASE
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def is_available(self) -> bool:
        """Check if Retell client is configured with API key."""
        return bool(self.api_key)

    async def create_agent(self, config: RetellAgentConfig) -> RetellCallResult:
        """
        Create a new Retell agent with Custom LLM.

        Args:
            config: RetellAgentConfig with LLM WebSocket URL

        Returns:
            RetellCallResult with agent_id if successful
        """
        if not self.is_available():
            return RetellCallResult(success=False, error="Retell API key not configured")

        try:
            payload = config.to_dict()

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/create-agent",
                    headers=self.headers,
                    json=payload
                ) as response:
                    data = await response.json()

                    if response.status in (200, 201):
                        agent_id = data.get("agent_id")
                        logger.info(f"Created Retell agent: {agent_id}")
                        return RetellCallResult(
                            success=True,
                            agent_id=agent_id,
                            data=data
                        )
                    else:
                        error_msg = data.get("message", data.get("error", f"HTTP {response.status}"))
                        logger.error(f"Failed to create Retell agent: {error_msg}")
                        return RetellCallResult(success=False, error=error_msg)

        except aiohttp.ClientError as e:
            logger.error(f"Network error creating Retell agent: {e}")
            return RetellCallResult(success=False, error=f"Network error: {e}")
        except Exception as e:
            logger.error(f"Failed to create Retell agent: {e}")
            return RetellCallResult(success=False, error=str(e))

    async def get_agent(self, agent_id: str) -> RetellCallResult:
        """
        Get agent details.

        Args:
            agent_id: ID of the agent to retrieve

        Returns:
            RetellCallResult with agent data
        """
        if not self.is_available():
            return RetellCallResult(success=False, error="Retell API key not configured")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/get-agent/{agent_id}",
                    headers=self.headers
                ) as response:
                    data = await response.json()

                    if response.status == 200:
                        return RetellCallResult(
                            success=True,
                            agent_id=agent_id,
                            data=data
                        )
                    else:
                        return RetellCallResult(
                            success=False,
                            error=data.get("message", f"HTTP {response.status}")
                        )

        except Exception as e:
            logger.error(f"Failed to get Retell agent: {e}")
            return RetellCallResult(success=False, error=str(e))

    async def update_agent(self, agent_id: str, updates: Dict[str, Any]) -> RetellCallResult:
        """
        Update an existing agent.

        Args:
            agent_id: ID of the agent to update
            updates: Dictionary of fields to update

        Returns:
            RetellCallResult with updated agent data
        """
        if not self.is_available():
            return RetellCallResult(success=False, error="Retell API key not configured")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.patch(
                    f"{self.base_url}/update-agent/{agent_id}",
                    headers=self.headers,
                    json=updates
                ) as response:
                    data = await response.json()

                    if response.status == 200:
                        logger.info(f"Updated Retell agent: {agent_id}")
                        return RetellCallResult(
                            success=True,
                            agent_id=agent_id,
                            data=data
                        )
                    else:
                        return RetellCallResult(
                            success=False,
                            error=data.get("message", f"HTTP {response.status}")
                        )

        except Exception as e:
            logger.error(f"Failed to update Retell agent: {e}")
            return RetellCallResult(success=False, error=str(e))

    async def delete_agent(self, agent_id: str) -> RetellCallResult:
        """
        Delete an agent.

        Args:
            agent_id: ID of the agent to delete

        Returns:
            RetellCallResult
        """
        if not self.is_available():
            return RetellCallResult(success=False, error="Retell API key not configured")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.delete(
                    f"{self.base_url}/delete-agent/{agent_id}",
                    headers=self.headers
                ) as response:
                    if response.status in (200, 204):
                        logger.info(f"Deleted Retell agent: {agent_id}")
                        return RetellCallResult(success=True, agent_id=agent_id)
                    else:
                        data = await response.json()
                        return RetellCallResult(
                            success=False,
                            error=data.get("message", f"HTTP {response.status}")
                        )

        except Exception as e:
            logger.error(f"Failed to delete Retell agent: {e}")
            return RetellCallResult(success=False, error=str(e))

    async def list_agents(self) -> RetellCallResult:
        """
        List all agents.

        Returns:
            RetellCallResult with list of agents
        """
        if not self.is_available():
            return RetellCallResult(success=False, error="Retell API key not configured")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/list-agents",
                    headers=self.headers
                ) as response:
                    data = await response.json()

                    if response.status == 200:
                        return RetellCallResult(success=True, data=data)
                    else:
                        return RetellCallResult(
                            success=False,
                            error=data.get("message", f"HTTP {response.status}")
                        )

        except Exception as e:
            logger.error(f"Failed to list Retell agents: {e}")
            return RetellCallResult(success=False, error=str(e))

    async def create_phone_call(
        self,
        from_number: str,
        to_number: str,
        agent_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        retell_llm_dynamic_variables: Optional[Dict[str, Any]] = None
    ) -> RetellCallResult:
        """
        Create an outbound phone call.

        Args:
            from_number: Phone number to call from (E.164 format)
            to_number: Phone number to call (E.164 format)
            agent_id: ID of the agent to use
            metadata: Optional call metadata
            retell_llm_dynamic_variables: Optional dynamic variables for LLM

        Returns:
            RetellCallResult with call_id if successful
        """
        if not self.is_available():
            return RetellCallResult(success=False, error="Retell API key not configured")

        try:
            payload = {
                "from_number": from_number,
                "to_number": to_number,
                "agent_id": agent_id
            }

            if metadata:
                payload["metadata"] = metadata

            if retell_llm_dynamic_variables:
                payload["retell_llm_dynamic_variables"] = retell_llm_dynamic_variables

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/create-phone-call",
                    headers=self.headers,
                    json=payload
                ) as response:
                    data = await response.json()

                    if response.status in (200, 201):
                        call_id = data.get("call_id")
                        logger.info(f"Created Retell call: {call_id} to {to_number}")
                        return RetellCallResult(
                            success=True,
                            call_id=call_id,
                            agent_id=agent_id,
                            data=data
                        )
                    else:
                        error_msg = data.get("message", f"HTTP {response.status}")
                        logger.error(f"Failed to create call: {error_msg}")
                        return RetellCallResult(success=False, error=error_msg)

        except Exception as e:
            logger.error(f"Failed to create Retell call: {e}")
            return RetellCallResult(success=False, error=str(e))

    async def create_web_call(
        self,
        agent_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> RetellCallResult:
        """
        Create a web call (WebRTC).

        Args:
            agent_id: ID of the agent to use
            metadata: Optional call metadata

        Returns:
            RetellCallResult with call_id and access_token
        """
        if not self.is_available():
            return RetellCallResult(success=False, error="Retell API key not configured")

        try:
            payload = {"agent_id": agent_id}

            if metadata:
                payload["metadata"] = metadata

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/create-web-call",
                    headers=self.headers,
                    json=payload
                ) as response:
                    data = await response.json()

                    if response.status in (200, 201):
                        call_id = data.get("call_id")
                        logger.info(f"Created Retell web call: {call_id}")
                        return RetellCallResult(
                            success=True,
                            call_id=call_id,
                            agent_id=agent_id,
                            data=data
                        )
                    else:
                        return RetellCallResult(
                            success=False,
                            error=data.get("message", f"HTTP {response.status}")
                        )

        except Exception as e:
            logger.error(f"Failed to create web call: {e}")
            return RetellCallResult(success=False, error=str(e))

    async def get_call(self, call_id: str) -> RetellCallResult:
        """
        Get call details.

        Args:
            call_id: ID of the call

        Returns:
            RetellCallResult with call data
        """
        if not self.is_available():
            return RetellCallResult(success=False, error="Retell API key not configured")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/get-call/{call_id}",
                    headers=self.headers
                ) as response:
                    data = await response.json()

                    if response.status == 200:
                        return RetellCallResult(
                            success=True,
                            call_id=call_id,
                            data=data
                        )
                    else:
                        return RetellCallResult(
                            success=False,
                            error=data.get("message", f"HTTP {response.status}")
                        )

        except Exception as e:
            logger.error(f"Failed to get call: {e}")
            return RetellCallResult(success=False, error=str(e))

    async def list_calls(
        self,
        agent_id: Optional[str] = None,
        limit: int = 20,
        sort_order: str = "descending"
    ) -> RetellCallResult:
        """
        List calls, optionally filtered by agent.

        Args:
            agent_id: Optional agent ID to filter by
            limit: Maximum number of calls to return
            sort_order: "ascending" or "descending" by start time

        Returns:
            RetellCallResult with list of calls
        """
        if not self.is_available():
            return RetellCallResult(success=False, error="Retell API key not configured")

        try:
            payload = {
                "limit": limit,
                "sort_order": sort_order
            }
            if agent_id:
                payload["filter_criteria"] = [
                    {"member": "agent_id", "operator": "eq", "value": agent_id}
                ]

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/list-calls",
                    headers=self.headers,
                    json=payload
                ) as response:
                    data = await response.json()

                    if response.status == 200:
                        return RetellCallResult(success=True, data=data)
                    else:
                        return RetellCallResult(
                            success=False,
                            error=data.get("message", f"HTTP {response.status}")
                        )

        except Exception as e:
            logger.error(f"Failed to list calls: {e}")
            return RetellCallResult(success=False, error=str(e))

    async def health_check(self) -> bool:
        """
        Check if Retell API is accessible.

        Returns:
            True if API is healthy, False otherwise
        """
        if not self.is_available():
            return False

        try:
            result = await self.list_agents()
            return result.success
        except Exception:
            return False
