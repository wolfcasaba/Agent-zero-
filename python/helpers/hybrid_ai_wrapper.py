"""
Hybrid AI Wrapper for Agent Zero
================================
Supports Claude CLI AND external API providers.

Providers:
- claude-cli: Claude CLI (Claude Max subscription, no API cost)
- claude-api: Anthropic API (paid)
- openai: OpenAI API (GPT-4, GPT-4o, etc.)
- ollama: Local models (Llama, Mistral, Qwen, etc.)
- openrouter: OpenRouter (many models via one API)
- groq: Groq API (fast inference)
- google: Google Gemini API

Usage:
    from python.helpers.hybrid_ai_wrapper import AIManager, AIProvider

    # Claude CLI (default)
    manager = AIManager(provider="claude-cli")

    # Or with API
    manager = AIManager(
        provider="openai",
        api_key="sk-...",
        model="gpt-4o"
    )

    response = await manager.chat(messages)
"""

import asyncio
import json
import os
from typing import AsyncIterator, List, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

import httpx


# =============================================================================
# Enums and Dataclasses
# =============================================================================

class AIProvider(Enum):
    """Supported AI providers"""
    CLAUDE_CLI = "claude-cli"
    CLAUDE_API = "claude-api"
    OPENAI = "openai"
    OLLAMA = "ollama"
    OPENROUTER = "openrouter"
    GROQ = "groq"
    GOOGLE = "google"
    CUSTOM = "custom"


@dataclass
class ProviderConfig:
    """Provider configuration"""
    provider: AIProvider
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    model: Optional[str] = None
    timeout: int = 300
    max_tokens: int = 4096
    temperature: float = 0.7
    extra_params: Dict = field(default_factory=dict)


@dataclass
class ChatMessage:
    """Chat message"""
    role: str
    content: str
    name: Optional[str] = None
    tool_calls: Optional[List[Dict]] = None
    tool_call_id: Optional[str] = None


@dataclass
class ChatResponse:
    """Chat response"""
    content: str
    model: str
    provider: str
    tool_calls: Optional[List[Dict]] = None
    finish_reason: str = "stop"
    usage: Optional[Dict] = None


# =============================================================================
# Default Model Names
# =============================================================================

DEFAULT_MODELS = {
    AIProvider.CLAUDE_CLI: "claude-cli",
    AIProvider.CLAUDE_API: "claude-sonnet-4-5-20250929",
    AIProvider.OPENAI: "gpt-4o",
    AIProvider.OLLAMA: "llama3.1:8b",
    AIProvider.OPENROUTER: "anthropic/claude-3.5-sonnet",
    AIProvider.GROQ: "llama-3.1-70b-versatile",
    AIProvider.GOOGLE: "gemini-1.5-pro",
}

DEFAULT_API_BASES = {
    AIProvider.CLAUDE_API: "https://api.anthropic.com",
    AIProvider.OPENAI: "https://api.openai.com/v1",
    AIProvider.OLLAMA: "http://localhost:11434",
    AIProvider.OPENROUTER: "https://openrouter.ai/api/v1",
    AIProvider.GROQ: "https://api.groq.com/openai/v1",
    AIProvider.GOOGLE: "https://generativelanguage.googleapis.com/v1beta",
}


# =============================================================================
# Base Provider Class
# =============================================================================

class BaseProvider(ABC):
    """Abstract provider class"""

    def __init__(self, config: ProviderConfig):
        self.config = config

    @abstractmethod
    async def chat(self, messages: List[Dict], **kwargs) -> ChatResponse:
        pass

    @abstractmethod
    async def stream(self, messages: List[Dict], **kwargs) -> AsyncIterator[str]:
        pass


# =============================================================================
# Claude CLI Provider
# =============================================================================

class ClaudeCLIProvider(BaseProvider):
    """
    Claude CLI provider - usable with Claude Max subscription.
    No API cost!
    """

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.working_dir = config.extra_params.get("working_dir", os.getcwd())
        self.allowed_tools = config.extra_params.get(
            "allowed_tools",
            ["Read", "Write", "Edit", "Bash"]
        )

    def _messages_to_prompt(self, messages: List[Dict]) -> str:
        parts = []
        system_prompt = None

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                system_prompt = content
            elif role == "assistant":
                parts.append(f"[Assistant]\n{content}")
            elif role == "user":
                parts.append(content)
            elif role == "tool":
                parts.append(f"[Tool Result: {msg.get('name', 'unknown')}]\n{content}")

        prompt = "\n\n".join(parts)

        if system_prompt:
            prompt = f"<system>\n{system_prompt}\n</system>\n\n{prompt}"

        return prompt

    def _build_command(self, prompt: str, stream: bool = False) -> List[str]:
        cmd = ["claude", "-p", prompt]

        if self.working_dir:
            cmd.extend(["--cwd", self.working_dir])

        if self.allowed_tools:
            cmd.extend(["--allowedTools", ",".join(self.allowed_tools)])

        if stream:
            cmd.append("--stream")

        return cmd

    async def chat(self, messages: List[Dict], **kwargs) -> ChatResponse:
        prompt = self._messages_to_prompt(messages)
        cmd = self._build_command(prompt)

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.working_dir
        )

        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=self.config.timeout
        )

        if process.returncode != 0:
            raise RuntimeError(f"Claude CLI error: {stderr.decode()}")

        return ChatResponse(
            content=stdout.decode().strip(),
            model="claude-cli",
            provider="claude-cli"
        )

    async def stream(self, messages: List[Dict], **kwargs) -> AsyncIterator[str]:
        prompt = self._messages_to_prompt(messages)
        cmd = self._build_command(prompt, stream=True)

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.working_dir
        )

        async for line in process.stdout:
            chunk = line.decode().strip()
            if chunk:
                yield chunk

        await process.wait()


# =============================================================================
# OpenAI-Compatible Provider (OpenAI, Groq, OpenRouter, Ollama)
# =============================================================================

class OpenAICompatibleProvider(BaseProvider):
    """
    OpenAI-compatible API provider.
    Works with: OpenAI, Groq, OpenRouter, Ollama, LM Studio, etc.
    """

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.api_base = config.api_base or DEFAULT_API_BASES.get(config.provider, "")
        self.model = config.model or DEFAULT_MODELS.get(config.provider, "gpt-4o")

        self.headers = {
            "Content-Type": "application/json",
        }

        if config.api_key:
            self.headers["Authorization"] = f"Bearer {config.api_key}"

        if config.provider == AIProvider.OPENROUTER:
            self.headers["HTTP-Referer"] = "https://agent-zero.local"
            self.headers["X-Title"] = "Agent Zero Claude"

    async def chat(self, messages: List[Dict], **kwargs) -> ChatResponse:
        url = f"{self.api_base}/chat/completions"

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
        }

        headers = self.headers.copy()
        if self.config.provider == AIProvider.OLLAMA:
            headers.pop("Authorization", None)

        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

        choice = data["choices"][0]

        return ChatResponse(
            content=choice["message"]["content"],
            model=data.get("model", self.model),
            provider=self.config.provider.value,
            tool_calls=choice["message"].get("tool_calls"),
            finish_reason=choice.get("finish_reason", "stop"),
            usage=data.get("usage")
        )

    async def stream(self, messages: List[Dict], **kwargs) -> AsyncIterator[str]:
        url = f"{self.api_base}/chat/completions"

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "stream": True,
        }

        headers = self.headers.copy()
        if self.config.provider == AIProvider.OLLAMA:
            headers.pop("Authorization", None)

        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            async with client.stream("POST", url, json=payload, headers=headers) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            delta = data["choices"][0].get("delta", {})
                            if "content" in delta:
                                yield delta["content"]
                        except json.JSONDecodeError:
                            continue


# =============================================================================
# Anthropic API Provider
# =============================================================================

class AnthropicProvider(BaseProvider):
    """
    Anthropic API provider (claude-api).
    Paid, uses the Anthropic API directly.
    """

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.api_base = config.api_base or "https://api.anthropic.com"
        self.model = config.model or "claude-sonnet-4-5-20250929"

        if not config.api_key:
            raise ValueError("Anthropic API key required")

        self.headers = {
            "Content-Type": "application/json",
            "x-api-key": config.api_key,
            "anthropic-version": "2023-06-01",
        }

    def _convert_messages(self, messages: List[Dict]) -> tuple:
        system_prompt = None
        converted = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                system_prompt = content
            elif role in ["user", "assistant"]:
                converted.append({"role": role, "content": content})

        return system_prompt, converted

    async def chat(self, messages: List[Dict], **kwargs) -> ChatResponse:
        url = f"{self.api_base}/v1/messages"

        system_prompt, converted_messages = self._convert_messages(messages)

        payload = {
            "model": self.model,
            "messages": converted_messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
        }

        if system_prompt:
            payload["system"] = system_prompt

        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            response = await client.post(url, json=payload, headers=self.headers)
            response.raise_for_status()
            data = response.json()

        content = ""
        for block in data.get("content", []):
            if block.get("type") == "text":
                content += block.get("text", "")

        return ChatResponse(
            content=content,
            model=data.get("model", self.model),
            provider="claude-api",
            finish_reason=data.get("stop_reason", "stop"),
            usage=data.get("usage")
        )

    async def stream(self, messages: List[Dict], **kwargs) -> AsyncIterator[str]:
        url = f"{self.api_base}/v1/messages"

        system_prompt, converted_messages = self._convert_messages(messages)

        payload = {
            "model": self.model,
            "messages": converted_messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "stream": True,
        }

        if system_prompt:
            payload["system"] = system_prompt

        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            async with client.stream("POST", url, json=payload, headers=self.headers) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])
                            if data.get("type") == "content_block_delta":
                                delta = data.get("delta", {})
                                if delta.get("type") == "text_delta":
                                    yield delta.get("text", "")
                        except json.JSONDecodeError:
                            continue


# =============================================================================
# Google Gemini Provider
# =============================================================================

class GoogleGeminiProvider(BaseProvider):
    """Google Gemini API provider"""

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.model = config.model or "gemini-1.5-pro"

        if not config.api_key:
            raise ValueError("Google API key required")

        self.api_key = config.api_key
        self.api_base = "https://generativelanguage.googleapis.com/v1beta"

    def _convert_messages(self, messages: List[Dict]) -> tuple:
        system_instruction = None
        contents = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                system_instruction = content
            elif role == "user":
                contents.append({"role": "user", "parts": [{"text": content}]})
            elif role == "assistant":
                contents.append({"role": "model", "parts": [{"text": content}]})

        return system_instruction, contents

    async def chat(self, messages: List[Dict], **kwargs) -> ChatResponse:
        url = f"{self.api_base}/models/{self.model}:generateContent?key={self.api_key}"

        system_instruction, contents = self._convert_messages(messages)

        payload = {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": kwargs.get("max_tokens", self.config.max_tokens),
                "temperature": kwargs.get("temperature", self.config.temperature),
            }
        }

        if system_instruction:
            payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}

        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()

        content = ""
        candidates = data.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            content = "".join(p.get("text", "") for p in parts)

        return ChatResponse(
            content=content,
            model=self.model,
            provider="google",
            usage=data.get("usageMetadata")
        )

    async def stream(self, messages: List[Dict], **kwargs) -> AsyncIterator[str]:
        url = f"{self.api_base}/models/{self.model}:streamGenerateContent?key={self.api_key}&alt=sse"

        system_instruction, contents = self._convert_messages(messages)

        payload = {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": kwargs.get("max_tokens", self.config.max_tokens),
                "temperature": kwargs.get("temperature", self.config.temperature),
            }
        }

        if system_instruction:
            payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}

        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            async with client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])
                            candidates = data.get("candidates", [])
                            if candidates:
                                parts = candidates[0].get("content", {}).get("parts", [])
                                for part in parts:
                                    if "text" in part:
                                        yield part["text"]
                        except json.JSONDecodeError:
                            continue


# =============================================================================
# AI Manager - Main Interface
# =============================================================================

class AIManager:
    """
    Main AI manager class.
    Manages all providers with a unified interface.

    Usage:
        # Claude CLI (default, free with Claude Max)
        manager = AIManager(provider="claude-cli")

        # OpenAI API
        manager = AIManager(
            provider="openai",
            api_key="sk-...",
            model="gpt-4o"
        )

        # Ollama (local)
        manager = AIManager(
            provider="ollama",
            api_base="http://localhost:11434",
            model="llama3.1:8b"
        )

        # Chat
        messages = [{"role": "user", "content": "Hello!"}]
        response = await manager.chat(messages)
        print(response.content)

        # Streaming
        async for chunk in manager.stream(messages):
            print(chunk, end="", flush=True)
    """

    PROVIDER_MAP = {
        AIProvider.CLAUDE_CLI: ClaudeCLIProvider,
        AIProvider.CLAUDE_API: AnthropicProvider,
        AIProvider.OPENAI: OpenAICompatibleProvider,
        AIProvider.OLLAMA: OpenAICompatibleProvider,
        AIProvider.OPENROUTER: OpenAICompatibleProvider,
        AIProvider.GROQ: OpenAICompatibleProvider,
        AIProvider.GOOGLE: GoogleGeminiProvider,
        AIProvider.CUSTOM: OpenAICompatibleProvider,
    }

    def __init__(
        self,
        provider: str = "claude-cli",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: int = 300,
        **extra_params,
    ):
        self.provider_enum = AIProvider(provider)

        self.config = ProviderConfig(
            provider=self.provider_enum,
            api_key=api_key,
            api_base=api_base,
            model=model or DEFAULT_MODELS.get(self.provider_enum),
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            extra_params=extra_params,
        )

        provider_class = self.PROVIDER_MAP.get(self.provider_enum)
        if not provider_class:
            raise ValueError(f"Unsupported provider: {provider}")

        self._provider = provider_class(self.config)

    async def chat(self, messages: List[Dict], **kwargs) -> ChatResponse:
        """Send chat messages and get a response"""
        return await self._provider.chat(messages, **kwargs)

    async def stream(self, messages: List[Dict], **kwargs) -> AsyncIterator[str]:
        """Stream chat response chunks"""
        async for chunk in self._provider.stream(messages, **kwargs):
            yield chunk

    def get_provider_info(self) -> Dict[str, Any]:
        """Return current provider info"""
        return {
            "provider": self.provider_enum.value,
            "model": self.config.model,
            "api_base": self.config.api_base,
            "has_api_key": bool(self.config.api_key),
        }
