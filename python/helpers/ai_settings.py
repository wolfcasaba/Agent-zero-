"""
AI Provider Settings for Agent Zero
====================================
Web UI-managed provider settings.

Settings file: tmp/ai_settings.json
"""

import os
import json
from typing import Dict, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path


# =============================================================================
# Settings Dataclasses
# =============================================================================

@dataclass
class ProviderSettings:
    """Settings for a single provider"""
    enabled: bool = False
    api_key: str = ""
    api_base: str = ""
    model: str = ""
    temperature: float = 0.7
    max_tokens: int = 4096


@dataclass
class AISettings:
    """All AI settings"""
    active_provider: str = "claude-cli"

    claude_cli: ProviderSettings = None
    claude_api: ProviderSettings = None
    openai: ProviderSettings = None
    ollama: ProviderSettings = None
    openrouter: ProviderSettings = None
    groq: ProviderSettings = None
    google: ProviderSettings = None
    custom: ProviderSettings = None

    def __post_init__(self):
        if self.claude_cli is None:
            self.claude_cli = ProviderSettings(enabled=True, model="claude-cli")
        if self.claude_api is None:
            self.claude_api = ProviderSettings(model="claude-sonnet-4-5-20250929")
        if self.openai is None:
            self.openai = ProviderSettings(model="gpt-4o")
        if self.ollama is None:
            self.ollama = ProviderSettings(
                api_base="http://localhost:11434",
                model="llama3.1:8b"
            )
        if self.openrouter is None:
            self.openrouter = ProviderSettings(model="anthropic/claude-3.5-sonnet")
        if self.groq is None:
            self.groq = ProviderSettings(model="llama-3.1-70b-versatile")
        if self.google is None:
            self.google = ProviderSettings(model="gemini-1.5-pro")
        if self.custom is None:
            self.custom = ProviderSettings()


# =============================================================================
# Settings Manager
# =============================================================================

class AISettingsManager:
    """
    AI settings manager.
    Loads/saves settings from tmp/ai_settings.json.
    """

    DEFAULT_PATH = "tmp/ai_settings.json"

    def __init__(self, settings_path: str = None):
        self.settings_path = Path(settings_path or self.DEFAULT_PATH)
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> AISettings:
        if not self.settings_path.exists():
            return AISettings()

        try:
            with open(self.settings_path, "r") as f:
                data = json.load(f)
            return self._dict_to_settings(data)
        except (json.JSONDecodeError, KeyError, TypeError):
            return AISettings()

    def save(self, settings: AISettings) -> None:
        data = self._settings_to_dict(settings)
        with open(self.settings_path, "w") as f:
            json.dump(data, f, indent=2)

    def _settings_to_dict(self, settings: AISettings) -> Dict:
        return {
            "active_provider": settings.active_provider,
            "providers": {
                "claude-cli": asdict(settings.claude_cli),
                "claude-api": asdict(settings.claude_api),
                "openai": asdict(settings.openai),
                "ollama": asdict(settings.ollama),
                "openrouter": asdict(settings.openrouter),
                "groq": asdict(settings.groq),
                "google": asdict(settings.google),
                "custom": asdict(settings.custom),
            }
        }

    def _dict_to_settings(self, data: Dict) -> AISettings:
        providers = data.get("providers", {})

        def _make_provider(name: str) -> Optional[ProviderSettings]:
            p = providers.get(name)
            if p and isinstance(p, dict):
                return ProviderSettings(**{
                    k: v for k, v in p.items()
                    if k in ProviderSettings.__dataclass_fields__
                })
            return None

        return AISettings(
            active_provider=data.get("active_provider", "claude-cli"),
            claude_cli=_make_provider("claude-cli"),
            claude_api=_make_provider("claude-api"),
            openai=_make_provider("openai"),
            ollama=_make_provider("ollama"),
            openrouter=_make_provider("openrouter"),
            groq=_make_provider("groq"),
            google=_make_provider("google"),
            custom=_make_provider("custom"),
        )

    def get_active_provider_config(self) -> Dict:
        """Returns the active provider's configuration"""
        settings = self.load()
        provider_name = settings.active_provider

        provider_map = {
            "claude-cli": settings.claude_cli,
            "claude-api": settings.claude_api,
            "openai": settings.openai,
            "ollama": settings.ollama,
            "openrouter": settings.openrouter,
            "groq": settings.groq,
            "google": settings.google,
            "custom": settings.custom,
        }

        provider_settings = provider_map.get(provider_name, settings.claude_cli)

        return {
            "provider": provider_name,
            "api_key": provider_settings.api_key or self._get_env_key(provider_name),
            "api_base": provider_settings.api_base,
            "model": provider_settings.model,
            "temperature": provider_settings.temperature,
            "max_tokens": provider_settings.max_tokens,
        }

    def _get_env_key(self, provider: str) -> str:
        env_map = {
            "claude-api": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
            "groq": "GROQ_API_KEY",
            "google": "GOOGLE_API_KEY",
        }
        return os.environ.get(env_map.get(provider, ""), "")


# =============================================================================
# API Helper Functions
# =============================================================================

def get_ai_settings_api() -> Dict:
    """GET /api/ai-settings - Get settings (masks API keys)"""
    manager = AISettingsManager()
    settings = manager.load()

    data = manager._settings_to_dict(settings)
    for provider in data["providers"].values():
        if provider.get("api_key"):
            key = provider["api_key"]
            provider["api_key"] = f"{key[:8]}...{key[-4:]}" if len(key) > 12 else "***"

    return data


def save_ai_settings_api(data: Dict) -> Dict:
    """POST /api/ai-settings - Save settings"""
    manager = AISettingsManager()
    current = manager.load()

    if "active_provider" in data:
        current.active_provider = data["active_provider"]

    if "providers" in data:
        for provider_name, provider_data in data["providers"].items():
            attr_name = provider_name.replace("-", "_")
            current_provider = getattr(current, attr_name, None)
            if current_provider:
                if "api_key" in provider_data:
                    if "..." in provider_data["api_key"] or provider_data["api_key"] == "***":
                        provider_data.pop("api_key")

                for key, value in provider_data.items():
                    if hasattr(current_provider, key):
                        setattr(current_provider, key, value)

    manager.save(current)
    return {"status": "ok"}


def get_available_models_api(provider: str) -> Dict:
    """GET /api/ai-models/{provider} - Get available models"""
    models = {
        "claude-cli": ["claude-cli"],
        "claude-api": [
            "claude-sonnet-4-5-20250929",
            "claude-opus-4-5-20251101",
            "claude-haiku-4-5-20251001",
        ],
        "openai": [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-4",
            "gpt-3.5-turbo",
        ],
        "ollama": [],
        "openrouter": [
            "anthropic/claude-3.5-sonnet",
            "openai/gpt-4o",
            "google/gemini-pro-1.5",
            "meta-llama/llama-3.1-405b-instruct",
        ],
        "groq": [
            "llama-3.1-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
        ],
        "google": [
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "gemini-1.0-pro",
        ],
    }

    # Dynamically fetch Ollama models
    if provider == "ollama":
        try:
            import httpx
            response = httpx.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                models["ollama"] = [m["name"] for m in data.get("models", [])]
        except Exception:
            pass

    return {"models": models.get(provider, [])}


# =============================================================================
# Integration with hybrid_ai_wrapper
# =============================================================================

def create_ai_manager_from_settings():
    """
    Creates an AIManager from saved settings.

    Usage:
        from python.helpers.ai_settings import create_ai_manager_from_settings

        manager = create_ai_manager_from_settings()
        response = await manager.chat(messages)
    """
    from python.helpers.hybrid_ai_wrapper import AIManager

    settings_manager = AISettingsManager()
    config = settings_manager.get_active_provider_config()

    return AIManager(
        provider=config["provider"],
        api_key=config["api_key"] if config["api_key"] else None,
        api_base=config["api_base"] if config["api_base"] else None,
        model=config["model"] if config["model"] else None,
        temperature=config["temperature"],
        max_tokens=config["max_tokens"],
    )


# =============================================================================
# Web UI HTML Fragment
# =============================================================================

SETTINGS_UI_HTML = """
<!-- AI Provider Settings Panel -->
<div class="ai-settings-panel" x-data="aiSettingsStore">
    <h3>AI Provider</h3>

    <!-- Provider Selection -->
    <div class="form-group">
        <label>Active Provider</label>
        <select x-model="activeProvider" @change="switchProvider()">
            <option value="claude-cli">Claude CLI (Max subscription)</option>
            <option value="claude-api">Claude API (Anthropic)</option>
            <option value="openai">OpenAI (GPT-4)</option>
            <option value="ollama">Ollama (Local)</option>
            <option value="openrouter">OpenRouter</option>
            <option value="groq">Groq</option>
            <option value="google">Google Gemini</option>
            <option value="custom">Custom Endpoint</option>
        </select>
    </div>

    <!-- Claude CLI -->
    <div x-show="activeProvider === 'claude-cli'" class="provider-settings">
        <div class="info-box success">
            Claude CLI configured. No API cost with Claude Max subscription.
        </div>
    </div>

    <!-- API Providers -->
    <template x-if="activeProvider !== 'claude-cli'">
        <div class="provider-settings">
            <!-- API Key -->
            <div class="form-group" x-show="needsApiKey()">
                <label>API Key</label>
                <input type="password"
                       x-model="providers[activeProvider].api_key"
                       placeholder="sk-... or AIza...">
            </div>

            <!-- API Base (Ollama, Custom) -->
            <div class="form-group" x-show="activeProvider === 'ollama' || activeProvider === 'custom'">
                <label>API Base URL</label>
                <input type="text"
                       x-model="providers[activeProvider].api_base"
                       placeholder="http://localhost:11434">
            </div>

            <!-- Model -->
            <div class="form-group">
                <label>Model</label>
                <select x-model="providers[activeProvider].model">
                    <template x-for="model in availableModels">
                        <option :value="model" x-text="model"></option>
                    </template>
                </select>
            </div>

            <!-- Temperature -->
            <div class="form-group">
                <label>Temperature: <span x-text="providers[activeProvider].temperature"></span></label>
                <input type="range" min="0" max="1" step="0.1"
                       x-model="providers[activeProvider].temperature">
            </div>

            <!-- Max Tokens -->
            <div class="form-group">
                <label>Max Tokens</label>
                <input type="number"
                       x-model="providers[activeProvider].max_tokens"
                       min="100" max="128000">
            </div>
        </div>
    </template>

    <!-- Save Button -->
    <button @click="saveSettings()" class="btn-primary">
        Save
    </button>

    <!-- Status -->
    <div x-show="statusMessage" class="status-message" x-text="statusMessage"></div>
</div>

<script>
document.addEventListener('alpine:init', () => {
    Alpine.store('aiSettingsStore', {
        activeProvider: 'claude-cli',
        providers: {},
        availableModels: [],
        statusMessage: '',

        async init() {
            const response = await fetch('/ai_settings_get');
            const data = await response.json();
            this.activeProvider = data.active_provider;
            this.providers = data.providers;
            await this.loadModels();
        },

        needsApiKey() {
            return !['claude-cli', 'ollama'].includes(this.activeProvider);
        },

        async switchProvider() {
            await this.loadModels();
        },

        async loadModels() {
            const response = await fetch('/ai_models_get', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({provider: this.activeProvider})
            });
            const data = await response.json();
            this.availableModels = data.models;
        },

        async saveSettings() {
            const response = await fetch('/ai_settings_set', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    active_provider: this.activeProvider,
                    providers: this.providers
                })
            });

            if (response.ok) {
                this.statusMessage = 'Settings saved!';
            } else {
                this.statusMessage = 'Error saving settings!';
            }

            setTimeout(() => this.statusMessage = '', 3000);
        }
    });
});
</script>

<style>
.ai-settings-panel {
    padding: 1rem;
    background: var(--bg-secondary);
    border-radius: 8px;
}
.form-group {
    margin-bottom: 1rem;
}
.form-group label {
    display: block;
    margin-bottom: 0.5rem;
    font-weight: 500;
}
.form-group input,
.form-group select {
    width: 100%;
    padding: 0.5rem;
    border: 1px solid var(--border-color);
    border-radius: 4px;
    background: var(--bg-primary);
}
.info-box.success {
    padding: 1rem;
    background: rgba(0, 255, 0, 0.1);
    border-radius: 4px;
    border-left: 3px solid green;
}
.btn-primary {
    padding: 0.75rem 1.5rem;
    background: var(--accent-color);
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
}
.status-message {
    margin-top: 1rem;
    padding: 0.5rem;
    border-radius: 4px;
}
</style>
"""
