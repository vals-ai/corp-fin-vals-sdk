import os

provider_args = {
    "openai": {"api_key": os.getenv("OPENAI_API_KEY")},
    "google": {
        "api_key": os.getenv("GOOGLE_API_KEY"),
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
    },
    "anthropic": {
        "api_key": os.getenv("ANTHROPIC_API_KEY"),
        "base_url": "https://api.anthropic.com/v1/",
    },
    "together": {
        "api_key": os.getenv("TOGETHER_API_KEY"),
        "base_url": "https://api.together.xyz/v1/",
    },
    "fireworks": {
        "api_key": os.getenv("FIREWORKS_API_KEY"),
        "base_url": "https://api.fireworks.ai/v1/",
    },
    "mistralai": {
        "api_key": os.getenv("MISTRAL_API_KEY"),
        "base_url": "https://api.mistral.ai/v1/",
    },
    "grok": {
        "api_key": os.getenv("GROK_API_KEY"),
        "base_url": "https://api.x.ai/v1",
    },
    "cohere": {
        "api_key": os.getenv("COHERE_API_KEY"),
        "base_url": "https://api.cohere.ai/compatibility/v1/",
    },
}