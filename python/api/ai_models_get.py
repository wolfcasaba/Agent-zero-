from python.helpers.api import ApiHandler, Request, Response
from python.helpers.ai_settings import get_available_models_api


class AiModelsGet(ApiHandler):
    async def process(self, input: dict, request: Request) -> dict | Response:
        provider = input.get("provider", "claude-cli")
        return get_available_models_api(provider)

    @classmethod
    def get_methods(cls) -> list[str]:
        return ["GET", "POST"]
