from python.helpers.api import ApiHandler, Request, Response
from python.helpers.ai_settings import get_ai_settings_api


class AiSettingsGet(ApiHandler):
    async def process(self, input: dict, request: Request) -> dict | Response:
        return get_ai_settings_api()

    @classmethod
    def get_methods(cls) -> list[str]:
        return ["GET", "POST"]
