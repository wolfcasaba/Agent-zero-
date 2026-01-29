from python.helpers.api import ApiHandler, Request, Response
from python.helpers.ai_settings import save_ai_settings_api


class AiSettingsSet(ApiHandler):
    async def process(self, input: dict, request: Request) -> dict | Response:
        return save_ai_settings_api(input)
