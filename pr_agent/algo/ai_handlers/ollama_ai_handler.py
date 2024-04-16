import os
from ollama import Client

from pr_agent.algo.ai_handlers.base_ai_handler import BaseAiHandler
from pr_agent.config_loader import get_settings
from pr_agent.log import get_logger

from retry import retry
import functools

OPENAI_RETRIES = 5

class OllamaAIHandler(BaseAiHandler):
    def __init__(self):
        # Initialize OpenAIHandler specific attributes here
        super().__init__()
        # although env:OLLAMA_HOST is supported in ollama library, we'd prefer to get host name explicit for better readability
        self.host = get_settings().get("OLLAMA.HOST", os.getenv('OLLAMA_HOST'))
        assert self.host is not None
        self.client = Client(host=self.host)
    
    @property
    def chat(self):
        pass

    @property
    def deployment_id(self):
        """
        Returns the deployment ID for the OpenAI API.
        """
        return None

    async def chat_completion(self, system: str, user: str, model: str = "llama3", temperature: float = 0.2):
        try:
            response = self.client.chat(
                model=model,
                messages=[
                    {
                        'role': 'system',
                        'content': system,
                    },
                    {
                        'role': 'user',
                        'content': user,
                    },
                ],
                options={
                    "temperature": temperature
                }
            )
            get_logger().info(response['message']['content'])

            finish_reason="completed"
            return response['message']['content'], finish_reason
        
        except (Exception) as e:
            get_logger().error("Unknown error during ollama inference: ", e)
            raise e
