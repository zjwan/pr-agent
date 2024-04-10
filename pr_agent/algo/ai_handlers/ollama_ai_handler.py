try:
    from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
    from langchain.schema import SystemMessage, HumanMessage
except: # we don't enforce langchain as a dependency, so if it's not installed, just move on
    pass

from pr_agent.algo.ai_handlers.base_ai_handler import BaseAiHandler
from pr_agent.config_loader import get_settings
from pr_agent.log import get_logger

from openai.error import APIError, RateLimitError, Timeout, TryAgain
from retry import retry
import functools

OPENAI_RETRIES = 5

class OllamaAIHandler(BaseAiHandler):
    def __init__(self):
        # Initialize OpenAIHandler specific attributes here
        super().__init__()
    
    @property
    def chat(self):
        pass

    @property
    def deployment_id(self):
        """
        Returns the deployment ID for the OpenAI API.
        """
        return None

    @retry(exceptions=(APIError, Timeout, TryAgain, AttributeError, RateLimitError),
           tries=OPENAI_RETRIES, delay=2, backoff=2, jitter=(1, 3))
    async def chat_completion(self, model: str, system: str, user: str, temperature: float = 0.2):
        try:
            messages=[SystemMessage(content=system), HumanMessage(content=user)]
            
            # get a chat completion from the formatted messages
            resp = self.chat(messages, model=model, temperature=temperature)
            finish_reason="completed"
            return resp.content, finish_reason
        
        except (Exception) as e:
            get_logger().error("Unknown error during OpenAI inference: ", e)
            raise e
