from typing import Any
from langchain.callbacks.base import BaseCallbackHandler

class AgentCallbackHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs) -> Any:
        print('ON LLM START {}'.format(prompts[0]))
    def on_llm_end(self, response, **kwargs):
        print("LLM RESPONSE: {}".format(response.generations[0][0].text))
  