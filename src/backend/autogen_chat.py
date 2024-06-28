from autogen import ConversableAgent
from user_proxy_webagent import UserProxyWebAgent
import asyncio
import os
from dotenv import load_dotenv
from prompts import PROMPT_DONA, PROMPT_RACHEL

load_dotenv()

rachel_config = {
    "config_list": [
        {
            "api_type": "mistral",
            "model": "mistral-large-latest",
            "api_key": os.environ.get("MISTRAL_API_KEY"),
            "temperature": 0.0,
        }
    ]
}

dona_config = {
    "config_list": [
        {
            "api_type": "mistral",
            "model": 'ft:open-mistral-7b:41dfebed:20240628:a9a7c195',
            "api_key": os.environ.get("MISTRAL_API_KEY"),
            "temperature": 0.0,
        }
    ]
}

class AutogenChat:
    def __init__(self, chat_id=None, websocket=None):
        self.websocket = websocket
        self.chat_id = chat_id
        self.client_sent_queue = asyncio.Queue()
        self.client_receive_queue = asyncio.Queue()

        self.agent_client = UserProxyWebAgent(
            "client",
            llm_config=False,
            code_execution_config=False,
            human_input_mode="ALWAYS",
        )

        self.agent_dona = ConversableAgent(
            "dona",
            llm_config=dona_config,
            human_input_mode="NEVER",
            system_message=PROMPT_DONA,
        )

        self.agent_rachel = ConversableAgent(
            "rachel",
            llm_config=rachel_config,
            human_input_mode="NEVER",
            system_message=PROMPT_RACHEL,
        )

        # add the queues to communicate
        self.agent_client.set_queues(self.client_sent_queue, self.client_receive_queue)

    async def clarify(self, message):
        await self.agent_client.a_initiate_chat(
            self.agent_dona, clear_history=True, message=message
        )

    async def research(self, message):
        await self.agent_client.a_initiate_chat(
            self.agent_rachel, clear_history=True, message=message
        )
