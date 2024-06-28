from autogen import ConversableAgent
from user_proxy_webagent import UserProxyWebAgent
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

llm_config = {
    "config_list": [
        {
            "api_type": "mistral",
            "model": "mistral-large-latest",
            "api_key": os.environ.get("MISTRAL_API_KEY"),
            "temperature": 0.0,
        }
    ]
}


#############################################################################################
# this is where you put your Autogen logic, here I have a simple 2 agents with a function call
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
            llm_config=llm_config,
            human_input_mode="NEVER",
            system_message="""
            You are Dona a swiss legal assistant, you need to help the client clarify their legal issue according to the swiss law.
            Summarize concisely the situation provided using appropriate legal terms.
            Start your summary by saying "Case Summary:".
            Be concise!
            Do not reference specific legal article numbers.
            Finish your reply by asking the client to either say "exit" if the summary is good or to provide any additional informations or corrections to edit the summary.
            If relevant and appropriate, suggest few examples of key informations that could be added to the summary.
            """,
        )

        self.agent_rachel = ConversableAgent(
            "rachel",
            llm_config=llm_config,
            human_input_mode="NEVER",
            system_message="""
            You are Rachel a swiss para-legal researcher, you need to help a client clarify a legal situation under Swiss law.
            You are given:
            * a case summary of the client situation by Dona your legal assistant.
            * a set of retrieved relevant Swiss law articles corresponding to the situation.
            Use the case summary and the relevant Swiss law articles to provide your expert opinion on the client situation.
            Always quote the relevant Swiss law articles in your reply.
            """,
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

    # MOCH Function call
    # def search_db(self, order_number=None, customer_number=None):
    #    return "Order status: delivered TERMINATE"
