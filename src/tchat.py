from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import MessagesPlaceholder, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.memory import ConversationSummaryMemory, FileChatMessageHistory
from dotenv import load_dotenv

load_dotenv() #define your API key in a .env file

chat = ChatOpenAI()

memory = ConversationSummaryMemory(
    #chat_memory = FileChatMessageHistory("messages.json"),
    memory_key = "messages",
    return_messages = True,
    llm = chat,
) 

prompt = ChatPromptTemplate(
    messages = [
        MessagesPlaceholder(variable_name = "messages"),
        HumanMessagePromptTemplate.from_template("{content}")
    ],
    input_variables = ["content", "messages"],
    )

chain = LLMChain(
    llm = chat, 
    prompt = prompt,
    memory = memory,
    verbose = True,
    )
 
while True:
    content = input(">> ")

    result = chain({"content": content})

    print(result["text"])