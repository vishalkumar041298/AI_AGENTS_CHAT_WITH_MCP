from typing import Union
from dotenv import load_dotenv
from langchain.agents import tool
from langchain.prompts import PromptTemplate
from langchain.tools.render import render_text_description
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from pydantic import BaseModel
from langchain_core.agents import AgentAction, AgentFinish
from langchain.tools import Tool
from langchain.agents.format_scratchpad import format_log_to_str
from callbacks import AgentCallbackHandler

load_dotenv()


@tool
def get_text_len(text: str) -> int:
    '''
        Returns the length of a text by characters
    '''
    text = text.strip("'\n").strip('"') # stripping away non alphabetic charecters just in case
    return len(text)


def find_tool_by_name(tools: list[Tool], tool_name: str) -> Tool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool with name {tool_name} not found")


class Result(BaseModel):
    text: str
    length: int
    
    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "length": self.length
        }

if __name__ == '__main__':
    print("Hello ReAct LangChain!")
    tools = [get_text_len]
    template = """
        Answer the following questions as best you can. You have access to the following tools:

        {tools}

        Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        Begin!

        Question: {input}
        Thought: {agent_scratchpad}
        
    """
    parser = PydanticOutputParser(pydantic_object=Result)
    format_instructions = parser.get_format_instructions()
    # prompt_template = PromptTemplate.from_template(template=template,partial_variables={'format_instructions': format_instructions}).partial(tools=render_text_description(tools), tool_names=", ".join([t.name for t in tools]))
    prompt_template = PromptTemplate.from_template(template=template).partial(tools=render_text_description(tools), tool_names=", ".join([t.name for t in tools]))
    
    llm = ChatOpenAI(temperature=0, model='gpt-4o-mini', stop=["\nObservation"], callbacks=[AgentCallbackHandler()])
    
    # chain = prompt_template | llm | parser  
    # res = chain.invoke(input="What is the length 'CATTT'?")
    intermediate_steps = []
    agent = (
       {'input': lambda x: x['input'], 'agent_scratchpad': lambda x: format_log_to_str(x['agent_scratchpad'])}
       | prompt_template
       | llm
       | ReActSingleInputOutputParser()  
    )
    res = ''
    
    while not isinstance(res, AgentFinish):
        res: Union[AgentAction, AgentFinish] = agent.invoke(input={'input': "What is the length 'CATTT'?", 'agent_scratchpad': intermediate_steps})

        if isinstance(res, AgentAction):
            tool_name = res.tool
            tool_to_use = find_tool_by_name(tools, tool_name)
            tool_input = res.tool_input
            
            observation = tool_to_use.func(str(tool_input))
            print(observation)
            intermediate_steps.append((res, observation))
            res: Union[AgentAction, AgentFinish] = agent.invoke(
                input={'input': "What is the length 'CATTT'?",
                    'agent_scratchpad': intermediate_steps}
            )
            print(res)

        

    