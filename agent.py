from langchain.agents import create_agent
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tool import web_search, scrape_url
from dotenv import load_dotenv

load_dotenv()

#model setup
llm = ChatMistralAI(model ="mistral-large-2512",temperature=0)

#first agent
def build_search_agent():
    return create_agent(model =llm, tools=[web_search])

#second agent
def build_reader_agent():
    return create_agent(model=llm, tools=[scrape_url])

#writer chain
writer_prompt = ChatPromptTemplate.from_messages([
    ("system","you are an expert reasearch writer.write clear, structured and insightful report."),
    ("human","""Write the detail reaserch report on the topic below.
    Topic:{topic}
    Reasearch geather:
    {research}

    Structure the report as:
    -Introduction
    -key finding(minimum 3 well explained points)
    -conclusion
    -source(list the url find in the research)

    Be detailed factional and proffesional."""),
]) 



writer_chain = writer_prompt | llm | StrOutputParser()

#critic_chain 

critic_prompt = ChatPromptTemplate.from_messages([
     ("system", "You are a sharp and constructive research critic. Be honest and specific."),
    ("human", """Review the research report below and evaluate it strictly.

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

One line verdict:
..."""),
])

critic_chain = critic_prompt | llm | StrOutputParser()