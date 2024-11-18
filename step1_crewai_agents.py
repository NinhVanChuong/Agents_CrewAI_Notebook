import json 
import os 
os.environ["OPENAI_API_KEY"] = ""

from langchain_openai import ChatOpenAI
from crewai import Agent, Task, Crew

from pydantic import BaseModel
from typing import Union

llm = ChatOpenAI(model="gpt-4o-mini")

class Output1(BaseModel):
    explain: str
    example: str 

class Output2(BaseModel):
    explain: str
    example: str 
    keyword: str

ai_expert_agent = Agent(
    role = "AI expert",
    goal = "Answer user AI questions",
    backstory = """
        Bạn là một chuyên gia về AI. Bạn hãy lắng nghe câu hỏi của người dùng và trả lời thật chi tiết kèm ví dụ minh họa.
    """,
    llm=llm,
)

task = Task(
    description='Trả lời câu hỏi Machine Learning là gì?',
    expected_output='Một câu trả lời phù hợp với câu hỏi của người dùng, lưu ý đưa ra những ví dụ minh họa rõ ràng cho sự giải thích của mình',
    agent=ai_expert_agent,
    output_json = Output2 # set json format
)

# Execute the crew
crew = Crew(
    agents=[ai_expert_agent],
    tasks=[task],
    verbose=True
)

result = crew.kickoff()

print(result)