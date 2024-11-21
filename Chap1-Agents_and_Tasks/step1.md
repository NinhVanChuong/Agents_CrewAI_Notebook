# Defining and Executing AI Tasks with Agents
The code defines a flow for executing AI-driven tasks using the GPT-4o-mini model. It starts by setting up the environment and initializing the ChatOpenAI instance as the language model. Structured output formats are defined using Pydantic models to ensure consistency. An AI expert agent is then created with a specific role, goal, and backstory to handle tasks related to AI questions. A task is defined, detailing the problem ("What is Machine Learning?"), the expected output, and assigning it to the agent. These components are grouped into a Crew, which organizes the agents and tasks for execution. The crew.kickoff() method triggers the agent to process the task using the GPT-4o-mini model and generate a structured response, which is then displayed in the console. This design is modular, allowing for easy extension with additional tasks or agents.
## 1. Environment Setup
```python
import json 
import os 
os.environ["OPENAI_API_KEY"] = ""
```
- import json: Used to work with JSON data, typically for managing input and output.
- import os: A library to interact with the operating system.
- os.environ["OPENAI_API_KEY"]: Sets the OPENAI_API_KEY environment variable for accessing OpenAI’s API. Replace "" with your actual API key.

## 2. Import libraries for LangChain and data structure definitions
```python
from langchain_openai import ChatOpenAI
from crewai import Agent, Task, Crew
from pydantic import BaseModel
from typing import Union
```
- ChatOpenAI: Represents the language model (LLM) used (here, gpt-4o-mini).
- Agent, Task, Crew: Components from the crewai library:
    - Agent: Represents an AI agent with a specific role and goal.
    - Task: Defines tasks to be accomplished, assigned to an Agent.
    - Crew: A group of Agents that carry out tasks together.
- BaseModel (Pydantic): Defines the structure for expected output, ensuring data consistency.
- Union: Allows defining combined data types, if needed.

## 3. Initialize the Language Model
```
llm = ChatOpenAI(model="gpt-4o-mini")
```
- ChatOpenAI: Initializes the language model (gpt-4o-mini) to be used for processing.

## 4. Define output structure
```python
class Output1(BaseModel):
    explain: str
    example: str 

class Output2(BaseModel):
    explain: str
    example: str 
    keyword: str
```
**Output1** and **Output2**: Classes describing the expected structure of task output.
- explain: Explanation field.
- example: Example field.
- keyword (only in Output2): Relevant keyword(s) for the output.

## 5. Create an AI Agent
```python
ai_expert_agent = Agent(
    role = "AI expert",
    goal = "Answer user AI questions",
    backstory = """
        Bạn là một chuyên gia về AI. Bạn hãy lắng nghe câu hỏi của người dùng và trả lời thật chi tiết kèm ví dụ minh họa.
    """,
    llm=llm,
)
```
Agent: Represents an AI Agent.
- role: The agent’s role is "AI Expert."
- goal: The agent’s goal is to answer user questions about AI.
- backstory: Context or description for the agent’s behavior.
- llm: Uses the GPT-4o-mini language model.

## 6. Define a Task
```python
task = Task(
    description='Trả lời câu hỏi Machine Learning là gì?',
    expected_output='Một câu trả lời phù hợp với câu hỏi của người dùng, lưu ý đưa ra những ví dụ minh họa rõ ràng cho sự giải thích của mình',
    agent=ai_expert_agent,
    output_json = Output2 # set json format
)
```
Task: Defines a specific task for the agent.
- description: The task is to answer the question: "What is Machine Learning?"
- expected_output: Describes the expected result, emphasizing clarity and illustrative examples.
- agent: Assigns the task to ai_expert_agent.
- output_json: Specifies the output format using the Output2 class.

## 7. Initialize the Crew
```python
crew = Crew(
    agents=[ai_expert_agent],
    tasks=[task],
    verbose=True
)
```
Crew: A collection of Agents performing Tasks.
- agents: Includes the ai_expert_agent.
- tasks: Includes the defined task.
- verbose: If True, displays detailed logs during execution.

## 8. Execute the Crew
```python
result = crew.kickoff()
print(result)
```
- crew.kickoff(): Starts the execution of tasks by the Crew.
- print(result): Displays the result returned by the Agent upon completing the task.

## Summary
- The code creates an AI system with:
    - An Agent (ai_expert_agent) acting as an AI expert.
    - A Task that asks the Agent to explain "What is Machine Learning?" with examples.
    - A Crew to organize and execute the task.
- The Agent uses GPT-4o-mini to generate detailed responses formatted according to predefined JSON structures.

