# Referring to Other Tasks
The code demonstrates a collaborative workflow using AI agents to create a high-quality blog on Machine Learning. First, the environment is set up, and two agents are defined: an AI research agent for generating keywords and outlines, and an AI blog agent for writing the blog. Three tasks are created: generating keywords, creating a detailed blog outline, and writing a blog, with the latter depending on the outputs of the first two tasks. These agents and tasks are assembled into a Crew, which executes the tasks in sequence. The final output includes a list of keywords, a structured outline, and a completed blog post, showcasing modular task management and agent collaboration.

## **1. Environment Setup**
```python
import json 
import os 
os.environ["OPENAI_API_KEY"] = ""
```
- **`json`**: Handles input/output in JSON format.
- **`os`**: Interacts with the operating system.
- **`os.environ["OPENAI_API_KEY"]`**: Sets the OpenAI API key for accessing GPT-4o-mini.

---

## **2. Import Libraries**
```python
from langchain_openai import ChatOpenAI
from crewai import Agent, Task, Crew
from pydantic import BaseModel
from typing import Union
```
- **`ChatOpenAI`**: Interface to interact with the GPT-4o-mini model.
- **`Agent, Task, Crew`**: Components from the `crewai` library:
  - **Agent**: Represents individual AI roles with specific goals and behavior.
  - **Task**: Defines tasks for agents to execute.
  - **Crew**: Manages the coordination of agents and tasks.
- **`BaseModel`**: Enables structured data validation (not directly used in this code).
- **`Union`**: Supports combined type hints (not used here).

---

## **3. Initialize the Language Model**
```python
llm = ChatOpenAI(model="gpt-4o-mini")
```
- Initializes the GPT-4o-mini model for use by the agents.

---

## **4. Create AI Agents**
### **Agent 1: AI Research Agent**
```python
ai_research_agent = Agent(
    role = "Chuyên gia về AI",
    goal = "Hỗ trợ giải đáp những câu hỏi liên quan tới AI",
    backstory='''Bạn là một chuyên gia trong lĩnh vực AI. Hãy lắng nghe câu hỏi của người dùng và trả lời thật phù hợp''',
    llm=llm,
    verbose=False,
)
```
- **Role**: AI Expert.
- **Goal**: Answer AI-related questions.
- **Backstory**: Contextualizes the agent’s expertise and behavior.

### **Agent 2: AI Blog Agent**
```python
ai_blog_agent = Agent(
    role = "Chuyên gia viết blog",
    goal = "Viết một bài blog chất lượng.",
    backstory='''Bạn là một người viết bài blog về AI, luôn sẵn sàng tiếp nhận và chỉnh sửa bài viết theo nhận xét.''',
    llm=llm,
    verbose=False,
)
```
- **Role**: Blog Writer.
- **Goal**: Write high-quality blogs, specifically in Vietnamese.
- **Backstory**: Outlines the agent's writing capabilities and readiness to incorporate feedback.

---

## **5. Define Tasks**
### **Task 1: Generate Keywords**
```python
list_keyword_task = Task(
    description='Đưa ra những keyword về chủ đề Machine Learning',
    expected_output='Danh sách các keyword phù hợp để viết blog cho chủ đề Machine Learning',
    agent=ai_research_agent,
)
```
- **Description**: Generate keywords relevant to Machine Learning.
- **Expected Output**: A list of keywords for blog writing.
- **Agent**: Assigned to `ai_research_agent`.

### **Task 2: Create Blog Outline**
```python
list_outline_task = Task(
    description="Đưa ra dàn ý phù hợp cho blog chủ đề Machine Learning",
    expected_output="Một dàn ý chi tiết",
    agent=ai_research_agent,
)
```
- **Description**: Generate a detailed outline for a blog on Machine Learning.
- **Expected Output**: A structured blog outline.
- **Agent**: Assigned to `ai_research_agent`.

### **Task 3: Write the Blog**
```python
write_blog_task = Task(
    description="Viết một bài blog về chủ đề Machine Learning.",
    expected_output="Một bài viết khoảng 2000 từ",
    agent=ai_blog_agent,
    context=[list_keyword_task, list_outline_task]
)
```
- **Description**: Write a 2000-word blog on Machine Learning.
- **Expected Output**: A detailed blog post.
- **Agent**: Assigned to `ai_blog_agent`.
- **Context**: Relies on the outputs of `list_keyword_task` and `list_outline_task`.

---

## **6. Assemble and Execute the Crew**
### **Assemble the Crew**
```python
crew = Crew(
    agents=[ai_research_agent, ai_blog_agent],
    tasks=[list_keyword_task, list_outline_task, write_blog_task],
    verbose=True
)
```
- **Agents**: Combines `ai_research_agent` and `ai_blog_agent`.
- **Tasks**: Includes all tasks in the execution flow.
- **Verbose**: Enables detailed logging during execution.

### **Execute the Crew**
```python
result = crew.kickoff()
print(result)
```
- **`crew.kickoff()`**: Executes the tasks in sequence, ensuring dependencies are resolved.
- **`print(result)`**: Outputs the final results, including the blog, generated keywords, and outline.

---

## **Flow Summary**
1. **Agent Creation**:
   - An AI research agent generates keywords and outlines for a blog.
   - An AI blog agent writes the blog using the outputs of the research agent.
2. **Task Execution**:
   - The research agent completes `list_keyword_task` and `list_outline_task`.
   - The blog agent uses these outputs as context to perform `write_blog_task`.
3. **Output**:
   - The final result includes:
     - A list of keywords.
     - A detailed blog outline.
     - A complete blog post.

This modular flow allows scalable task management and collaboration between agents for efficient content creation.
