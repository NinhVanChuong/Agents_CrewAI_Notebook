# Callback Mechanism
The code defines a workflow where two AI agents collaborate to generate a blog about Machine Learning. The first agent, an AI expert, is responsible for generating keywords and creating a detailed blog outline. The second agent, an AI blogger, uses these outputs to write a 2000-word blog post. Each task has a callback function to handle post-task actions, such as logging the results. All tasks and agents are managed within a Crew, which executes the tasks sequentially while resolving dependencies. The final output includes the keywords, the blog outline, and the completed blog post. This modular design ensures efficiency and extensibility.
## **1. Import Libraries and Set Environment**
```python
import json 
import os 
os.environ["OPENAI_API_KEY"] = ""
```
- **`json`**: Handles data in JSON format.
- **`os`**: Interacts with the operating system.
- **`os.environ["OPENAI_API_KEY"]`**: Sets the OpenAI API key for accessing the GPT-4o-mini model.

---

## **2. Import AI Workflow Libraries**
```python
from langchain_openai import ChatOpenAI
from crewai import Agent, Task, Crew
from crewai.tasks.task_output import TaskOutput
```
- **`ChatOpenAI`**: Initializes the GPT-4o-mini model for task execution.
- **`Agent, Task, Crew`**: Core components for building and managing AI workflows:
  - **Agent**: Represents an AI entity with a specific role and goal.
  - **Task**: Represents an activity or problem the Agent solves.
  - **Crew**: Organizes and manages Agents and Tasks.
- **`TaskOutput`**: Provides access to the task's description and raw output for post-processing.

---

## **3. Define a Callback Function**
```python
def callback_function(output: TaskOutput):
    print(f"Task completed!\nTask: {output.description}\nOutput: {output.raw}")
```
- **Purpose**: Defines an action to take after a task is completed.
  - In this example, it prints the task description and its raw output.
  - This function can be extended (e.g., to send notifications or save results).

---

## **4. Create AI Agents**
### **Agent 1: AI Research Agent**
```python
ai_research_agent = Agent(
    role = "AIExpert",
    goal = "Hỗ trợ giải đáp những câu hỏi liên quan tới AI",
    backstory='''Bạn là một chuyên gia trong lĩnh vực AI. Hãy lắng nghe câu hỏi của người dùng và trả lời thật phù hợp''',
    llm=llm
)
```
- **Role**: AI expert for answering questions.
- **Goal**: Provide clear answers related to AI.
- **Backstory**: Describes its expertise and user-oriented behavior.

### **Agent 2: AI Blog Agent**
```python
ai_blog_agent = Agent(
    role = "AIBlogger",
    goal = "Viết một bài blog chất lượng.",
    backstory='''Bạn là một người viết bài blog về AI, luôn sẵn sàng chỉnh sửa dựa trên nhận xét.''',
    llm=llm
)
```
- **Role**: AI blogger specializing in creating quality blog posts.
- **Goal**: Write a 2000-word blog in Vietnamese.
- **Backstory**: Describes the agent's writing skills and feedback-oriented approach.

---

## **5. Define Tasks**
### **Task 1: Generate Keywords**
```python
list_keyword_task = Task(
    description='Đưa ra những keyword về chủ đề Machine Learning',
    expected_output='Danh sách các keyword phù hợp để viết blog cho chủ đề Machine Learning',
    agent=ai_research_agent,
    callback=callback_function
)
```
- **Description**: Task to generate keywords related to Machine Learning.
- **Agent**: Assigned to `ai_research_agent`.
- **Callback**: Executes `callback_function` after the task is completed.

### **Task 2: Create Blog Outline**
```python
list_outline_task = Task(
    description="Đưa ra dàn ý phù hợp cho blog chủ đề Machine Learning",
    expected_output="Một dàn ý chi tiết",
    agent=ai_research_agent,
    callback=callback_function
)
```
- **Description**: Task to create a detailed blog outline.
- **Agent**: Assigned to `ai_research_agent`.
- **Callback**: Executes `callback_function` after the task is completed.

### **Task 3: Write Blog**
```python
write_blog_task = Task(
    description="Viết một bài blog về chủ đề Machine Learning.",
    expected_output="Một bài viết khoảng 2000 từ",
    agent=ai_blog_agent,
    callback=callback_function,
    context=[list_keyword_task, list_outline_task]
)
```
- **Description**: Task to write a 2000-word blog.
- **Agent**: Assigned to `ai_blog_agent`.
- **Context**: Depends on outputs of `list_keyword_task` and `list_outline_task`.
- **Callback**: Executes `callback_function` after the task is completed.

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
- **Agents**: Includes `ai_research_agent` and `ai_blog_agent`.
- **Tasks**: Adds all tasks defined above.
- **Verbose**: Enables detailed logging during task execution.

### **Execute the Crew**
```python
result = crew.kickoff()
print(result)
```
- **`crew.kickoff()`**: Executes the tasks in sequence.
- **`print(result)`**: Outputs the final results of all tasks, including the blog, keywords, and outline.

---

## **Flow Summary**
1. **Agent Setup**: Two agents (`ai_research_agent` and `ai_blog_agent`) are configured with specific roles and goals.
2. **Task Definition**: Three tasks are created:
   - Generate keywords for the blog topic.
   - Create a detailed outline.
   - Write the blog, using the outputs of the first two tasks as context.
3. **Callback Integration**: A callback function handles post-task actions, such as logging or sending notifications.
4. **Crew Execution**: The Crew executes tasks in order, ensuring dependencies are resolved.
5. **Final Output**: Includes the list of keywords, the blog outline, and the complete blog post.

This modular design enables efficient task management, dependency resolution, and extensible workflows.
