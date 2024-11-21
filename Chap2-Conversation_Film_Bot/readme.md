# Conversation Film Bot
Conversation Film Bot is a chatbot developed to provide movie recommendations and advice based on user preferences and specific requirements. This project leverages the CrewAI framework to manage and coordinate AI agents, enabling natural and efficient conversations.

## 🎯 Objectives
- Deliver personalized movie recommendations tailored to individual user preferences.
- Create a friendly and engaging conversational experience.
- Utilize advanced AI technologies to analyze and understand user requests effectively.

## Configuration

- **`agents.yaml`**: Defines the assistant agent responsible for handling user inquiries related to the film domain. The bot is designed to:
    - Provide answers to user questions about movies.
    - Offer personalized movie recommendations based on predefined genres and details outlined in its backstory.
    - Collect user information, such as email and phone number, to facilitate follow-up consultations and enhance the user experience.
- **`tasks.yaml`**:Describes the tasks assigned to the assistant, specifically focusing on responding to user messages in a meaningful and engaging way.

## Bot 

```python
@CrewBase
class CrewaiFilmChatbotCrew:
    """CrewaiFilmChatbotCrew crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def assistant(self) -> Agent:
        return Agent(
            config=self.agents_config["assistant"],
            llm=llm,
            verbose=False,
        )

    @task
    def assistant_task(self) -> Task:
        return Task(config=self.tasks_config["assistant_task"], agent=self.assistant())

    @crew
    def crew(self) -> Crew:
        """Creates the CrewaiFilmChatbotCrew crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
```
Class **`CrewaiFilmChatbotCrew`**, which is a part of a framework for managing AI agents and tasks, likely designed to handle a conversational film chatbot. Here's a brief description of the main components:
### 1. Class Definition
- The **`CrewaiFilmChatbotCrew`** class inherits from **`CrewBase`**, indicating it is part of a larger system for orchestrating agents and tasks.
### 2. Agents and Tasks Configuration
- It uses two configuration files, **`config/agents.yaml`** and **`config/tasks.yaml`**, to define the behavior and settings for the agents and tasks.
### 3. Assistant Agent
- The **`assistant`** method defines an agent using the configuration for an "assistant" from the agents' YAML file.
- The **`llm`** parameter (likely referring to a language model) is left undefined, allowing for further customization.
- The **`verbose`** flag is set to False, reducing unnecessary logging output.
### 4. Assistant Task
The **`assistant_task`** method defines a task associated with the assistant agent, utilizing its configuration from the tasks YAML file.
### 5. Crew Definition
- The crew method initializes a Crew object that ties the agents and tasks together in a sequential processing order (**`Process.sequential`**).
- The crew represents the execution environment for the chatbot's functionality, managing the flow between agents and tasks.

## Usage
To start the chatbot, run the *app.py* file:
```
streamlit run app.py
```

