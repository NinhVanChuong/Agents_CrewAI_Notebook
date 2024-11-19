import json 
import os 
os.environ["OPENAI_API_KEY"] = ""

from langchain_openai import ChatOpenAI
from crewai import Agent, Task, Crew
from crewai.tasks.task_output import TaskOutput

llm = ChatOpenAI(model="gpt-4o-mini")

def callback_function(output: TaskOutput):
    # Do something after the task is completed
    # Example: Send an email to the manager
    print(f"""
        Task completed!
        Task: {output.description}
        Output: {output.raw}
    """)

ai_research_agent = Agent(
    role = "AIExpert",
    goal = "Hỗ trợ giải đáp những câu hỏi liên quan tới AI",
    backstory='''
        Bạn là một chuyên gia trong lĩnh vực AI.
        Hãy lắng nghe câu hỏi của người dùng và trả lời thật phù hợp
    ''',
    llm=llm
)

ai_blog_agent = Agent(
    role = "AIBlogger",
    goal = "Viết một bài blog chất lượng.",
    backstory='''
        Bạn là một người viết bài blog, có khả năng viết blog về lĩnh vực AI.
        Bạn tạo ra một bài viết mỗi lần một lượt.
        Bạn không bao giờ đưa ra nhận xét đánh giá.
        Bạn sẵn sàng tiếp nhận nhận xét từ người đánh giá và sẵn lòng chỉnh sửa bài viết dựa trên các nhận xét đó.
        Bạn hãy viết blog bằng tiếng Việt.
    ''',
    llm=llm
)

list_keyword_task = Task(
    description='Đưa ra những keyword về chủ đề Machine Learning',
    expected_output='Danh sách các keyword phù hợp để viết blog cho chủ đề Machine Learning',
    agent=ai_research_agent,
    callback=callback_function
)

list_outline_task = Task(
    description="Đưa ra dàn ý phù hợp cho blog chủ đề Machine Learning",
    expected_output="Một dàn ý chi tiết",
    agent=ai_research_agent,
    callback=callback_function
)

write_blog_task = Task(
    description="Viết một bài blog về chủ đề Machine Learning.",
    expected_output="Một bài viết khoảng 2000 từ",
    agent=ai_blog_agent,
    callback=callback_function,
    context=[list_keyword_task, list_outline_task]
)

# Execute the crew
crew = Crew(
    agents=[ai_research_agent, ai_blog_agent],
    tasks=[list_keyword_task, list_outline_task, write_blog_task],
    verbose=True
)

result = crew.kickoff()

print(result)