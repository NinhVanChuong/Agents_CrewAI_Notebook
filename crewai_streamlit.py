import streamlit as st 
import os 
os.environ["OPENAI_API_KEY"] = ""

from crewai import Crew, Process, Agent, Task 
from langchain_core.callbacks import BaseCallbackHandler 
from typing import TYPE_CHECKING, Any, Dict, Optional 
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

avators = {"Writer":"https://cdn-icons-png.flaticon.com/512/320/320336.png",
            "Reviewer":"https://cdn-icons-png.freepik.com/512/9408/9408201.png"}

class MyCustomHandler(BaseCallbackHandler):
    def __init__(self, agent_name: str) -> None:
        self.agent_name = agent_name

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Print out that we are entering a chain."""
        print("input chain: ", inputs["input"])
        st.session_state.messages.append({"role": "assistant", "content": inputs["input"]})
        st.chat_message("assistant").write(inputs["input"])

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Print out that we finished a chain."""
        print("end chain: ", outputs["output"])
        st.session_state.messages.append({"role": self.agent_name, "content": outputs["output"]})
        st.chat_message(self.agent_name, avatar=avators[self.agent_name]).write(outputs["output"])  

writer = Agent(
    role='Chuyên gia viết blog',
    backstory='''
        Bạn là một người viết bài blog, có khả năng viết blog về lĩnh vực AI.
        Bạn tạo ra một bài viết mỗi lần một lượt.
        Bạn không bao giờ đưa ra nhận xét đánh giá.
        Bạn sẵn sàng tiếp nhận nhận xét từ người đánh giá và sẵn lòng chỉnh sửa bài viết dựa trên các nhận xét đó.
        Bạn hãy viết blog bằng tiếng Việt.
    ''',
    goal="Viết và chỉnh sửa một bài blog chất lượng.",
    llm=llm,
    callbacks=[MyCustomHandler("Writer")],
)
reviewer = Agent(
    role='Chuyên gia đánh giá chất lượng blog',
    backstory='''
        Bạn là một người đánh giá bài viết chuyên nghiệp và rất hữu ích trong việc cải thiện bài viết.
        Bạn đánh giá các bài viết và đưa ra các khuyến nghị thay đổi để bài viết phù hợp hơn với yêu cầu của người dùng.
        Bạn sẽ đưa ra nhận xét sau khi đọc toàn bộ bài viết, vì vậy bạn sẽ không tạo ra bất kỳ nội dung nào nếu bài viết chưa được cung cấp đầy đủ.
        Bạn không bao giờ tự mình tạo ra các bài blog.
        Những đánh giá của bạn được thể hiện bằng tiếng Việt
    ''',
    goal="Liệt kê các yếu tố cần cải thiện của một bài blog cụ thể. Không đưa ra nhận xét về phần tóm tắt hoặc mở đầu của bài viết.",
    llm=llm,
    callbacks=[MyCustomHandler("Reviewer")],
)

st.title("💬 Chatbot")
st.caption("🚀 I'm a Local Bot")

if "messages" not in st.session_state:
    # Load chat history from the database
    st.session_state["messages"] = []
    st.session_state["messages"] = [{"role": "assistant", "content":"Bạn muốn tôi giúp bạn viết blog nào?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    task1 = Task(
      description=f"""Viết một bài blog về {prompt}. """,
      agent=writer,
      expected_output="Một bài viết khoảng 2000 từ"
    )

    task2 = Task(
      description="""Liệt kê các nhận xét đánh giá để cải thiện toàn bộ nội dung của bài blog nhằm giúp nó lan tỏa mạnh mẽ hơn trên mạng xã hội.""",
      agent=reviewer,
      expected_output="Liệt kê các điểm chính về những chỗ cần được cải thiện.",
    )
    # Establishing the crew with a hierarchical process
    project_crew = Crew(
        tasks=[task1, task2],  # Tasks to be delegated and executed under the manager's supervision
        agents=[writer, reviewer],
        manager_llm=llm,
        process=Process.hierarchical   
    )

    final = project_crew.kickoff()

    result = f"## Đây là blog mà tôi viết cho bạn \n\n {final}"
    st.session_state.messages.append({"role": "assistant", "content": result})
    st.chat_message("assistant").write(result)