import streamlit as st 
from crew import CrewaiSaleChatbotCrew

st.title("💬 Chatbot")
st.caption("🚀 I'm a Local Bot")

if "messages" not in st.session_state:
    # Load chat history from the database
    st.session_state["messages"] = []
    st.session_state["messages"] = [{"role": "assistant", "content":"Bạn cần tôi hỗ trợ gì ạ?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    context = "\n".join(f"{message['role']}: {message['content']}" for message in st.session_state["messages"])
    inputs = {
        "user_message": f"{prompt}",
        "context": f"{context}",
    }

    # final = project_crew.kickoff()
    response = CrewaiSaleChatbotCrew().chat(inputs)

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)