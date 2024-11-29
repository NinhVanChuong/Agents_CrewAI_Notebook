import pandas as pd 
import os
import chromadb
import sys
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding

from readers import parse_multiple_files
from crewai.tools import BaseTool
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI

os.environ["OPENAI_API_KEY"] = ""

llm = ChatOpenAI(model="gpt-4o-mini")
file_path = 'db/product.csv' 
df_product = pd.read_csv(file_path)

embed_model = OpenAIEmbedding(model="text-embedding-3-large")
chroma_client = chromadb.PersistentClient(path="./DB/rag")
chroma_collection = chroma_client.get_or_create_collection("test")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
KB_index = VectorStoreIndex.from_vector_store(
    vector_store, storage_context=storage_context, embed_model=embed_model, show_progress=True
)
query_engine = KB_index.as_query_engine()

class KBSearchTool(BaseTool):
    name: str = "KB Search tool"
    description: str = "Trả lời những câu hỏi liên quan tới công ty"

    def _run(self, user_input: str) -> str:
        response = query_engine.query(user_input)

        return response


class ProductSearchTool(BaseTool):
    name: str = "Product Search tool"
    description: str = "Trả lời những câu hỏi liên quan đến tìm kiếm thông tin về sản phẩm cụ thể của công công ty"

    def _run(self, product_name: str) -> str:
        product_info = df_product[df_product['name'] == product_name]
        
        if not product_info.empty:
            price = product_info.iloc[0]['price']
            description = product_info.iloc[0]['description']
            return f"Thông tin sản phẩm {product_name}: {price}, {description}"
        else:
            return f"product_name: {product_name}"

sale_agent = Agent(
    role = "Sale",
    goal = "Trả lời câu hỏi của người dùng",
    backstory = """
        Bạn là một nhân viên bán hàng chuyên nghiệp. Bạn hãy lắng nghe câu hỏi của người dùng và trả lời thật chính xác.
    """,
    llm=llm,
)

search_task = Task(
    description='Phản hồi tin nhắn của người dùng: {user_message}',
    expected_output='Một câu trả lời phù hợp với câu hỏi của người dùng.',
    agent=sale_agent,
    tools=[KBSearchTool(), ProductSearchTool()]
)

crew = Crew(
    agents=[sale_agent],
    tasks=[search_task],
    manager_llm=llm,
    verbose=True
)

prompt = "cho a xin thông tin về HŨ CHOCOLATE CRUNCH WITH NUTS - BÁNH SOCOLA HẠT"

inputs = {
    "user_message": f"{prompt}",
}

response = crew.kickoff(inputs=inputs)

print("response: ", response)