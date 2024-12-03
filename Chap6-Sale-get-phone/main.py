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
from thefuzz import process

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

    def _run(self, user_message: str) -> str:
        response = query_engine.query(user_message)

        return response


class ProductSearchTool(BaseTool):
    name: str = "Product Search tool"
    description: str = """
        Trả lời những câu hỏi liên quan đến tìm kiếm thông tin về sản phẩm cụ thể của công công ty.
        Những loại câu hỏi không liên quan đến tìm kiếm sản phẩm cụ thể như:
        - Sản phẩm bên em có gì
        - A xin thông tin sản phẩm bên em
        - Danh mục sản phẩm bên em có gì
        - Bên em bán những gì
        - Bên em có những loại sản phẩm nào
    """

    def _run(self, product_name: str) -> str:
        product_list = df_product['name'].tolist()
        list_product_name_by_fuzzy = process.extract(product_name, product_list, limit=3) # [(str, int)]
        
        result_str = f"Một số sản phẩm liên quan đến {product_name} gồm: \n"
        for index, product_name_by_fuzzy in enumerate(list_product_name_by_fuzzy):
            product_name_by_fuzzy = product_name_by_fuzzy[0]
            product_info = df_product[df_product['name'] == product_name_by_fuzzy]
            if not product_info.empty:
                name = product_info.iloc[0]['name']
                price = product_info.iloc[0]['price']
                description = product_info.iloc[0]['description']
                url = product_info.iloc[0]['url']
                result_str += f"{index+1}- Sản phẩm: {name}, Giá: {price}, Mô tả: {description}, Link sản phẩm: {url} \n"
        
        return result_str

sale_agent = Agent(
    role = "Sale",
    goal = "Trả lời câu hỏi của người dùng",
    backstory = """
        Bạn là một nhân viên bán hàng chuyên nghiệp. Bạn hãy lắng nghe câu hỏi của người dùng và trả lời thật chính xác.
    """,
    llm=llm,
)

search_task = Task(
    description="""
        Phản hồi tin nhắn của người dùng: {user_message}.
        Nếu câu phản hồi của bạn sử dụng thông tin từ Product Search tool thì hãy chủ động xin số điện thoại của khách hàng ngay khi đưa ra thông tin sản phẩm
    """,
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

prompt = "a xin thông tin chôclate crunch" # HŨ CHOCOLATE CRUNCH WITH NUTS - BÁNH SOCOLA HẠT" 

inputs = {
    "user_message": f"{prompt}",
}

response = crew.kickoff(inputs=inputs)

print("response: ", response)