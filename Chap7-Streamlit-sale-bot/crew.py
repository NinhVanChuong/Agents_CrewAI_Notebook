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
from crewai.project import CrewBase, agent, crew, task
from langchain_openai import ChatOpenAI
from thefuzz import process
from typing import Any

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
    description: str = """
        Trả lời những câu hỏi liên quan tới thông tin liên hệ, danh mục sản phẩm, các loại sản phẩm, thông tin chứng nhận, chính sách công ty (chính sách npp, chính sách sỉ, chính sách đổi trả)
        Nếu câu hỏi nào bạn không có thông tin hoặc không biết câu trả lời thì hãy trả lời là:"Với câu hỏi này hiện em chưa trả lời được. Anh/chị có thể cung cấp số điện thoại để em có thể hỗ trợ trả lời sau được không ạ?"
    """

    def _run(self, user_message: str) -> str:
        response = query_engine.query(user_message)

        return response


class ProductSearchTool(BaseTool):
    name: str = "Product Search tool"
    description: str = """
        Trả lời những câu hỏi liên quan đến tìm kiếm thông tin về một sản phẩm cụ thể của công công ty.
        Những loại câu hỏi không liên quan đến tìm kiếm sản phẩm cụ thể như:
        - Sản phẩm bên em có gì
        - A xin thông tin sản phẩm bên em
        - Danh mục sản phẩm bên em có gì
        - Bên em bán những gì
        - Bên em có những loại sản phẩm nào
        - Bên em bán những sản phẩm gì
        - Em có sp gì
        - sp bên em gồm những gì
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

class CrewaiSaleChatbotCrew:
    """CrewaiSaleChatbotCrew crew"""

    def __init__(self):
        sale_agent = Agent(
            role = "Sale",
            goal = "Trả lời câu hỏi của người dùng",
            backstory = """
                Bạn là một nhân viên chăm sóc khách hàng chuyên nghiệp và thân thiện của công ty NewTommy. Nhiệm vụ của bạn là hỗ trợ khách hàng bằng cách lắng nghe thắc mắc, giải đáp câu hỏi, và tư vấn sản phẩm với giọng điệu vui tươi, lịch sự, dễ thương.
                - Tuyệt đối không được bịa đặt thông tin, sử dụng ngôn ngữ không phù hợp, hoặc thô lỗ.
                - Bạn hãy xưng hô anh hoặc chị dựa theo cuộc trò chuyện với khách hàng.
                - Sử dụng từ ngữ như “dạ, vâng” trong câu trả lời để thể hiện sự chuyên nghiệp.
                - Khi được hỏi về sản phẩm bán chạy thì bạn trả lời thanh rong biển và các loại hộp quà.

                Khi được hỏi về những sản phẩm của công ty như:
                - Bên em có những sp gì?
                - Sản phẩm bên em có gì?
                - Bên em bán những gì?
                thì bạn hãy trả lời:"Những sản phẩm của công ty NewTommy bao gồm Mắc ca sấy mật ong, Hạt hỗn hợp, Điều bóc vỏ rang muối, Xoài sấy dẻo."


                Bạn hãy chủ động xin số điện thoại của khách hàng khi khách hàng muốn đặt mua sản phẩm của bạn như:
                - Anh/chị muốn đặt 2 sản phẩm này
                - Anh/Chị muốn mua 3 sản phẩm xxx với xxx là tên sản phẩm
                - Anh/Chị muốn đặt mua 1 sản phẩm xxx với xxx là tên sản phẩm

                Khi bạn nhận được số điện thoại từ câu chat của người dùng như:
                - số điện thoại của anh/chị là xxx với xxx là số điện thoại
                - xxx là số điện thoại anh/chị nha với xxx là số điện thoại
                - xxx với xxx là số điện thoại
                - sđt: xxx với xxx là số điện thoại
                - sđt anh/chị nè xxx với xxx là số điện thoại
                thì bạn hãy trả lời:"Em đã ghi nhận số điện thoại và phòng kinh doanh sẽ liên hệ lại anh/chị sau"
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

        self.crew = Crew(
            agents=[sale_agent],
            tasks=[search_task],
            manager_llm=llm,
            verbose=True
        )
    
    def chat(self, inputs: Any, *args, **kwargs) -> Any:
        return self.crew.kickoff(inputs=inputs).raw

        