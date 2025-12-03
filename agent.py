# agent.py
import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from Graph_RAG_new import Graph_RAG_new_tool
#from text_rag import text_rag_tool
#from amendment_tool import amendment_api
from templates import FINAL_SYNTHESIS_TEMPLATE
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
 
load_dotenv()
 
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",             # or gemini-1.5-pro
    temperature=0.1,
    google_api_key=os.getenv("GEMINI_API_KEY")
)
 
tools = [Graph_RAG_new_tool]
 
agent = create_agent(model = llm, tools = tools)
 
def answer_legal_query(query):
    # 1. Graph RAG
    graph_results = agent.invoke({
        "messages": [HumanMessage(content=f"Use the Graph_RAG_new_tool to answer: {query}")]
    })["messages"][-1].content
   
    '''
    # 2. Text RAG
    text_results = agent.invoke({
        "messages": [HumanMessage(content=f"Use the text_rag_tool to answer: {query}\n\nGraph results: {graph_results}")]
    })["messages"][-1].content
    '''
    '''
    # 3. Amendment API
    amendments = agent.invoke({
        "messages": [HumanMessage(content="Use the amendment_api to find amendments for Constitution of India")]
    })["messages"][-1].content
    '''
 
    # 4. Final synthesis
    prompt = PromptTemplate(
        template=FINAL_SYNTHESIS_TEMPLATE,
        input_variables=["graph_results", "text_results", "amendments", "query"]
    )
 
    return llm.invoke(
        prompt.format(
            graph_results=graph_results,
            query=query
        )
    ).content
 
 