from langchain_community.llms import Ollama
from langchain.agents import load_tools, initialize_agent
import streamlit as st
import os
from langchain_community.utilities import SerpAPIWrapper
from langchain.chains import LLMChain
from langchain import PromptTemplate


os.environ["SERPAPI_API_KEY"] = "SERPAPI_API_KEY"

llm=Ollama(model="Mistral")

st.title("Ask a question")
prompt = st.text_input("ask your question here..")

tool_names=['serpapi']
tools=load_tools(tool_names,llm=llm)

agent=initialize_agent(
    agent="structured-chat-zero-shot-react-description",
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=30
)

if prompt:
    response=agent.run(prompt)
    # response=llm.invoke(prompt)
    st.write(response)
