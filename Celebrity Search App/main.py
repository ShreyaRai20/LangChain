# Integrate our code OpenAI api

import os
from constants import openai_key
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain
from langchain.chains import SequentialChain

from langchain.memory import ConversationBufferMemory

import streamlit as st

os.environ["OPENAI_API_KEY"]=openai_key

# streamlit Framework

st.title('Celebrity Search Results')
input_text=st.text_input("Search the topic you want")

# Prompt Template
first_input_prompt=PromptTemplate(
    input_variable=['name'],
    template='tell me about celebrity {name}'
)

# Memory
person_memory = ConversationBufferMemory(input_key='name',memory_key='chat history')
dob_memory = ConversationBufferMemory(input_key='person',memory_key='chat history')
descr_memory = ConversationBufferMemory(input_key='dob',memory_key='description history')


# OpenAI LLMSS
llm=OpenAI(temperature=0.8)
chain1=LLMChain(llm=llm,prompt=first_input_prompt,verbose=True, output_key='person', memory=person_memory)

second_input_prompt=PromptTemplate(
    input_variable=['person'],
    template='tell me about celebrity {person}'
)

chain2=LLMChain(llm=llm,prompt=second_input_prompt,verbose=True, output_key='dob',memory=dob_memory)

third_input_prompt=PromptTemplate(
    input_variable=['dob'],
    template='Mention 5 major events happened around {dob} in the world'
)


chain3=LLMChain(llm=llm,prompt=third_input_prompt,verbose=True, output_key='description', memory=descr_memory)

parent_chain=SimpleSequentialChain(chains=[chain1, chain2], verbose=True)
parent_chain1=SequentialChain(chains=[chain1, chain2,chain3],input_variables=['name'],output_variables=['person','dob','description'] ,verbose=True)


# if input_text:
#     st.write(chain1.run(input_text))

if input_text:
    st.write(parent_chain1({'name':input_text}))

    with st.expander('Person Name'):
        st.info(person_memory.buffer)

    with st.expander('Major Events'):
        st.info(person_memory.buffer)

