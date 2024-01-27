## Integrate our code with openai API

import os
from constants import openai_key
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain

from langchain.chains import SimpleSequentialChain
import streamlit as st

os.environ["OPENAI_API_KEY"] = openai_key

# streamlit framework

st.title('Celebrity search result')
input_text = st.text_input('Search the topic u want')

# Prompt template 1

first_input_prompt = PromptTemplate(
    input_variables=['name'],
    template='Tell me about  celebrity {name}'
)


## OPENAI LLMS
llm=OpenAI(temperature=0.8)
chain=LLMChain(llm=llm,prompt=first_input_prompt,verbose=True,output_key='person')

# Prompt template 2

second_input_prompt = PromptTemplate(
    input_variables=['person'],
    template='when was {person} born'
)

chain2=LLMChain(llm=llm,prompt=first_input_prompt,verbose=True,output_key='dob')

parent_chain=SimpleSequentialChain(chains=[chain,chain2], verbose=True)

if input_text:
    st.write(chain.run(input_text))






