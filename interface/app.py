import sys
from pathlib import Path

# allow importing local modules
sys.path.append(str(Path(__file__).resolve().parent.parent))

from run.retrievalPipeline import runRetrievalPipeline
import streamlit as st

st.title('RAG MULTI AGENT SYSTEM')

query = st.text_input("insira sua pergunta")

if st.button("search") and query:
    response = runRetrievalPipeline(raw_query= query )

    if response:
        st.write(response)