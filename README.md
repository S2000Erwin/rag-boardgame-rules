# rag-boardgame-rules
RAG a boardgame rule and ask questions

# setup

* install ollama

the current method is
`curl -fsSL https://ollama.com/install.sh | sh` but you better check the Ollama website

* `ollama pull llama3.1` or pull other LLMs

* setup a LangChain account and obtain an API Key

* write the API Key to environment variable or to `rag.py`

* `pip install -r requirements`

# Run the program

* change the document URL to whatever links
* change the `questions` List
* run the program `python rag.py`


Erwin Lau 2024