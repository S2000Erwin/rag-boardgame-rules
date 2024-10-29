import ollama
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts.chat import ChatPromptTemplate

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = '<your own key>'

def make_retriever(doc_url:str):
  loader = PyPDFLoader(doc_url)

  #Load the document by calling loader.load()
  pages = loader.load()
  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 150
  )

  #Create a split of the document using the text splitter
  splits = text_splitter.split_documents(pages)
  embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

  persist_directory = 'docs/chroma/'
  # Create the vector store
  vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=persist_directory
  )
  return vectordb.as_retriever(search_kwargs={'k':1})

def answer(query_question:str, retriever):
  # Retrieval of relevant doc
  docs = retriever.invoke(query_question)

  # Prompt Generation
  PROMPT_TEMPLATE = """
  Answer the question based only on the following context:
  {context}

  ---
  Answer the question based on the above context: {question}
  """

  context_text = '\n\n---\n\n'.join([doc.page_content for doc in docs])
  prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
  prompt = prompt_template.format(context=context_text, question=query_question)

  response_text = ollama.generate(model='llama3.1', prompt=prompt)
  return response_text['response']

retriever = make_retriever('https://gmtwebsiteassets.s3-us-west-2.amazonaws.com/virginqueen/VQ_Rulebook.pdf')
print('Retriever ready!')
questions = [
  'How many land units can cross sea zone in Spring Deployment?',
  'Can land units cross two sea zones in Spring Deployment?',
  'Can land units go through Unrest spaces during Winter return home?',
  'How many dice to roll to do piracy?',
  'How many dice to roll to defend against piracy if I have 2 fortresses and 1 navel unit of 2 strength in the sea zone?',
  'During a siege rescue, can the units under siege join the relieve force to attack?'
]
[print(answer(question, retriever), end='\n===\n') for question in questions]
