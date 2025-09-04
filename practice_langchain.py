# Commented out IPython magic to ensure Python compatibility.
# %pip -q install -U langchain langchain-community langchain-google-genai faiss-cpu tiktoken

import os, getpass

if " " not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")  # fast, inexpensive model
resp = llm.invoke("Say hello from LangChain in one short sentence.")
print(resp.content)

import os
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.7,
    max_tokens=256,
)

# Simple usage
response = llm.invoke("What are the benefits of renewable energy?")
print(response.content)        # Text output
print(response.usage_metadata) # Optional: token usage details

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

chat_model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.3,
    max_tokens=256,
)

messages = [
    SystemMessage(content="You are a helpful AI assistant."),
    HumanMessage(content="Explain quantum computing in simple terms."),
]

response = chat_model.invoke(messages)

print(response.content)

from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

text = "LangChain is a powerful framework for LLM applications."
embedding_vector = embeddings.embed_query(text)

print(f"Embedding dimension: {len(embedding_vector)}")
print(f"First 5 values: {embedding_vector[:5]}")

from langchain_core.prompts import PromptTemplate

template = """
You are an expert {expertise} consultant.
Please provide advice on: {query}
Consider the following context: {context}
Your advice:
"""

prompt = PromptTemplate(
    input_variables=["expertise", "query", "context"],
    template=template,
)

formatted_prompt = prompt.format(
    expertise="financial planning",
    query="retirement savings strategies",
    context="for a 30-year-old software engineer earning $120k annually",
)
print("----- Formatted Prompt -----")
print(formatted_prompt)

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.3,
    max_tokens=300,
)

response = llm.invoke(formatted_prompt)
print("\n----- Model Response -----")
print(response.content)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# Create chat prompt template
system_template = "You are a {role} who {specialty}."
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

human_template = "{request}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([
    system_message_prompt,
    human_message_prompt,
])

# Format the messages
messages = chat_prompt.format_messages(
    role="data scientist",
    specialty="specializes in machine learning model optimization",
    request="How can I improve the performance of my neural network?"
)

for message in messages:
    print(f"{message.__class__.__name__}: {message.content}")

from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate

# Define examples
examples = [
    {
        "input": "I need to analyze sales data",
        "output": "I recommend using pandas for data manipulation, matplotlib for visualization, and seaborn for statistical plots."
    },
    {
        "input": "How do I build a web scraper?",
        "output": "For web scraping, use requests for HTTP, BeautifulSoup for HTML parsing, and optionally Selenium for dynamic pages."
    }
]

# Create example template
example_template = """
Input: {input}
Output: {output}
""".strip()

example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template=example_template,
)

# Create few-shot prompt
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="You are a Python programming expert. Provide helpful library recommendations:",
    suffix="Input: {query}\nOutput:",
    input_variables=["query"],
    example_separator="\n\n",
)

# Use the prompt
formatted = few_shot_prompt.format(query="I want to build a REST API for my project")
print(formatted)

from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

# Create components
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Write a brief summary about {topic}:"
)

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.7,
)

# Create chain (prompt -> model -> string)
chain = prompt | llm | StrOutputParser()

# Run chain
result = chain.invoke({"topic": "artificial intelligence"})
print(result)

# Commented out IPython magic to ensure Python compatibility.
# %pip -q install -U langchain-community pypdf

from langchain_community.document_loaders import PyPDFLoader

# Load a PDF file
loader = PyPDFLoader("Demo Setup Diagram.pdf")
pages = loader.load()

print(f"Loaded {len(pages)} pages")
for i, page in enumerate(pages[:3]):  # Show first 3 pages
    print(f"Page {i + 1}: {page.page_content[:150]}...")

# Commented out IPython magic to ensure Python compatibility.
# %pip -q install -U langchain langchain-community langchain-google-genai duckduckgo-search
# %pip install -U ddgs

import math

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

# Initialize LLM (Gemini 1.5 Flash)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

# Define custom tools
@tool
def calculator(expression: str) -> str:
    """Calculate mathematical expressions. Use this for any math calculations."""
    try:
        # Caution: eval is for demo purposes only.
        result = eval(expression, {"__builtins__": {}}, {"math": math})
        return f"The result is: {result}"
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def weather_info(city: str) -> str:
    """Get current weather information for a city (mock)."""
    # This is a mock implementation - replace with a real weather API
    return f"The weather in {city} is sunny with a temperature of 22°C"

@tool
def web_search(query: str) -> str:
    """Search the web for information."""
    search = DuckDuckGoSearchRun()
    return search.run(query)

# Create tools list
tools = [calculator, weather_info, web_search]

# Create prompt template (must include agent_scratchpad placeholder)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that can use tools to answer questions."),
    MessagesPlaceholder("chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

# Create agent using OpenAI-functions style tool calling
agent = create_openai_functions_agent(llm, tools, prompt)

# Create agent executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    return_intermediate_steps=True,
)

# Use the agent
def run_agent_query(query: str):
    """Run a query through the agent."""
    result = agent_executor.invoke({"input": query})
    print(f"Query: {query}")
    print(f"Answer: {result['output']}\n")
    print("Intermediate Steps:")
    for action, observation in result.get("intermediate_steps", []):
        print(f"- Action: {action.tool}")
        print(f"  Input:  {action.tool_input}")
        print(f"  Output: {observation}")

# Example usage
if __name__ == "__main__":
    queries = [
        "What is 15 * 23 + 47?",
        "What's the weather like in New York?",
        "Search for recent developments in artificial intelligence",
    ]
    for q in queries:
        print("=" * 50)
        run_agent_query(q)
        print()

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = PromptTemplate(
    input_variables=["topic"],
    template="Write a two-sentence overview about {topic}:"
)

chain = prompt | llm | StrOutputParser()
print(chain.invoke({"topic": "retrieval-augmented generation"}))

from langchain_core.prompts import ChatPromptTemplate

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Keep answers concise."),
    ("human", "{question}")
])

print((chat_prompt | llm | StrOutputParser()).invoke({"question": "What is LangChain?"}))

from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate

class Note(BaseModel):
    title: str = Field(..., description="Short title")
    bullets: list[str] = Field(..., description="3 concise bullet points")

parser = JsonOutputParser(pydantic_object=Note)

tpl = PromptTemplate(
    template="Create study notes as JSON about: {topic}\n{format_instructions}",
    input_variables=["topic"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

json_chain = tpl | llm | parser
print(json_chain.invoke({"topic": "vector databases"}))

from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

memory = ConversationBufferMemory(return_messages=True)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are friendly. Keep replies under 2 sentences."),
    MessagesPlaceholder("history"),
    ("human", "{input}")
])

def chat_turn(text):
    hist = memory.load_memory_variables({})["history"]
    out = (prompt | llm | StrOutputParser()).invoke({"history": hist, "input": text})
    memory.save_context({"input": text}, {"output": out})
    return out

print(chat_turn("My name is Chandan."))
print(chat_turn("What did I say my name is?"))

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

docs = [
    Document(page_content="LangChain provides prompts, chains, memory, and tools."),
    Document(page_content="RAG retrieves context before generation for grounded answers."),
    Document(page_content="FAISS enables fast vector similarity search."),
]

emb = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
vs = FAISS.from_documents(docs, emb)
retriever = vs.as_retriever(search_kwargs={"k": 2})

rag_prompt = PromptTemplate.from_template(
    "Use the context to answer the question.\n\nContext:\n{ctx}\n\nQuestion: {q}\n\nAnswer briefly:"
)

def join(ds): return "\n\n".join(d.page_content for d in ds)

rag_chain = ({"ctx": retriever | join, "q": RunnablePassthrough()} | rag_prompt | llm | StrOutputParser())
print(rag_chain.invoke("What does FAISS do in RAG?"))

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "Be concise."),
    ("human", "Explain retrieval-augmented generation in 5 lines.")
])
for chunk in (chat_prompt | llm | StrOutputParser()).stream({}):
    print(chunk, end="", flush=True)
print()

validator = PromptTemplate.from_template(
    "Is the following text 2-3 sentences and formal tone? Answer yes/no.\n\n{text}"
)
def ensure_constraints(topic):
    text = chain.invoke({"topic": topic})
    verdict = (validator | llm | StrOutputParser()).invoke({"text": text}).lower()
    if "yes" in verdict:
        return text
    fix = PromptTemplate.from_template(
        "Rewrite to 2-3 sentences, formal tone:\n\n{text}"
    ) | llm | StrOutputParser()
    return fix.invoke({"text": text})

print(ensure_constraints("contrast RAG vs fine-tuning"))

docs = [
    "LangChain composes prompts, models, and tools.",
    "RAG retrieves relevant chunks before generation.",
    "Vector DBs enable semantic search over embeddings."
]

map_prompt = ChatPromptTemplate.from_messages([
    ("human", "Write a one-sentence summary:\n\n{context}")
])
map_chain = map_prompt | llm | StrOutputParser()
mapped = map_chain.batch([{"context": d} for d in docs])

reduce_prompt = ChatPromptTemplate.from_messages([
    ("human", "Combine these summaries into one paragraph:\n\n{docs}")
])
reduce_chain = reduce_prompt | llm | StrOutputParser()
final_summary = reduce_chain.invoke({"docs": "\n".join(f"- {m}" for m in mapped)})
print(final_summary)

