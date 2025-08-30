from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

# specify model
model = OllamaLLM(model="llama3.2")

# template to tell the model what to do
template = """
You are an expert in answering questions about a pizza restaurant.
Here are some relevant reviews: {reviews}
Here is the question to answer: {question}
"""

# create chat prompt
chat_prompt = ChatPromptTemplate.from_template(template)
chain = chat_prompt | model

while True:
    print("\n\n--------------------------------------------------------")
    question = input("What is your question about the pizza restaurant?")
    print("\n\n")
    if question.lower() == "q":
        break

    reviews = retriever.invoke(question) # retrieve relevant reviews, embed qn 
    result = chain.invoke({"reviews": reviews, "question": question})
    print(result)
