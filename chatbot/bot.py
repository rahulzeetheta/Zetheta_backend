from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import CTransformers
from chatbot.config import get_vector_store
from ctransformers import AutoModelForCausalLM
from langchain_community.llms import CTransformers

# Load vector DB
vector_store = get_vector_store()

# Load local GGUF model
llm = CTransformers(
    model_path="chatbot_models\llama-2-7b-chat.Q4_0.gguf",  # 👈 update here
    model_type="llama",                                # or try "llama2"
    config={"max_new_tokens": 128, "temperature": 0.01}
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

chain = ConversationalRetrievalChain.from_llm(llm=llm,
                                              chain_type="stuff",
                                              retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
                                              memory=memory)

def get_response(query, chat_history):
    result = chain({"question": query, "chat_history": chat_history})
    return result["answer"]
