import streamlit as st
from backend.rag import load_qa_chain
from langchain.schema import HumanMessage, AIMessage

st.set_page_config(page_title="Constitution Chatbot", page_icon="📜")
st.title("Constitution Chatbot")
st.write("Ask anything about the Constitution!")

meta_responses = {
    "who are you": "I'm a chatbot trained to answer questions about the Constitution of India. Ask me anything related to constitutional articles, rights, duties, amendments, etc!",
    "what can you do": "I can help you explore and understand the Indian Constitution by answering questions about its articles, schedules, amendments, and more.",
    "hello": "Hi there! I'm your Constitution assistant. How can I help?",
    "hi": "Hello! What would you like to know about the Constitution?",
    "help": "You can ask me questions like 'What is Article 21?' or 'What are fundamental rights?'"
}



# Only create once per session
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain, st.session_state.is_relevant_question = load_qa_chain()

qa_chain = st.session_state.qa_chain
is_relevant_question = st.session_state.is_relevant_question

query = st.text_input("Enter your question:")

if query:
    normalized = query.lower().strip("?!. ")
    if normalized in meta_responses:
        st.success("**Answer:**")
        st.write(meta_responses[normalized])
    else:
        with st.spinner("Searching the Constitution..."):
            if is_relevant_question(query):
                result = qa_chain({"question": query})
                answer = result["answer"]

                st.success("**Answer:**")
                st.write(answer)

                memory_vars = qa_chain.memory.load_memory_variables({})
                st.write("### Chat History")
                for msg in memory_vars["chat_history"]:
                    role = "User" if isinstance(msg, HumanMessage) else "Assistant"
                    st.markdown(f"**{role}:** {msg.content}")
            else:
                st.warning("⚠️ That question may be out of the Constitution's scope.")