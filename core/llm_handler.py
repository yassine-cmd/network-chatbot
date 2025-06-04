import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from typing import Optional, AsyncIterator, Dict, Any, List, Tuple

# Load environment variables
load_dotenv()

LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemini-pro")
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.1))
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") # Ensure this is set in .env
MAX_HISTORY_LENGTH = int(os.getenv("MAX_HISTORY_LENGTH", "10")) # Maximum number of messages to keep in history

# --- Global LLM Instance ---
_llm: Optional[ChatGoogleGenerativeAI] = None

def get_llm() -> Optional[ChatGoogleGenerativeAI]:
    """Initializes and returns the ChatGoogleGenerativeAI instance."""
    global _llm
    if _llm is None:
        if not GOOGLE_API_KEY:
            print("Error: GOOGLE_API_KEY not found in environment variables.")
            print("Please set it in your .env file.")
            return None
        try:
            print(f"Initializing LLM: {LLM_MODEL_NAME} (Temperature: {TEMPERATURE})")
            _llm = ChatGoogleGenerativeAI(
                model=LLM_MODEL_NAME,
                temperature=TEMPERATURE,
                google_api_key=GOOGLE_API_KEY,
                model_kwargs={"stream": True} # Pass stream via model_kwargs
            )
            print("LLM initialized successfully.")
        except Exception as e:
            print(f"Error initializing LLM: {e}")
            _llm = None # Ensure it stays None if initialization fails
    return _llm

# --- System Instructions ---
SYSTEM_INSTRUCTIONS = """You are NetSage, an intelligent network assistant with expertise in networking concepts, protocols, configurations, and troubleshooting.
Your goal is to provide accurate, helpful information while maintaining a natural, conversational tone that reflects your identity as a network expert.

CONVERSATION HANDLING:
- Respond naturally to greetings like "hi", "hello", "how are you", "what's your name"
- When asked about your identity, explain that you are NetSage, an intelligent network assistant
- Include occasional light personality in responses to make conversations feel natural
- Use an engaging, friendly tone while maintaining professionalism

NETWORKING EXPERTISE:
- Provide detailed explanations of networking technologies (DNS, TCP/IP, routing, etc.)
- Include practical examples when explaining technical concepts
- Draw on your knowledge of networking fundamentals, protocols, and best practices
- Explain complex networking concepts in clear, accessible language

CONTEXTUAL AWARENESS:
- Remember previous messages in the conversation to provide consistent responses
- Understand follow-up questions without requiring the user to repeat context
- If a user asks about a topic and then says "Tell me more" or asks a related question, maintain the context

SOURCE CITATION:
- When answering from provided CONTEXT, cite sources clearly and precisely using the format [Source: filename.ext, Page: X].
- Only cite the specific page from which the information was taken.
- If multiple sources/pages are used, cite each one appropriately.
- **IMPORTANT: Do NOT provide citations for basic conversational queries (greetings, identity questions) or when declining a question because it's outside your networking expertise.**

ANSWERING STRATEGY:
1. **Priority 1: Basic Conversational Queries.** For greetings (e.g., "hi", "hello"), questions about your identity ("what's your name?", "who are you?", "what can you do?"), or simple interactions ("how are you?", "nice to meet you"):
    - Respond naturally, friendly, and conversationally.
    - **Do NOT search for this information in any documents.**
    - **Do NOT provide any source citations.**
    - **Do NOT state that this information is "not found in provided documents" or similar phrases implying a document search.**
    - Your primary goal here is a natural human-like interaction.
2. **Priority 2: Networking Questions with Context.** If the query is about networking and CONTEXT is provided:
    - **CRITICAL: Base your answer EXCLUSIVELY on the provided CONTEXT. Do NOT add information from your general knowledge.**
    - **STICK CLOSELY TO THE DOCUMENTS: Only provide information that is explicitly stated or directly derivable from the context.**
    - If the context contains relevant information, use it to answer the question completely and accurately.
    - Cite sources clearly and precisely (e.g., [Source: document.pdf, Page: 5]) for all information drawn from the context.
    - **Do NOT supplement with external knowledge, even if you think it would be helpful.**
3. **Priority 3: Networking Questions without Sufficient Context.** If the query is about networking, but the CONTEXT doesn't contain the answer or no CONTEXT is provided for a networking question:
    - **FIRST: Clearly state that the specific information wasn't found in the provided documents.**
    - **THEN: Politely explain that you can only provide information based on the available documents to ensure accuracy.**
    - **Do NOT provide general networking knowledge as a fallback.**
    - **Do NOT provide citations in this case.**
4. **Priority 4: Non-Networking, Non-Conversational Queries.** If the query is not a basic conversational query (as defined in Priority 1) and not about networking:
    - Politely explain that you are specialized in networking topics and cannot assist with that specific query.
    - **Do NOT provide any source citations.**

SCOPE BOUNDARIES (General Guidelines, refer to Answering Strategy for specifics):
- Your primary role is a networking assistant. Focus on providing expert networking help.
- For basic conversational interactions (greetings, self-introduction), be friendly and natural.
- If a query is clearly outside your networking expertise (such as math problems like "1+1", general knowledge questions, weather, time, etc.) and not a simple conversational interaction, politely decline to answer and explain that you are specialized in networking topics only.
"""

# --- RAG Prompt Template ---
def get_rag_prompt() -> ChatPromptTemplate:
    """Returns the ChatPromptTemplate for the RAG chain with conversation history."""
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(SYSTEM_INSTRUCTIONS),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("""
Context information is below:
----------------
{context}
----------------

Based on the conversation history and the context information above, respond to the following message as NetSage, the intelligent network assistant:
{question}

Remember:
1. For basic greetings, identity questions, or when declining a non-networking query, respond naturally and **do NOT provide source citations**.
2. **CRITICAL FOR NETWORKING QUESTIONS: Base your answer EXCLUSIVELY on the provided context. Do NOT add information from your general knowledge.**
3. **If the context contains relevant information, use ONLY that information to answer completely and accurately.**
4. **If the context doesn't contain sufficient information for a networking question, state that the information wasn't found in the provided documents and explain that you can only provide information based on available documents.**
5. When using context for networking questions, cite sources precisely: [Source: filename.ext, Page: X].
6. Maintain a friendly, professional tone with occasional personality.
""")
    ])
    return prompt

# --- Conversation History Management ---
def format_chat_history(messages: List[Dict[str, str]]) -> List[HumanMessage | AIMessage]:
    """
    Formats a list of message dictionaries into LangChain message objects.

    Args:
        messages: List of message dictionaries with 'role' and 'content' keys

    Returns:
        List of LangChain message objects
    """
    formatted_messages = []

    for message in messages:
        if message["role"] == "user":
            formatted_messages.append(HumanMessage(content=message["content"]))
        elif message["role"] == "assistant":
            formatted_messages.append(AIMessage(content=message["content"]))

    # Limit the number of messages to prevent token overflow
    if len(formatted_messages) > MAX_HISTORY_LENGTH:
        # Keep the most recent messages
        formatted_messages = formatted_messages[-MAX_HISTORY_LENGTH:]

    return formatted_messages

# --- Direct LLM Streaming ---
async def stream_llm_response(prompt: str, chat_history: List[Dict[str, str]] = None) -> AsyncIterator[str]:
    """
    Directly streams a response from the LLM given a prompt and optional chat history.

    Args:
        prompt: The user's prompt/question
        chat_history: Optional list of previous messages in the conversation

    Yields:
        Chunks of the LLM's response as they are generated
    """
    llm = get_llm()
    if not llm:
        yield "Error: LLM not available."
        return

    # Format chat history if provided
    formatted_history = []
    if chat_history:
        formatted_history = format_chat_history(chat_history)

    # Create a system message
    system_msg = SystemMessage(content=SYSTEM_INSTRUCTIONS)

    # Check if this is a simple conversational query (greeting, identity question)
    lower_prompt = prompt.lower().strip()
    is_simple_query = any([
        lower_prompt in ["hi", "hello", "hey", "greetings", "hola"],
        "how are you" in lower_prompt,
        "what's your name" in lower_prompt,
        "who are you" in lower_prompt,
        "what can you do" in lower_prompt,
        "nice to meet you" in lower_prompt
    ])

    # For simple queries, add a hint to respond conversationally
    if is_simple_query:
        prompt = f"{prompt}\n\nNote: This is a conversational query. Respond naturally as NetSage without requiring context information."

    # Combine all messages
    messages = [system_msg] + formatted_history + [HumanMessage(content=prompt)]

    print(f"Streaming LLM response for {'conversational' if is_simple_query else 'standard'} query...")
    async for chunk in llm.astream(messages):
        yield chunk.content

# Run example if executed directly
if __name__ == "__main__":
    import asyncio

    async def run_example():
        print("Running LLM handler script directly (example mode)...")
        llm = get_llm()
        if llm:
            # Test basic response
            test_prompt = "Explain the concept of Retrieval-Augmented Generation in 2 sentences."
            print(f"\nTesting LLM stream with prompt: '{test_prompt}'")
            print("Response:")
            async for token in stream_llm_response(test_prompt):
                print(token, end="", flush=True)
            print("\n\nStream finished.")

            # Test with conversation history
            print("\n--- Testing with conversation history ---")
            test_history = [
                {"role": "user", "content": "What is RAG?"},
                {"role": "assistant", "content": "RAG (Retrieval-Augmented Generation) is a technique that combines retrieval of relevant documents with text generation."},
                {"role": "user", "content": "How does it improve AI responses?"}
            ]

            print("Chat history:")
            for msg in test_history:
                print(f"{msg['role'].upper()}: {msg['content']}")

            print("\nResponse with history:")
            async for token in stream_llm_response(test_history[-1]["content"], test_history[:-1]):
                print(token, end="", flush=True)
            print("\n\nStream with history finished.")

            # Test RAG prompt template
            prompt_template = get_rag_prompt()
            print("\n--- RAG Prompt Template Example ---")
            example_input = {
                "context": "Document A, page 5: RAG combines retrieval with generation.\nDocument B, page 1: It uses retrieved docs to inform the LLM.",
                "question": "How does RAG work?",
                "chat_history": format_chat_history([
                    {"role": "user", "content": "What is RAG?"},
                    {"role": "assistant", "content": "RAG stands for Retrieval-Augmented Generation."}
                ])
            }
            formatted_prompt = prompt_template.format_prompt(**example_input)
            print(formatted_prompt.to_string())
        else:
            print("LLM could not be initialized. Ensure API key is set.")

    asyncio.run(run_example())