import os
import random
from typing import Dict
from mem0 import Memory
from datetime import datetime
from openai import OpenAI

from dotenv import load_dotenv

load_dotenv()


class SupportChatbot:
    def __init__(self):
        self.config = {
            "llm": {
                "provider": "openai",
                "config": {
                    "model": "qwen-max-latest",
                    "max_tokens": 3000,
                    "openai_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                },
            },

            "vector_store": {
                "provider": "tair_vector",
                "config": {
                    "collection_name": "mem0_support_chatbot",
                    "distance_method": "COSINE",
                    "embedding_model_dims": 1536,
                    "host": os.environ["TAIR_VECTOR_HOST"],
                    "port": 6379,
                    "username": os.environ["TAIR_VECTOR_USERNAME"],
                    "password": os.environ["TAIR_VECTOR_PASSWORD"],
                    "db": os.environ["TAIR_VECTOR_DB"],
                }
            },

            "embedder": {
                "provider": "openai",
                "config": {
                    "model": "text-embedding-v4",
                    "embedding_dims": 1536,
                    "openai_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                }
            }
        }

        self.client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

        self.memory = Memory.from_config(self.config)
        self.short_term_memory = []
        self.max_short_term_memory = 10

        # Define support context
        self.system_context = """
        You are a helpful assistant agent. Use the following guidelines:
        - Reference past interactions when relevant
        - Maintain consistent information across conversations
        - If you're unsure about something, ask for clarification
        - Keep track of open issues and follow-ups
        """

    def store_customer_interaction(self, user_id: str, message: str, response: str, metadata: Dict = None):
        """Store customer interaction in memory."""
        if metadata is None:
            metadata = {}

        # Add timestamp to metadata
        metadata["timestamp"] = datetime.now().isoformat()

        # Format conversation for storage
        conversation = [{"role": "user", "content": message}, {"role": "assistant", "content": response}]

        # Store in Mem0
        self.memory.add(conversation, user_id=user_id, metadata=metadata)

    def get_relevant_history(self, user_id: str, query: str) -> Dict:
        """Retrieve relevant past interactions."""
        return self.memory.search(
            query=query,
            user_id=user_id,
            limit=5,  # Adjust based on needs
        )

    def handle_customer_query(self, user_id: str, query: str) -> str:
        """Process customer query with context from past interactions."""
        # Get relevant past interactions
        relevant_history = self.get_relevant_history(user_id, query).get("results", [])

        # Build context from relevant history
        context = "Previous relevant interactions:\n"
        if relevant_history:
            for memory in relevant_history:
                context += f"{memory['memory']}\n"
                context += "---\n"
        else:
            context += "No relevant previous interactions found.\n"

            # 添加短期记忆
            context += "\nRecent conversation:\n"
            if self.short_term_memory:
                for interaction in self.short_term_memory:
                    context += f"Customer: {interaction['query']}\n"
                    context += f"Support: {interaction['response']}\n"
                    context += "---\n"
            else:
                context += "No recent interactions in short-term memory.\n"

        # Prepare prompt with context and current query
        prompt = f"""
        {self.system_context}

        {context}

        Current customer query: {query}

        Provide a helpful response that takes into account any relevant past interactions.
        """

        # Generate response using Claude with streaming
        response_content = ""
        for chunk in self.client.chat.completions.create(
                model="qwen-max-latest",
                messages=[{"role": "user", "content": prompt}],
                stream=True
        ):
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                print(content, end='', flush=True)
                response_content += content

        print()  # New line after streaming completes

        # Store interaction
        self.store_customer_interaction(
            user_id=user_id, message=query, response=response_content,
            metadata={"type": random.choice(["support_query", "support_response", "llm_output"])}
        )
        # Store dialog
        self.short_term_memory.append({"query": query, "response": response_content})
        if len(self.short_term_memory) > self.max_short_term_memory:
            self.short_term_memory.pop(0)
        return response_content


if __name__ == "__main__":
    chatbot = SupportChatbot()
    user_id = "first_user_001234"
    print("Welcome to Customer Support! Type 'exit' to end the conversation.")

    while True:
        # Get user input
        print("Customer:", end=' ', flush=True)
        query = input()

        # Check if user wants to exit
        if query.lower() == "exit":
            print("Thank you for using our support service. Goodbye!")
            break

        # Handle the query and print the response
        print("Support:", end=' ', flush=True)
        chatbot.handle_customer_query(user_id, query)
        print("\n")
