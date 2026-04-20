import os
from openai import OpenAI
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ResponderService:
    def __init__(self, model: str):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
    
    def respond(self, question: str, memory: str) -> dict:
        logger.info("Generating response for question: '%s' with memory \n'%s'", question, memory)
        try:
            if not memory:
                memory = "No relevant memory available."
            
            system_msg = (
                "You are an intelligent memory assistant tasked with retrieving accurate information "
                "from conversation memories.\n\n"
                "# CONTEXT:\n"
                "You have access to memories from two speakers in a conversation. These memories contain "
                "timestamped information that may be relevant to answering the question.\n\n"
                "# INSTRUCTIONS:\n"
                "1. Carefully analyze all provided memories from both speakers\n"
                "2. Pay special attention to the timestamps to determine the answer\n"
                "3. If the question asks about a specific event or fact, look for direct evidence in the memories\n"
                "4. If the memories contain contradictory information, prioritize the most recent memory\n"
                "5. If there is a question about time references (like 'last year', 'two months ago', etc.), "
                "calculate the actual date based on the memory timestamp\n"
                "6. Always convert relative time references to specific dates, months, or years based on the memory timestamp\n"
                "7. Focus only on the content of the memories. Do not confuse character names mentioned in "
                "memories with the actual users who created those memories\n"
                "8. Be generous in inferring answers: if the memory strongly implies the answer, state it\n"
                "9. The answer should be less than 5-6 words\n"
                "10. Only reply 'I don't know' if the memories contain NO information relevant to the question"
            )

            user_msg = (
                f"MEMORIES:\n{memory}\n\n"
                f"QUESTION: {question}\n\n"
                "# APPROACH (Think step by step):\n"
                "1. Examine all memories that contain information related to the question\n"
                "2. Check timestamps carefully for temporal questions\n"
                "3. If the answer requires calculation (e.g. relative time), show your reasoning briefly\n"
                "4. Formulate a precise, concise answer based solely on the evidence in the memories\n"
                "5. If no memory is relevant at all, reply exactly: \"I don't know\"\n\n"
                "Answer:"
            )

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0,
            )
            
            answer = response.choices[0].message.content.strip()
            
            return {
                "status": "success",
                "answer": answer
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }