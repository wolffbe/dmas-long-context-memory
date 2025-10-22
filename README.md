# A comparison of the accuracy and cost of long-context vector versus graph memory in distributed LLM-based multi-agent systems

## Chat

# Start container - it will auto-download locomo10.json
docker-compose up --build chat

# Check health
curl http://localhost:8002/health

# Get statistics
curl http://localhost:8002/stats

# Get all conversations (first 10)
curl http://localhost:8002/conversations

# Get specific conversation by sample_id
curl http://localhost:8002/conversations/h_0

# Get conversation by index
curl http://localhost:8002/conversations/index/0

# Get sessions for specific conversation
curl http://localhost:8002/sessions/h_0

# Get all questions
curl http://localhost:8002/questions

# Get questions for specific conversation
curl http://localhost:8002/questions/h_0

# Get questions by category
curl "http://localhost:8002/questions?category=single-hop"

# Python example
import requests

base_url = "http://localhost:8002"

# Get stats
stats = requests.get(f"{base_url}/stats").json()
print(f"Loaded {stats['total_conversations']} conversations")
print(f"Total sessions: {stats['total_sessions']}")
print(f"Total questions: {stats['total_questions']}")

# Get first conversation
conv = requests.get(f"{base_url}/conversations/index/0").json()
print(f"Conversation between {conv['speaker_a']} and {conv['speaker_b']}")
print(f"Sessions: {list(conv['sessions'].keys())}")

# Get all sessions for this conversation
sessions = requests.get(f"{base_url}/sessions/{conv['sample_id']}").json()
print(f"Found {sessions['total_sessions']} sessions")

# Get questions
questions = requests.get(f"{base_url}/questions/{conv['sample_id']}").json()
print(f"Found {questions['total_questions']} questions")
for q in questions['questions']:
    print(f"Q: {q['question']}")
    print(f"A: {q['answer']}")
    print(f"Category: {q['category']}")s