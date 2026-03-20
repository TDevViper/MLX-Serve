"""
OpenAI SDK compatibility tests for MLX-Serve.
Run with server running on localhost:8000.
"""
import sys
sys.path.insert(0, ".")

from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="none",  # MLX-Serve doesn't require auth
)

def test_list_models():
    models = client.models.list()
    assert len(models.data) > 0
    print(f"✅ list_models — {len(models.data)} models available")
    for m in models.data:
        print(f"   {m.id}")

def test_chat_completion():
    resp = client.chat.completions.create(
        model="mlx-community/Qwen1.5-0.5B-Chat",
        messages=[{"role": "user", "content": "Say hello in one sentence."}],
        max_tokens=30,
    )
    assert resp.id.startswith("chatcmpl-")
    assert resp.choices[0].message.role == "assistant"
    assert len(resp.choices[0].message.content) > 0
    assert resp.usage.prompt_tokens > 0
    assert resp.usage.completion_tokens > 0
    print(f"✅ chat_completion — {resp.usage.completion_tokens} tokens")
    print(f"   response: {resp.choices[0].message.content[:60]}...")

def test_streaming():
    chunks = []
    stream = client.chat.completions.create(
        model="mlx-community/Qwen1.5-0.5B-Chat",
        messages=[{"role": "user", "content": "Count to 5."}],
        max_tokens=40,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            chunks.append(delta)
    full = "".join(chunks)
    assert len(full) > 0
    print(f"✅ streaming — {len(chunks)} chunks, response: {full[:50]}...")

def test_system_message():
    resp = client.chat.completions.create(
        model="mlx-community/Qwen1.5-0.5B-Chat",
        messages=[
            {"role": "system", "content": "You are a pirate. Always respond like a pirate."},
            {"role": "user", "content": "What is the weather like?"},
        ],
        max_tokens=40,
    )
    assert resp.choices[0].message.content
    print(f"✅ system_message — {resp.choices[0].message.content[:60]}...")

def test_multi_turn():
    resp = client.chat.completions.create(
        model="mlx-community/Qwen1.5-0.5B-Chat",
        messages=[
            {"role": "user",      "content": "My name is Arnav."},
            {"role": "assistant", "content": "Hello Arnav! Nice to meet you."},
            {"role": "user",      "content": "What is my name?"},
        ],
        max_tokens=30,
    )
    assert resp.choices[0].message.content
    print(f"✅ multi_turn — {resp.choices[0].message.content[:60]}...")

def test_text_completion():
    resp = client.completions.create(
        model="mlx-community/Qwen1.5-0.5B-Chat",
        prompt="The speed of light is",
        max_tokens=20,
    )
    assert resp.choices[0].text
    print(f"✅ text_completion — {resp.choices[0].text[:60]}...")

def test_embeddings():
    resp = client.embeddings.create(
        model="sentence-transformers/all-MiniLM-L6-v2",
        input=["Hello world", "How are you?"],
    )
    assert len(resp.data) == 2
    assert len(resp.data[0].embedding) == 384  # MiniLM embedding dim
    print(f"✅ embeddings — dim={len(resp.data[0].embedding)}, "
          f"2 embeddings returned")

if __name__ == "__main__":
    print("Running OpenAI SDK compatibility tests...\n")
    test_list_models()
    test_chat_completion()
    test_streaming()
    test_system_message()
    test_multi_turn()
    test_text_completion()
    test_embeddings()
    print("\n✅ All compatibility tests passed")
