from memory import LongTermMemory

print("🧠 Long-term Memory Test")
m = LongTermMemory()
m.add("I love Sichuan cuisine", "Got it, you love Sichuan food!")
m.add("I live in Beijing", "Remembered, you live in Beijing.")

query = "Restaurant recommendation"
results = m.recall(query, k=2)
print(f"Query: 「{query}」")
for i, doc in enumerate(results, 1):
    print(f"{i}. {doc}")