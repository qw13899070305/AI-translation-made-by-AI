from memory import LongTermMemory

print("🧠 长期记忆独立测试")
m = LongTermMemory()

# 示例数据
m.add("我喜欢吃川菜", "好的，已经记住你喜欢川菜了")
m.add("我住在北京", "记住了，你在北京")

query = "推荐餐厅"
results = m.recall(query, k=2)
print(f"查询：「{query}」")
for i, doc in enumerate(results, 1):
    print(f"{i}. {doc}")