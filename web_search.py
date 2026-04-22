# web_search.py —— 独立联网搜索工具
import sys
import json
from duckduckgo_search import DDGS

def search(query, max_results=5, region="cn-zh"):
    """
    执行联网搜索，返回结果列表。
    每条结果包含：title, url, snippet
    """
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results, region=region))
    return [
        {
            "title": r.get("title", ""),
            "url": r.get("href", ""),
            "snippet": r.get("body", "")
        }
        for r in results
    ]

def format_results(results):
    """将搜索结果格式化为易读的文本"""
    if not results:
        return "未找到相关结果。"
    formatted = []
    for i, r in enumerate(results, 1):
        formatted.append(f"{i}. {r['title']}\n   {r['snippet']}\n   🔗 {r['url']}")
    return "\n\n".join(formatted)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("请输入搜索关键词: ")

    print(f"🔍 正在搜索: {query}\n")
    try:
        results = search(query)
        print(format_results(results))
    except Exception as e:
        print(f"搜索失败: {e}")

    # 可选：将结果保存为 JSON 供其他程序读取
    if "--json" in sys.argv:
        with open("search_result.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print("\n📁 结果已保存到 search_result.json")