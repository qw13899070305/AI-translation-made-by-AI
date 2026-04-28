import sys
import json
from duckduckgo_search import DDGS

def search(query, max_results=5, region="cn-zh"):
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results, region=region))
    return [{"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")} for r in results]

def format_results(results):
    if not results:
        return "No results found."
    formatted = []
    for i, r in enumerate(results, 1):
        formatted.append(f"{i}. {r['title']}\n   {r['snippet']}\n   {r['url']}")
    return "\n\n".join(formatted)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("Enter search query: ")
    print(f"Searching for: {query}\n")
    try:
        results = search(query)
        print(format_results(results))
    except Exception as e:
        print(f"Search failed: {e}")
    if "--json" in sys.argv:
        with open("search_result.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print("\nResults saved to search_result.json")