# export_history.py —— 对话历史导出工具（独立）
import sys
import os
import json
from datetime import datetime

def parse_conversation(text):
    """解析原始对话文本，提取用户和AI的发言"""
    lines = text.strip().split("\n")
    messages = []
    current_role = None
    current_content = []

    for line in lines:
        if line.startswith("用户:") or line.startswith("你:"):
            if current_role:
                messages.append({"role": current_role, "content": "\n".join(current_content).strip()})
            current_role = "user"
            current_content = [line.split(":", 1)[1].strip() if ":" in line else line]
        elif line.startswith("AI:") or line.startswith("助手:"):
            if current_role:
                messages.append({"role": current_role, "content": "\n".join(current_content).strip()})
            current_role = "assistant"
            current_content = [line.split(":", 1)[1].strip() if ":" in line else line]
        else:
            if current_role:
                current_content.append(line)

    if current_role:
        messages.append({"role": current_role, "content": "\n".join(current_content).strip()})
    return messages

def export_markdown(messages, title="AI 对话记录"):
    """将消息列表转换为 Markdown 格式"""
    md = f"# {title}\n\n"
    md += f"导出时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    md += "---\n\n"
    for msg in messages:
        role = "🧑 用户" if msg["role"] == "user" else "🤖 AI"
        md += f"### {role}\n\n"
        md += f"{msg['content']}\n\n"
    return md

def export_json(messages):
    """导出为 JSON 格式"""
    return json.dumps({
        "export_time": datetime.now().isoformat(),
        "messages": messages
    }, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    print("📜 对话历史导出工具")
    print("请将对话内容粘贴到下方，输入 END 结束：\n")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "END":
            break
        lines.append(line)

    if not lines:
        print("❌ 没有输入任何内容。")
        sys.exit(1)

    raw_text = "\n".join(lines)
    messages = parse_conversation(raw_text)

    if not messages:
        print("❌ 未能解析出对话内容。")
        sys.exit(1)

    print("\n选择导出格式:")
    print("1) Markdown (推荐)")
    print("2) JSON")
    print("3) 纯文本")
    choice = input("输入数字 (1-3): ").strip()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if choice == "1":
        filename = f"conversation_{timestamp}.md"
        content = export_markdown(messages)
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"✅ 已导出为 {filename}")
    elif choice == "2":
        filename = f"conversation_{timestamp}.json"
        content = export_json(messages)
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"✅ 已导出为 {filename}")
    elif choice == "3":
        filename = f"conversation_{timestamp}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(raw_text)
        print(f"✅ 已导出为 {filename}")
    else:
        print("❌ 无效选择。")