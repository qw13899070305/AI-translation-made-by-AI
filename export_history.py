import sys
import json
from datetime import datetime

def parse_conversation(text):
    lines = text.strip().split("\n")
    messages = []
    current_role = None
    current_content = []
    for line in lines:
        if line.startswith("User:") or line.startswith("You:"):
            if current_role: messages.append({"role": current_role, "content": "\n".join(current_content).strip()})
            current_role = "user"
            current_content = [line.split(":", 1)[1].strip() if ":" in line else line]
        elif line.startswith("AI:") or line.startswith("Assistant:"):
            if current_role: messages.append({"role": current_role, "content": "\n".join(current_content).strip()})
            current_role = "assistant"
            current_content = [line.split(":", 1)[1].strip() if ":" in line else line]
        else:
            if current_role: current_content.append(line)
    if current_role: messages.append({"role": current_role, "content": "\n".join(current_content).strip()})
    return messages

def export_markdown(messages, title="AI Conversation"):
    md = f"# {title}\n\nExported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n---\n\n"
    for msg in messages:
        role = "🧑 User" if msg["role"] == "user" else "🤖 AI"
        md += f"### {role}\n\n{msg['content']}\n\n"
    return md

def export_json(messages):
    return json.dumps({"export_time": datetime.now().isoformat(), "messages": messages}, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    print("📜 Conversation Export Tool")
    print("Paste conversation below, type END to finish:\n")
    lines = []
    while True:
        try: line = input()
        except EOFError: break
        if line.strip() == "END": break
        lines.append(line)
    if not lines: print("No input."); sys.exit(1)
    messages = parse_conversation("\n".join(lines))
    if not messages: print("Failed to parse."); sys.exit(1)
    print("\nExport format: 1) Markdown  2) JSON  3) Plain text")
    choice = input("> ").strip()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if choice == "1":
        with open(f"conversation_{ts}.md", "w", encoding="utf-8") as f: f.write(export_markdown(messages))
    elif choice == "2":
        with open(f"conversation_{ts}.json", "w", encoding="utf-8") as f: f.write(export_json(messages))
    elif choice == "3":
        with open(f"conversation_{ts}.txt", "w", encoding="utf-8") as f: f.write("\n".join(lines))
    else: print("Invalid choice.")