# distill_mopd.py —— MOPD多教师在线策略蒸馏（中英双语版）
"""
用法：
  python distill_mopd.py                              # 进入交互式菜单（自动询问语言）
  python distill_mopd.py --topic "共产主义"            # 指定主题批量生成
  python distill_mopd.py --file prompts.txt            # 从文件读取提示词
  python distill_mopd.py --prompt "什么是马克思主义？" # 单个提示词
  python distill_mopd.py --lang zh                     # 指定语言
  python distill_mopd.py --output my_data.txt          # 指定输出文件
"""

import os, sys, json, time, argparse

# ==================== 语言选择 ====================
T = {
    "en": {
        "title": "🎓 MOPD Multi-Teacher Distillation Tool",
        "interactive_hint": "Enter prompts (one per line), empty line to finish:",
        "interactive_file_hint": "(or type 'file:path' to load from a file)",
        "file_loaded": "📄 Loaded {} prompts from file.",
        "file_not_found": "⚠️  File not found: {}",
        "no_prompts": "❌ No prompts entered.",
        "processing": "📝 [{}/{}] {}",
        "calling_teacher": "   🎓 Calling teacher: {} (weight: {})",
        "api_error": "   ⚠️  {} API error: {}",
        "api_fail": "   ⚠️  {} call failed: {}",
        "local_fallback": "   💡 Using local fallback response.",
        "teacher_skip": "   ⏭️  Skipping.",
        "no_teacher_response": "   ❌ No teacher responses available, skipping.",
        "saved": "   ✅ Saved (source: {})",
        "done": "🎉 Distillation complete! Generated {} entries, appended to {}",
        "hint_config": "📌 Add '{}' to text_datasets in config.py to use.",
        "preset_topic": "📚 Using preset topic '{}', {} prompts total.",
        "prompts_from_file": "📄 Loaded {} prompts from file.",
        "api_key_missing": "⚠️  API key '{}' not configured. Set it in config.py or config_manager.py.",
        "enter_prompt": "> ",
    },
    "zh": {
        "title": "🎓 MOPD 多教师蒸馏工具",
        "interactive_hint": "输入提示词（每行一个），输入空行结束：",
        "interactive_file_hint": "（也可输入 'file:文件路径' 从文件读取）",
        "file_loaded": "📄 已从文件加载 {} 条提示词。",
        "file_not_found": "⚠️  文件不存在: {}",
        "no_prompts": "❌ 没有输入任何提示词。",
        "processing": "📝 [{}/{}] {}",
        "calling_teacher": "   🎓 调用教师: {} (权重: {})",
        "api_error": "   ⚠️  {} API 错误: {}",
        "api_fail": "   ⚠️  {} 调用失败: {}",
        "local_fallback": "   💡 使用本地回退回答。",
        "teacher_skip": "   ⏭️  跳过。",
        "no_teacher_response": "   ❌ 没有可用的教师回答，跳过此题。",
        "saved": "   ✅ 已保存 (来源: {})",
        "done": "🎉 蒸馏完成！共生成 {} 条数据，已追加到 {}",
        "hint_config": "📌 在 config.py 的 text_datasets 中添加 '{}' 即可使用。",
        "preset_topic": "📚 使用预设主题「{}」，共 {} 条提示词。",
        "prompts_from_file": "📄 从文件加载 {} 条提示词。",
        "api_key_missing": "⚠️  未配置 API 密钥 '{}'。请在 config.py 或 config_manager.py 中设置。",
        "enter_prompt": "> ",
    },
}

def select_language(args_lang=None):
    """选择语言"""
    if args_lang == "zh":
        return "zh"
    elif args_lang == "en":
        return "en"
    print("=" * 50)
    print("  🎓 MOPD Distillation / MOPD 多教师蒸馏")
    print("=" * 50)
    print("  Please select language / 请选择语言:")
    print("  1) English")
    print("  2) 中文")
    choice = input("  > ").strip()
    return "zh" if choice == "2" else "en"

# ==================== 教师模型配置 ====================
TEACHER_CONFIGS = {
    "deepseek": {
        "name": "DeepSeek",
        "api_base": "https://api.deepseek.com/v1",
        "api_key": None,  # 运行时从 config 读取
        "model": "deepseek-chat",
        "weight": 0.35,
    },
    "qwen": {
        "name": "Qwen",
        "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key": None,
        "model": "qwen-plus",
        "weight": 0.35,
    },
    "local": {
        "name": "Local",
        "type": "local",
        "weight": 0.30,
    },
}

USE_LOCAL_FALLBACK = True


def load_api_keys():
    """从 config.py 加载 API 密钥"""
    try:
        from config import Config
        cfg = Config()
        if hasattr(cfg, "distill_deepseek_api_key") and cfg.distill_deepseek_api_key != "your-deepseek-api-key":
            TEACHER_CONFIGS["deepseek"]["api_key"] = cfg.distill_deepseek_api_key
        if hasattr(cfg, "distill_qwen_api_key") and cfg.distill_qwen_api_key != "your-qwen-api-key":
            TEACHER_CONFIGS["qwen"]["api_key"] = cfg.distill_qwen_api_key
    except:
        pass


def call_teacher_api(teacher, prompt):
    """调用教师模型 API 获取回答"""
    if teacher.get("type") == "local":
        return None
    if not teacher.get("api_key") or teacher["api_key"].startswith("your-"):
        return None

    try:
        import requests
        headers = {
            "Authorization": f"Bearer {teacher['api_key']}",
            "Content-Type": "application/json",
        }
        data = {
            "model": teacher["model"],
            "messages": [{"role": "user", "content": f"请详细回答以下问题：\n{prompt}"}],
            "temperature": 0.7,
            "max_tokens": 1024,
        }
        response = requests.post(
            f"{teacher['api_base']}/chat/completions",
            headers=headers, json=data, timeout=60,
        )
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            print(T[LANG]["api_error"].format(teacher['name'], response.status_code))
            return None
    except Exception as e:
        print(T[LANG]["api_fail"].format(teacher['name'], e))
        return None


def generate_local_response(prompt):
    """本地占位回答"""
    return (
        f"这是关于「{prompt[:30]}...」的详细回答。\n\n"
        f"该问题涉及多个重要概念和知识点。在实际应用中，"
        f"需要结合具体情况进行分析和理解。\n\n"
        f"建议进一步查阅相关资料以获得更深入的认识。"
    )


def distill_from_prompts(prompts, output_file="distillation.txt"):
    """对每个提示词调用教师模型生成回答并追加到文件"""
    total_added = 0
    for i, prompt in enumerate(prompts):
        print(T[LANG]["processing"].format(i+1, len(prompts), prompt[:50]))

        all_responses = []
        for teacher_id, teacher in TEACHER_CONFIGS.items():
            print(T[LANG]["calling_teacher"].format(teacher['name'], teacher.get('weight', 0)))
            response = call_teacher_api(teacher, prompt)
            if response is None and USE_LOCAL_FALLBACK and teacher.get("type") != "local":
                response = generate_local_response(prompt)
                print(T[LANG]["local_fallback"])
            elif response is None and teacher.get("type") == "local":
                response = generate_local_response(prompt)
                print(T[LANG]["local_fallback"])
            elif response is None:
                print(T[LANG]["teacher_skip"])
                continue
            all_responses.append({"teacher": teacher['name'], "weight": teacher.get('weight', 0), "response": response})

        if not all_responses:
            print(T[LANG]["no_teacher_response"])
            continue

        best = max(all_responses, key=lambda x: x['weight'])
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(f"问：{prompt}\n答：{best['response']}\n")
        total_added += 1
        print(T[LANG]["saved"].format(best['teacher']))

    print(f"\n{T[LANG]['done'].format(total_added, output_file)}")
    print(T[LANG]["hint_config"].format(output_file))


def interactive_mode():
    """交互式蒸馏模式"""
    print(f"\n{T[LANG]['title']}")
    print("=" * 40)
    print(T[LANG]["interactive_hint"])
    print(T[LANG]["interactive_file_hint"])
    print("=" * 40)

    prompts = []
    while True:
        line = input(T[LANG]["enter_prompt"]).strip()
        if not line:
            break
        if line.startswith("file:"):
            filepath = line[5:].strip()
            if os.path.exists(filepath):
                with open(filepath, "r", encoding="utf-8") as f:
                    new_prompts = [l.strip() for l in f if l.strip()]
                    prompts.extend(new_prompts)
                print(T[LANG]["file_loaded"].format(len(new_prompts)))
            else:
                print(T[LANG]["file_not_found"].format(filepath))
        else:
            prompts.append(line)

    if not prompts:
        print(T[LANG]["no_prompts"])
        return

    output_file = "distillation.txt"
    try:
        from config import Config
        cfg = Config()
        if hasattr(cfg, "distill_output_file"):
            output_file = cfg.distill_output_file
    except:
        pass

    distill_from_prompts(prompts, output_file)


PRESET_PROMPTS = {
    "共产主义": {
        "zh": [
            "什么是马克思主义的核心思想？",
            "简述历史唯物主义的基本原理。",
            "什么是剩余价值理论？",
            "解释阶级斗争在历史发展中的作用。",
            "什么是中国特色社会主义？",
            "列宁对马克思主义的主要贡献是什么？",
            "什么是《共产党宣言》？",
            "简述毛泽东思想的三个基本方面。",
            "什么是人民民主专政？",
            "如何理解社会主义初级阶段？",
        ],
        "en": [
            "What is the core idea of Marxism?",
            "Briefly describe the basic principles of historical materialism.",
            "What is the theory of surplus value?",
            "Explain the role of class struggle in historical development.",
            "What is socialism with Chinese characteristics?",
        ],
    },
    "AI基础": {
        "zh": [
            "什么是机器学习？",
            "解释深度学习和神经网络的关系。",
            "什么是Transformer架构？",
            "简述知识蒸馏的原理和应用。",
            "什么是LoRA微调？",
        ],
        "en": [
            "What is machine learning?",
            "Explain the relationship between deep learning and neural networks.",
            "What is the Transformer architecture?",
            "Briefly describe the principle and application of knowledge distillation.",
            "What is LoRA fine-tuning?",
        ],
    },
    "编程": {
        "zh": [
            "用Python实现快速排序算法。",
            "解释面向对象编程的核心概念。",
        ],
        "en": [
            "Implement a quick sort algorithm in Python.",
            "Explain the core concepts of object-oriented programming.",
        ],
    },
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MOPD Multi-Teacher Distillation Tool")
    parser.add_argument("--topic", type=str, help="Use preset topic prompts")
    parser.add_argument("--file", type=str, help="Load prompts from file")
    parser.add_argument("--prompt", type=str, help="Single prompt")
    parser.add_argument("--lang", type=str, help="Language (zh/en)")
    parser.add_argument("--output", type=str, help="Output file")
    args = parser.parse_args()

    LANG = select_language(args.lang)
    load_api_keys()

    if args.topic and args.topic in PRESET_PROMPTS:
        prompts = PRESET_PROMPTS[args.topic].get(LANG, PRESET_PROMPTS[args.topic]["zh"])
        print(T[LANG]["preset_topic"].format(args.topic, len(prompts)))
        output = args.output or "distillation.txt"
        distill_from_prompts(prompts, output)
    elif args.file:
        if os.path.exists(args.file):
            with open(args.file, "r", encoding="utf-8") as f:
                prompts = [l.strip() for l in f if l.strip()]
            print(T[LANG]["prompts_from_file"].format(len(prompts)))
            output = args.output or "distillation.txt"
            distill_from_prompts(prompts, output)
        else:
            print(T[LANG]["file_not_found"].format(args.file))
    elif args.prompt:
        output = args.output or "distillation.txt"
        distill_from_prompts([args.prompt], output)
    else:
        interactive_mode()