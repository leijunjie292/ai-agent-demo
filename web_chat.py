# -*- coding: utf-8 -*-
import gradio as gr
import requests
import os
from dotenv import load_dotenv

# ------------------- 强制读取当前脚本目录下的.env（彻底解决路径问题） -------------------
# 获取当前web_chat.py所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 拼接.env完整路径
env_path = os.path.join(current_dir, ".env")
# 加载环境变量
load_dotenv(dotenv_path=env_path)

# 读取密钥、强校验
API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not API_KEY:
    raise ValueError(f"❌ 未读取到密钥！请检查：{env_path} 文件是否存在、变量名是否为 DASHSCOPE_API_KEY")

# 通义千问API基础配置
API_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}
MODEL = "qwen-turbo"
SYSTEM_PROMPT = {"role": "system", "content": "你是专业、简洁、友好的AI助手，支持上下文记忆对话"}

# 全局保存对话历史
conversation_history = []

def chat_response(user_input):
    global conversation_history
    # 组装完整上下文
    messages = [SYSTEM_PROMPT]
    for item in conversation_history:
        messages.append({"role": "user", "content": item["user"]})
        messages.append({"role": "assistant", "content": item["ai"]})
    messages.append({"role": "user", "content": user_input})

    # 请求体
    payload = {
        "model": MODEL,
        "input": {"messages": messages},
        "parameters": {"result_format": "message"}
    }

    try:
        res = requests.post(API_URL, headers=HEADERS, json=payload, timeout=30)
        res.raise_for_status()
        result = res.json()
        answer = result["output"]["choices"][0]["message"]["content"]
        # 保存历史
        conversation_history.append({"user": user_input, "ai": answer})
        # 格式化输出
        display = ""
        for item in conversation_history:
            display += f"你：{item['user']}\n\nAI：{item['ai']}\n\n——————————\n\n"
        return display
    except Exception as e:
        return f"请求失败：{str(e)}"

def clear_chat():
    global conversation_history
    conversation_history = []
    return ""

# Gradio界面
with gr.Blocks(title="通义千问聊天助手") as demo:
    gr.Markdown("# 🤖 通义千问网页聊天助手（记忆版）")
    gr.Markdown("基于qwen-turbo，支持上下文记忆、一键清空")

    user_msg = gr.Textbox(label="输入问题", placeholder="在这里输入你的问题...")
    chat_box = gr.Textbox(label="对话记录", lines=16)

    with gr.Row():
        send_btn = gr.Button("发送", variant="primary")
        clear_btn = gr.Button("清空对话", variant="secondary")

    send_btn.click(chat_response, inputs=user_msg, outputs=chat_box)
    clear_btn.click(clear_chat, inputs=None, outputs=chat_box)

if __name__ == "__main__":
    print("✅ 启动成功！正在打开浏览器...")
    demo.launch(inbrowser=True, share=False)