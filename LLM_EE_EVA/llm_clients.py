from langchain_openai import ChatOpenAI
from config import DEEPSEEK_API_KEY, QWEN_API_KEY, KIMI_API_KEY, CHATGLM_API_KEY

def get_deepseek():
    return ChatOpenAI(
        model="deepseek-chat",
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com",
        temperature=0
    )

def get_qwen():
    return ChatOpenAI(
        model="qwen3.5-plus",
        api_key=QWEN_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        extra_body={"enable_thinking": False},
        temperature=0
    )

def get_kimi():
    return ChatOpenAI(
        model="kimi-k2.5",
        api_key=KIMI_API_KEY,
        base_url="https://api.moonshot.cn/v1",
        temperature=0.6,
        extra_body={"thinking": {"type": "disabled"}} 
    )

def get_chatglm():
    return ChatOpenAI(
        model="glm-5",
        api_key=CHATGLM_API_KEY,
        base_url="https://open.bigmodel.cn/api/paas/v4",
        temperature=0,
        extra_body={"thinking": {"type": "disabled"}}
    )