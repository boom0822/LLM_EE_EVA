# LLM_EE_EVA
测评deepseek、qwen、kimi、chatglm四个大语言模型在公开数据集DuEE上的事件抽取效果

# Requirement

langchain-core     1.4.0
langchain-openai   1.2.1
langchain-protocol 0.0.15
langsmith          0.8.3
numpy              2.4.4
openai             2.36.0
pandas             3.0.3
pip                26.1.1
tiktoken           0.12.0
tqdm               4.67.3

#项目说明
.
├── config.py                     # API Key 及文件路径配置
├── prompt.py                     # Prompt 模板（单条/批量）及 schema 格式化
├── utils.py                      # 数据加载、schema 加载、JSON 提取、触发词示例加载
├── llm_clients.py                # 各模型初始化函数（DeepSeek/Qwen/Kimi/ChatGLM）
├── evaluator.py                  # 评估函数（严格匹配 & 模糊匹配）
├── batch_processor.py            # 批量推理核心（prompt 构造、解析、清洗）
├── generate_trigger_examples.py  # 从训练集提取触发词并保存为 JSON
├── batch_main_{model}.py         # 主运行脚本（批量处理）
├── data/
│   ├── event_schema.json         # 事件类型与角色定义
│   ├── trigger_examples.json     # 触发词示例库（由 generate 脚本生成）
│   └── DuEE/
│       ├── train.json            # 实验数据
│       └── ceshi.json            # 测试数据
├── outputs/                      # 输出目录
│   ├── {model}_result.json       # 每条文本的详细预测与标注
│   ├── {model}_empty_pred.json   # 空预测样本详情
│   └── {model}_final_metrics.csv # 总体指标
└── README.md
