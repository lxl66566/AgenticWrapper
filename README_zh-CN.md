# AgenticWrapper

[English](README.md) | 简体中文

AgenticWrapper 是一个极为轻量级的 Python Agent 框架，它可以包装你的单个自定义 LLM 函数，使其拥有 Agent 功能，包括**工具调用、记忆管理、结构化输出**等。

AgenticWrapper 没有任何运行时依赖。

## 快速开始

### 安装

```bash
pip install AgenticWrapper
```

### 基础使用

```py
from AgenticWrapper import Agent

async def llm_func(messages: list[dict[str, str]]) -> str:
    # 假设这是你的自定义 LLM 函数
    return "你好，我是 AgenticWrapper。"

agent = Agent(llm_func)
response = await agent.query("你好")
print(response)
agent.clear_memory()
```

### 结构化输出

使用 Python dataclass 定义输出结构：

```python
@dataclass
class SearchResult:
    query: str
    results: List[str]
    total_count: int

response = await agent.query("搜索相关内容", structured_output_type=SearchResult)
assert isinstance(response, SearchResult)
```

### 工具调用

定义工具函数，最好带有函数文档：

```python
async def get_weather(location: str) -> str:
    """获取天气信息"""
    # 模拟 API 调用
    await asyncio.sleep(0.1)
    if location.lower() == "london":
        return "天气晴朗，气温为 15°C。"
    elif location.lower() == "paris":
        return "天气晴朗，气温为 18°C。"
    else:
        return f"我不知道 {location} 的天气情况。"
```

在 Agent 中使用工具：

```python
agent = Agent(llm_interaction_func=mock_llm_func, tools=[get_weather])
response = await agent.query("查询 london 的天气")
print(response)
```

工具函数的参数类型和返回类型必须是 `str`。如果你的工具函数拥有复杂的参数，你需要定义一个新的 wrapper 函数进行参数的序列化/反序列化，并将其格式告知 LLM。

### 更多示例

更多示例可以在 [example.py](example.py) 中找到。
