from dataclasses import dataclass
from typing import List

import pytest

from AgenticWrapper import Agent


# 模拟 LLM 函数（用于测试）
async def mock_llm_func(messages: List[dict[str, str]]) -> str:
    """模拟 LLM 响应，用于测试"""
    last_message = messages[-1]["content"]

    # 模拟工具调用响应
    if last_message.strip().startswith("计算 "):
        return f'{{"tool_name": "calculator", "arguments": {{"expression": "{last_message.strip().removeprefix("计算 ")}"}}}}'

    # 模拟结构化输出响应
    if "TestResponse" in last_message:
        return '{"message": "测试消息", "confidence": 0.95, "tags": ["test", "demo"]}'
    if "SimpleResponse" in last_message:
        return '{"result": "测试消息"}'

    # 模拟普通对话响应
    return f"FINISH\n这是对 '{last_message}' 的模拟回复"


# 测试用的数据类
@dataclass
class TestResponse:
    message: str
    confidence: float
    tags: List[str]


@dataclass
class SimpleResponse:
    result: str


# 模拟工具函数
async def calculator(expression: str) -> str:
    """一个简单的计算器工具，接受数学表达式并返回结果"""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"计算错误: {str(e)}"


async def echo(text: str) -> str:
    """简单的回显工具，返回输入的文本"""
    return f"回显: {text}"


@pytest.mark.asyncio
async def test_basic_conversation():
    """测试基本对话功能"""
    agent = Agent(mock_llm_func)
    response = await agent.query("你好")
    assert isinstance(response, str)
    assert "你好" in response or "回复" in response


@pytest.mark.asyncio
async def test_tool_usage():
    """测试工具调用功能"""
    agent = Agent(mock_llm_func, tools=[calculator, echo])

    # 测试计算器工具
    response = await agent.query("计算 1+1")
    assert "2" in str(response)

    # 测试回显工具
    response = await agent.query("echo 测试文本")
    assert "测试文本" in str(response)


@pytest.mark.asyncio
async def test_structured_output():
    """测试结构化输出功能"""
    agent = Agent(mock_llm_func, multiturn_mode=False)

    # 测试复杂结构化输出
    response = await agent.query(
        "返回 TestResponse", structured_output_type=TestResponse
    )
    assert isinstance(response, TestResponse)
    assert isinstance(response.confidence, float)
    assert isinstance(response.tags, list)

    # 测试简单结构化输出
    response = await agent.query(
        "返回 SimpleResponse", structured_output_type=SimpleResponse
    )
    assert isinstance(response, SimpleResponse)
    assert hasattr(response, "result")


@pytest.mark.asyncio
async def test_memory_management():
    """测试记忆管理功能"""
    agent = Agent(mock_llm_func)

    # 测试记忆累积
    await agent.query("第一条消息")
    await agent.query("第二条消息")
    assert len(agent.memory) > 2  # 包含系统提示和对话历史

    # 测试记忆清除
    agent.clear_memory()
    assert len(agent.memory) == len(agent.initial_prompt)


@pytest.mark.asyncio
async def test_error_handling():
    """测试错误处理"""
    agent = Agent(mock_llm_func)

    # 测试无效的结构化输出类型
    with pytest.raises(ValueError):
        await agent.query("测试", structured_output_type=str)  # str 不是 dataclass

    # 测试工具执行错误
    agent = Agent(mock_llm_func, tools=[calculator])
    response = await agent.query("计算 1/0")
    assert "错误" in str(response)


@pytest.mark.asyncio
async def test_max_iterations():
    """测试最大迭代次数限制"""
    agent = Agent(mock_llm_func, max_iterations=2)
    response = await agent.query("测试迭代")
    assert isinstance(response, str)


if __name__ == "__main__":
    # import logging

    # logging.basicConfig(level=logging.DEBUG)
    pytest.main([__file__])
