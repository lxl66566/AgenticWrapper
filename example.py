import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, List

from ollama import AsyncClient

from AgenticWrapper import Agent

client = AsyncClient()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


# 一个模拟的 LLM 交互函数
async def mock_llm_func(messages: List[Dict[str, str]]) -> str:
    response = await client.chat(
        model="rhkrdfl/qwq:7b-q4",
        messages=messages,
    )
    return response.message.content or "LLM response is empty."


# 一个模拟的 LLM 交互函数，带有 temperature 参数
async def mock_llm_func_with_temperature(
    messages: List[Dict[str, str]], temperature: float
) -> str:
    response = await client.chat(
        model="rhkrdfl/qwq:7b-q4",
        messages=messages,
        options={"temperature": temperature},
    )
    return response.message.content or "LLM response is empty."


@dataclass
class NounTag:
    tags: List[str]


async def main():
    agent = Agent(mock_llm_func)
    response = await agent.query("你好")
    print(response)

    agent = Agent(mock_llm_func)
    response = await agent.query(
        "请你为“金字塔”赋予两到三个 tag", structured_output_type=NounTag
    )
    print(response)

    async def calculator(expression: str) -> str:
        """一个简单的计算器工具，接受 Python 数学表达式并返回结果"""
        try:
            return str(eval(expression))
        except Exception as e:
            return f"计算错误: {str(e)}"

    agent = Agent(mock_llm_func, tools=[calculator])
    response = await agent.query(
        "请你调用工具为我计算 1.056^9 * 18765 / 123，其中除法为整除"
    )
    print(response)

    # with temperature and other kwargs

    agent = Agent(mock_llm_func_with_temperature, default_temperature=0.4)
    response = await agent.query(
        """这是带有 temperature 参数的 LLM 交互函数测试。
        请注意，query 的 temperature 参数会覆盖 agent 的默认 temperature 参数""",
        temperature=0.5,
    )
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
