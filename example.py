import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Dict, List

import litellm
from ollama import AsyncClient

from AgenticWrapper import Agent

client = AsyncClient()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
# disable debug logging for other libs
logging.getLogger("LiteLLM").setLevel(logging.CRITICAL)
logging.getLogger("httpcore").setLevel(logging.CRITICAL)
logging.getLogger("httpx").setLevel(logging.CRITICAL)


# ollama example
async def mock_llm_func_ollama(messages: List[Dict[str, str]]) -> str:
    response = await client.chat(
        model="rhkrdfl/qwq:7b-q4",
        messages=messages,
    )
    return response["message"]["content"] or "LLM response is empty."


# litellm example
async def mock_llm_func_litellm(messages: List[Dict[str, str]]) -> str:
    # fill in your api key here
    os.environ["GEMINI_API_KEY"] = ""

    response = await litellm.acompletion(
        model="gemini/gemini-2.5-flash-preview-04-17",
        messages=messages,
    )
    return response.choices[0].message.content or "LLM response is empty."  # type: ignore


# example of how to specify custom temperature and other options
async def mock_llm_func_with_temperature(
    messages: List[Dict[str, str]], temperature: float
) -> str:
    response = await client.chat(
        model="rhkrdfl/qwq:7b-q4",
        messages=messages,
        options={"temperature": temperature},
    )
    return response.message.content or "LLM response is empty."


# structured output example
@dataclass
class NounTag:
    tags: List[str]


async def main():
    # 1. simple usage
    agent = Agent(mock_llm_func_ollama)
    response = await agent.query("你好")
    print(response)

    # 2. tool calling
    async def calculator(expression: str) -> str:
        """一个简单的计算器工具，接受 Python 数学表达式并返回结果"""
        try:
            return str(eval(expression))
        except Exception as e:
            return f"计算错误: {str(e)}"

    agent = Agent(mock_llm_func_litellm, tools=[calculator])
    response = await agent.query(
        "请你调用工具为我计算 1.056^9 * 18765 / 123，其中除法为整除"
    )
    print(response)  # 249

    # 3. tool parameters can be any json type, but the return value must be str
    async def multiply(a: int, b: int) -> str:
        return str(a * b)

    agent = Agent(mock_llm_func_litellm, tools=[multiply])

    response = await agent.query("请你用工具计算 999954 * 12325")
    print(response)

    # 4. with temperature and other kwargs
    agent = Agent(mock_llm_func_with_temperature, default_temperature=0.4)
    response = await agent.query(
        """这是带有 temperature 参数的 LLM 交互函数测试。
        请注意，query 的 temperature 参数会覆盖 agent 的默认 temperature 参数""",
        temperature=0.5,
    )
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
