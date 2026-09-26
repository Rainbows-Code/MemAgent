import logging
from typing import Dict, List, Any, Optional
from app.llm import LLMClient, llm_client
from app.config import settings

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "你是一个智能助手。请根据用户的问题以及提供的历史对话上下文，精准回答用户的问题。"
)


class BaselineAgent:
    """
    无高级记忆系统的基线 Agent (Full Context Baseline)
    只在内存中保留最近 N 轮原始对话上下文，并直接拼进 Prompt。
    用于后续离线评估的对照组。
    """

    def __init__(
        self,
        user_id: str = "default_user",
        window_size: int = settings.WORKING_MEMORY_WINDOW_SIZE,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        llm: Optional[LLMClient] = None,
    ):
        self.user_id = user_id
        self.window_size = window_size  # 轮数，1轮 = 1条user + 1条assistant
        self.system_prompt = system_prompt
        self.llm = llm or llm_client
        self.history: List[Dict[str, str]] = []

    def get_context_messages(self) -> List[Dict[str, str]]:
        """
        获取拼接后的 Prompt 上下文
        裁剪并只保留最近 window_size 轮对话 (最大 2 * window_size 条消息)
        """
        max_messages = self.window_size * 2
        recent_history = self.history[-max_messages:] if len(self.history) > max_messages else self.history

        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(recent_history)
        return messages

    def chat(self, user_input: str) -> Dict[str, Any]:
        """
        发送消息并接收 LLM 回复

        :param user_input: 用户本轮输入
        :return: {
            "response": str,
            "usage": dict,
            "context_messages": list,
            "history_count": int
        }
        """
        # 1. 记录用户消息
        self.history.append({"role": "user", "content": user_input})

        # 2. 获取包含 System Prompt + 裁剪历史的消息列表
        messages = self.get_context_messages()

        # 3. 调用 LLM
        llm_result = self.llm.chat_completion(messages=messages)
        assistant_response = llm_result["content"]
        usage = llm_result["usage"]

        # 4. 记录模型回复
        self.history.append({"role": "assistant", "content": assistant_response})

        return {
            "response": assistant_response,
            "usage": usage,
            "context_messages": messages,
            "history_count": len(self.history),
        }

    async def achat(self, user_input: str) -> Dict[str, Any]:
        """
        异步发送消息接口
        """
        self.history.append({"role": "user", "content": user_input})
        messages = self.get_context_messages()

        llm_result = await self.llm.achat_completion(messages=messages)
        assistant_response = llm_result["content"]
        usage = llm_result["usage"]

        self.history.append({"role": "assistant", "content": assistant_response})

        return {
            "response": assistant_response,
            "usage": usage,
            "context_messages": messages,
            "history_count": len(self.history),
        }

    def clear_history(self) -> None:
        """清空内存中的历史记录"""
        self.history.clear()
