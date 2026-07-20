import os
import json
import urllib.request
import urllib.error
import urllib.parse
from utils.logger_handler import logger
from langchain_core.tools import tool
from rag.rag_service import RagSummarizeService
import random

rag = RagSummarizeService()

# 高德地图 Web 服务 API Key（从环境变量读取，未设置时天气/定位功能降级）
AMAP_API_KEY = os.environ.get("AMAP_API_KEY", "ce34f5ed99c414fc11a9a345c98ff79f")
AMAP_IP_URL = "https://restapi.amap.com/v3/ip"
AMAP_WEATHER_URL = "https://restapi.amap.com/v3/weather/weatherInfo"


def _http_get(url: str) -> dict:
    """发送 HTTP GET 请求，返回解析后的 JSON"""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ZhiSaoTong-Agent/2.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        logger.error(f"[HTTP] {url} 返回 HTTP {e.code}")
        return {}
    except Exception as e:
        logger.error(f"[HTTP] {url} 请求失败: {str(e)}")
        return {}


@tool(description="从向量存储中检索参考资料")
def rag_summarize(query: str) -> str:
    return rag.rag_summarize(query)


@tool(description="获取指定城市的天气，以消息字符串的形式返回")
def get_weather(city: str) -> str:
    """通过高德天气 API 获取指定城市的实时天气"""
    if not AMAP_API_KEY:
        return f"天气服务未配置（缺少 AMAP_API_KEY），无法获取{city}的天气信息"

    city_encoded = urllib.parse.quote(city)
    url = f"{AMAP_WEATHER_URL}?key={AMAP_API_KEY}&city={city_encoded}&extensions=base"
    data = _http_get(url)

    if data.get("status") != "1" or not data.get("lives"):
        logger.warning(f"[get_weather] 高德天气 API 返回异常: {data}")
        return f"未能获取{city}的天气信息"

    live = data["lives"][0]
    weather = live.get("weather", "未知")
    temperature = live.get("temperature", "未知")
    humidity = live.get("humidity", "未知")
    winddirection = live.get("winddirection", "未知")
    windpower = live.get("windpower", "未知")
    reporttime = live.get("reporttime", "")

    result = (
        f"城市{city}天气情况（更新时间{reporttime}）："
        f"天气{weather}，气温{temperature}摄氏度，"
        f"空气湿度{humidity}%，{winddirection}风{windpower}级。"
    )

    logger.info(f"[get_weather] {city}: {weather} {temperature}°C 湿度{humidity}%")
    return result


@tool(description="获取用户所在城市的名称，以纯字符串形式返回")
def get_user_location() -> str:
    """通过高德 IP 定位 API 获取当前用户的所在城市"""
    if not AMAP_API_KEY:
        fallback = random.choice(["深圳", "北京", "上海", "杭州", "广州"])
        logger.info(f"[get_user_location] 未配置 AMAP_API_KEY，兜底返回: {fallback}")
        return fallback

    url = f"{AMAP_IP_URL}?key={AMAP_API_KEY}"
    data = _http_get(url)

    if data.get("status") != "1":
        logger.warning(f"[get_user_location] 高德 IP 定位 API 返回异常: {data}")
        fallback = random.choice(["深圳", "北京", "上海", "杭州", "广州"])
        logger.info(f"[get_user_location] 兜底返回: {fallback}")
        return fallback

    city = data.get("city", "")
    province = data.get("province", "")

    if not city:
        city = province or "未知城市"

    logger.info(f"[get_user_location] IP定位结果: {province} {city}")
    return city


@tool(description="检索历史对话记忆和用户偏好。当用户提到之前聊过的内容、询问与自己相关的问题、或需要参考历史对话时调用")
def memory_recall(query: str) -> str:
    """
    长期记忆检索工具：Agent 主动调用以回忆历史信息

    Args:
        query: 检索查询（关注用户要说的话中的话题相关性）

    Returns:
        相关的历史记忆摘要
    """
    try:
        from memory.manager import MemoryManager
        import asyncio
        import concurrent.futures

        mgr = MemoryManager.get_instance()
        long_term = mgr.long_term
        if long_term is None:
            return "记忆系统未初始化"

        async def _retrieve():
            return await long_term.retrieve_relevant(query)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # 没有运行中的 event loop，直接运行
            memories = asyncio.run(_retrieve())
        else:
            # 有运行中的 event loop，在独立线程中运行避免冲突
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, _retrieve())
                memories = future.result(timeout=10)

        if not memories:
            return "未找到相关历史记忆"

        return long_term.format_for_prompt(memories)
    except Exception as e:
        from utils.logger_handler import logger
        logger.warning(f"[memory_recall] 检索失败: {str(e)}")
        return f"记忆检索失败: {str(e)}"
