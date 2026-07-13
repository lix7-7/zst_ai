import os
import json
import urllib.request
import urllib.error
import urllib.parse
from utils.logger_handler import logger
from langchain_core.tools import tool
from rag.rag_service import RagSummarizeService
import random
from utils.config_handler import agent_conf
from utils.path_tool import get_abs_path

rag = RagSummarizeService()

# 高德地图 Web 服务 API Key（优先读环境变量）
AMAP_API_KEY = os.environ.get("AMAP_API_KEY", "")
AMAP_IP_URL = "https://restapi.amap.com/v3/ip"
AMAP_WEATHER_URL = "https://restapi.amap.com/v3/weather/weatherInfo"

user_ids = ["1001", "1002", "1003", "1004", "1005", "1006", "1007", "1008", "1009", "1010",]
month_arr = ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06",
             "2025-07", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12", ]

external_data = {}


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
    url = f"{AMAP_IP_URL}?key={AMAP_API_KEY}"
    data = _http_get(url)

    if data.get("status") != "1":
        logger.warning(f"[get_user_location] 高德 IP 定位 API 返回异常: {data}")
        # 兜底：返回一个随机的国内主要城市
        fallback = random.choice(["深圳", "北京", "上海", "杭州", "广州"])
        logger.info(f"[get_user_location] 兜底返回: {fallback}")
        return fallback

    city = data.get("city", "")
    province = data.get("province", "")

    if not city:
        # 如果 city 为空（如直辖市），尝试用 province
        city = province or "未知城市"

    logger.info(f"[get_user_location] IP定位结果: {province} {city}")
    return city


@tool(description="获取用户的ID，以纯字符串形式返回")
def get_user_id() -> str:
    return random.choice(user_ids)


@tool(description="获取当前月份，以纯字符串形式返回")
def get_current_month() -> str:
    return random.choice(month_arr)


def generate_external_data():
    """
    {
        "user_id": {
            "month" : {"特征": xxx, "效率": xxx, ...}
            "month" : {"特征": xxx, "效率": xxx, ...}
            "month" : {"特征": xxx, "效率": xxx, ...}
            ...
        },
        "user_id": {
            "month" : {"特征": xxx, "效率": xxx, ...}
            "month" : {"特征": xxx, "效率": xxx, ...}
            "month" : {"特征": xxx, "效率": xxx, ...}
            ...
        },
        "user_id": {
            "month" : {"特征": xxx, "效率": xxx, ...}
            "month" : {"特征": xxx, "效率": xxx, ...}
            "month" : {"特征": xxx, "效率": xxx, ...}
            ...
        },
        ...
    }
    :return:
    """
    if not external_data:
        external_data_path = get_abs_path(agent_conf["external_data_path"])

        if not os.path.exists(external_data_path):
            raise FileNotFoundError(f"外部数据文件{external_data_path}不存在")

        with open(external_data_path, "r", encoding="utf-8") as f:
            for line in f.readlines()[1:]:
                arr: list[str] = line.strip().split(",")

                user_id: str = arr[0].replace('"', "")
                feature: str = arr[1].replace('"', "")
                efficiency: str = arr[2].replace('"', "")
                consumables: str = arr[3].replace('"', "")
                comparison: str = arr[4].replace('"', "")
                time: str = arr[5].replace('"', "")

                if user_id not in external_data:
                    external_data[user_id] = {}

                external_data[user_id][time] = {
                    "特征": feature,
                    "效率": efficiency,
                    "耗材": consumables,
                    "对比": comparison,
                }


@tool(description="从外部系统中获取指定用户在指定月份的使用记录，以纯字符串形式返回， 如果未检索到返回空字符串")
def fetch_external_data(user_id: str, month: str) -> str:
    generate_external_data()

    try:
        return external_data[user_id][month]
    except KeyError:
        logger.warning(f"[fetch_external_data]未能检索到用户：{user_id}在{month}的使用记录数据")
        return ""


@tool(description="无入参，无返回值，调用后触发中间件自动为报告生成的场景动态注入上下文信息，为后续提示词切换提供上下文信息")
def fill_context_for_report():
    return "fill_context_for_report已调用"


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

        mgr = MemoryManager.get_instance()
        long_term = mgr.long_term
        if long_term is None:
            return "记忆系统未初始化"

        # 同步调用异步方法
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 在已有事件循环中，使用 run_coroutine_threadsafe 或创建新 loop
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run, long_term.retrieve_relevant(query)
                    )
                    memories = future.result(timeout=5)
            else:
                memories = loop.run_until_complete(long_term.retrieve_relevant(query))
        except RuntimeError:
            memories = asyncio.run(long_term.retrieve_relevant(query))

        if not memories:
            return "未找到相关历史记忆"

        return long_term.format_for_prompt(memories)
    except Exception as e:
        from utils.logger_handler import logger
        logger.warning(f"[memory_recall] 检索失败: {str(e)}")
        return f"记忆检索失败: {str(e)}"
