# Runtime Context 详解

> 本文配合 **智扫通** 项目（`agent/tools/middleware.py`）食用，讲清楚动态提示词切换的底层基石——Runtime Context。

---

## 一句话定义

**Runtime Context 是 LangGraph 在执行 Agent 时维护的一个字典对象，它在一次 `agent.stream()` 调用的全生命周期中持续存在，所有 Middleware 和工具都能读写它。**

---

## 在你的代码里，它是怎么流动的

### 第一步：Agent 启动时初始化 Context

```python
# agent/react_agent.py 第 27 行
for chunk in self.agent.stream(
    input_dict,
    stream_mode="values",
    context={"report": False}    # ← 这里初始化 Runtime Context
):
```

这个 `{"report": False}` 就是你放进 Runtime Context 的初始值。LangGraph 内部会把它包装成 `Runtime` 对象，跟随整个推理过程。

### 第二步：工具调用时，Middleware 修改 Context

```python
# agent/tools/middleware.py 第 26 行
@wrap_tool_call
def monitor_tool(request, handler):
    # ...
    if request.tool_call['name'] == "fill_context_for_report":
        request.runtime.context["report"] = True   # ← 在这里打标记
    # ...
```

`request.runtime.context` 就是这个 Runtime Context 字典。当 Agent 调用了 `fill_context_for_report` 工具，Middleware 把 `report` 从 `False` 改成 `True`。

### 第三步：下一轮模型调用前，Middleware 读取 Context

```python
# agent/tools/middleware.py 第 48 行
@dynamic_prompt
def report_prompt_switch(request):
    is_report = request.runtime.context.get("report", False)  # ← 读取标记
    if is_report:
        return load_report_prompts()    # 返回报告提示词
    return load_system_prompts()        # 返回默认提示词
```

同一个 `request.runtime.context`，这次读到的是已经被改成 `True` 的值，于是切换了提示词。

---

## 用图来理解

```
用户说"生成我的报告"
        │
        ▼
┌─────────────────────────────────────────────┐
│    agent.stream(context={"report": False})    │  ← 初始化
│                                               │
│  Runtime Context: {"report": False}           │
└─────────────────────────────────────────────┘
        │
        ▼  Agent 思考 → 决定调用 fill_context_for_report
        │
┌─────────────────────────────────────────────┐
│  monitor_tool 拦截到这次调用                   │
│  request.runtime.context["report"] = True     │  ← 修改
│                                               │
│  Runtime Context: {"report": True}            │
└─────────────────────────────────────────────┘
        │
        ▼  Agent 继续思考，准备调下一个工具
        │
┌─────────────────────────────────────────────┐
│  report_prompt_switch 在模型调用前检查          │
│  request.runtime.context.get("report") → True │  ← 读取
│  返回 report_prompt.txt 而不是 main_prompt.txt │
└─────────────────────────────────────────────┘
```

---

## Runtime Context 跟你熟悉的几个概念的区别

| 概念 | 作用域 | 谁可以写 | 类比 |
|---|---|---|---|
| **Runtime Context** | 一次 `stream()` 调用全程 | Middleware + 业务代码 | 一次请求的"全局变量" |
| **AgentState (messages)** | Agent 推理过程中 | Agent 框架本身 | 对话历史 + 推理状态 |
| **工具入参/出参** | 单次工具调用 | 工具函数 | 函数参数和返回值 |
| **st.session_state** | Streamlit 用户会话 | 任意代码 | 前端会话级缓存 |

关键区别：**AgentState 里的 messages 是 Agent 框架自己管的，你一般不直接改；Runtime Context 是你主动放进去、主动读写的自定义数据。**

---

## 为什么不用全局变量，而用 Runtime Context？

### 1. 线程安全

如果服务化了，多个用户同时请求，全局变量会串。Runtime Context 每个 `stream()` 调用独立，天然隔离。

```python
# 如果用全局变量（反面教材）
report_mode = False  # 用户A和用户B同时访问，直接串了

# 用 Runtime Context（正确做法）
context={"report": False}  # 每次 stream() 调用独立一份
```

### 2. 生命周期绑定

Context 随 `stream()` 调用结束而销毁，不会有"上次的标记残留到下次"的问题。全局变量你得手动清理。

```python
# 全局变量的问题
report_mode = True   # 上次请求改成了 True
# ... 请求结束，忘记重置
# 下一个请求进来，report_mode 还是 True → Bug!

# Runtime Context
# 每次 agent.stream(context={"report": False}) 都是全新的
# 请求结束自动销毁，不存在残留问题
```

### 3. 可追溯

Context 的读写都发生在 Middleware 的明确定义点上（`@wrap_tool_call`、`@dynamic_prompt`），读写在一条清晰的链路上，debug 时一目了然。

```
写入点：monitor_tool (@wrap_tool_call) → request.runtime.context["report"] = True
读取点：report_prompt_switch (@dynamic_prompt) → request.runtime.context.get("report")
```

---

## 延伸思考：Runtime Context 还能干什么？

你项目里只用了 `{"report": False/True}` 这一个标记，但这个机制可以扩展很多场景：

| 场景 | Context 标记 | 效果 |
|---|---|---|
| 多语言切换 | `{"lang": "en"}` | 检测到英文用户，切换英文提示词 |
| 权限控制 | `{"vip": True}` | VIP 用户解锁更多工具 |
| 多轮对话意图追踪 | `{"intent": "purchase"}` | 从咨询模式切换到推荐模式 |
| 调试模式 | `{"debug": True}` | 输出中间推理步骤给前端 |
| 用户情绪检测 | `{"angry": True}` | 切换为安抚话术的提示词 |

核心模式不变：**工具调用 → Middleware 写标记 → 下轮模型前 Middleware 读标记 → 切换行为。**
