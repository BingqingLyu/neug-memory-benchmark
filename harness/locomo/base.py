"""LoCoMo 统一 harness 的系统适配接口。

每个对比系统（Graphiti/Mem0/Cognee/OpenViking/graphify 各后端配置）实现一个
SystemAdapter。harness 只依赖此接口，保证所有系统走同一协议（benchmark-plan.md 3.1）。
"""
import inspect
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class SearchHit:
    """一次检索返回的单条结果。payload 为送入答题模型的上下文文本。"""
    id: str
    score: float
    payload: str


@dataclass
class SearchTrace:
    """检索侧遥测：延时与上下文量，交叉到性能维度（3.1 报告维度 7）。"""
    latency_ms: float = 0.0
    context_chars: int = 0
    extra: dict = field(default_factory=dict)


class SystemAdapter(ABC):
    """一个 (系统, 后端) 配置 = 一个 adapter 实例。

    生命周期：setup() -> ingest_all(sessions) -> search(question) × N -> teardown()。
    ingestion 的 LLM 抽取缓存纪律（3.1 协议 6）：同一系统换后端时抽取结果必须相同，
    adapter 应把抽取产物落在 cache_dir 下按 (system, sample_id) 键控。
    """

    #: 唯一标识，如 "graphiti-neo4j" / "mem0-neug" / "openviking"
    name: str = "base"
    #: 子场景：A=agent 记忆管理，B=内容索引（graphify 跨类参加 A 任务）
    sub_scenario: str = "A"

    @abstractmethod
    def setup(self, work_dir: str, cache_dir: str):
        """初始化后端（建库/起容器/加载扩展）。work_dir 每配置独立。"""

    @abstractmethod
    def ingest_session(self, sample_id: str, session_idx: int, date_time: str, text: str):
        """按 session 增量摄入（LoCoMo 协议：272 session 逐条进）。"""

    @abstractmethod
    def search(self, question: str, top_k: int = 20, sample_id: str | None = None
               ) -> tuple[list[SearchHit], SearchTrace]:
        """检索与问题相关的记忆上下文。延时在 harness 侧统一计时（见 run_eval）。

        sample_id：题目归属的样本（约束 1 会话隔离）。支持隔离的 adapter 应优先用它，
        而不是摄入侧的"最后一次摄入样本"——多样本全量跑时后者只会命中最后一个样本。
        旧 adapter 未实现时保持不接收（timed_search 会探测签名）。
        """

    def teardown(self):
        """关连接/清容器。必须保证正常 close（NeuG ghost PK 纪律）。"""

    def reuse_ingested(self, sessions):
        """work_dir 已含摄入数据时（run>0 复用 run0 的库）恢复摄入期内存状态。

        查询阶段只读且摄入确定性（抽取经缓存），多 run 复用同一库是安全的。
        无状态 adapter 保持默认空实现。
        """

    # ---- 便捷封装 ----
    def ingest_all(self, sessions):
        for s in sessions:
            self.ingest_session(s["sample_id"], s["session_idx"], s["date_time"], s["text"])


def timed_search(adapter: SystemAdapter, question: str, top_k: int = 20,
                 sample_id: str | None = None):
    """统一计时包装：p50/p95 由 harness 侧采集，避免各 adapter 自报口径。"""
    t0 = time.perf_counter()
    # 未声明 sample_id 的旧 adapter 保持原调用方式，避免 TypeError
    params = inspect.signature(adapter.search).parameters
    if sample_id is not None and "sample_id" in params:
        hits, trace = adapter.search(question, top_k=top_k, sample_id=sample_id)
    else:
        hits, trace = adapter.search(question, top_k=top_k)
    trace.latency_ms = (time.perf_counter() - t0) * 1000.0
    trace.context_chars = sum(len(h.payload) for h in hits)
    return hits, trace
