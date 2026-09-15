"""性能赛道（LongMemEval-M）系统适配接口。

与 LoCoMo 的 SystemAdapter 不同：性能赛道做 **search 层直注**——adapter 不走各系统
的 LLM 抽取摄入，而是把预计算好的 chunks/embeddings/规范共现图直接载入后端，
再通过各系统的检索路径跑 4 类 query。这样把"存储引擎检索"隔离为唯一变量
（benchmark-plan.md §3.4）。

四类 query（与 build_dataset1.py derive 产物一一对应）：
  vector_topk     语义 top-k，GT=暴力余弦 top-10
  fts_keyword     AND 语义关键词，GT=全含词集合
  hybrid          向量粗筛→BM25 重排，GT=RRF(暴力向量序,暴力 BM25 序) top-10
  graph_multihop  规范共现图 2 跳邻域，GT=精确 BFS 邻域
"""
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

VECTOR_TOPK = "vector_topk"
FTS_KEYWORD = "fts_keyword"
HYBRID = "hybrid"
GRAPH_MULTIHOP = "graph_multihop"
ALL_CLASSES = (VECTOR_TOPK, FTS_KEYWORD, HYBRID, GRAPH_MULTIHOP)


@dataclass
class PerfCorpus:
    """直注进后端的预计算语料。全系统同一份，保证唯一变量是存储引擎。

    session_ids/texts/embeddings 三者按行对齐（第 i 行 = 第 i 个 session）。
    graph_edges 的节点编号 = session 行号。
    """
    session_ids: list          # 长度 n
    texts: list                # 长度 n
    embeddings: np.ndarray     # (n, dim) float32，已 L2 归一
    graph_edges: np.ndarray    # (E, 2) int32 无向边表
    #: mem0 fts doc 侧口径：core add() 对所有 backend 都存 lemmatized text（main.py:1031），
    #: 故 fts 的 doc 本应 lemmatized。预计算缓存（precompute_lemmatized.py）按行序对齐
    #: session_ids；None=未预计算，由 adapter 现场 lemmatize 兜底。三臂统一从此读，既还原
    #: mem0 真实行为、又把这份 core 固定成本移出 benchmark 的 load 计时（与 embeddings 同理）。
    text_lemmatized: list | None = None   # 长度 n 或 None


class PerfAdapter(ABC):
    """一个 (系统, 后端) 配置 = 一个 PerfAdapter 实例。

    生命周期：setup() -> load(corpus) -> query_*() × N -> teardown()。
    """

    #: 唯一标识，如 "mem0-neug" / "graphiti-neo4j" / "bruteforce-numpy"
    name: str = "base"
    #: 该配置支持的查询类。不支持的类由 runner 标 N/A（能力完整性矩阵，
    #: 与 benchmark-plan §3.2 N/A 规则互证）。默认全支持。
    supported_classes: frozenset = frozenset(ALL_CLASSES)

    @abstractmethod
    def setup(self, work_dir: str):
        """初始化后端（建库/起容器/加载扩展）。work_dir 每配置独立。"""

    @abstractmethod
    def load(self, corpus: PerfCorpus):
        """把预计算语料直注进后端：向量 + 文本（FTS）+ 规范共现图。

        不走 LLM 抽取。embeddings/graph_edges 原样载入，保证各后端拿到同一份数据。
        """

    # ---- 四类查询。均返回按相关度排序的 session_id 列表。----
    # 不支持的类保持抛 NotImplementedError，runner 依据 supported_classes 跳过。
    def query_vector(self, qvec: np.ndarray, top_k: int) -> list:
        raise NotImplementedError

    def query_fts(self, keywords: list, top_k: int) -> list:
        raise NotImplementedError

    def query_hybrid(self, qvec: np.ndarray, keywords: list, top_k: int) -> list:
        raise NotImplementedError

    def query_graph(self, seed_session_id: str, max_nodes: int) -> list:
        raise NotImplementedError

    def teardown(self):
        """关连接/清容器。必须正常 close（NeuG checkpoint 纪律）。"""


def timed(fn, *args, **kwargs):
    """统一计时包装（harness 侧 perf_counter，adapter 不自报延时）。

    返回 (result, latency_ms)。
    """
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, (time.perf_counter() - t0) * 1000.0
