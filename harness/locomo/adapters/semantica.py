"""semantica LoCoMo adapter：semantica-neug / semantica-native 双臂。

HANDOFF-semantica-benchmark.md（§2 质量赛道）：
- 接原生抽取管线：GraphBuilder.build（spacy NER + pattern 三元组，零 LLM），
  抽取产物按 (system, sample_id) 键控落 cache_dir，两配置共享同一份产物。
- semantica-neug：NeuG 单库承载三路检索——HNSW 向量 + bm25 全文 + 图遍历。
  向量与 bm25 经原生 VectorStore(backend=neug) 门面由后端 §1.4 内聚融合，图路经
  GraphStore(backend=neug).get_neighbors；两门面经 §1.2 注册表共享单 engine。跨池
  （向量+bm25 融合池 / 图池）合并仍用原生 ContextRetriever._rank_and_merge。
- semantica-native：内存 ContextGraph + 原生 FAISS vector store，检索是
  向量+图两路；不接全文路（原生栈无全文，见 HANDOFF §1 源码事实），也不写
  客户端关键词兜底（决策 B3）。
- 两臂都裸调 ContextRetriever._rank_and_merge 当纯融合函数、不走 retrieve()：
  retrieve() 尾部的语义重排以 self.vector_store 为开关，会对每个结果再 embed 一次
  content[:500]（缓存键是截断串、必 miss），实测每题多次真实 API 调用计入检索延时；
  两臂必须同构（约束 5「后端切换唯一变量」），故都不接 vector_store。neug 臂的
  向量+bm25 融合在后端 §1.4 完成，不经这条尾部重排。
- native 臂的向量分不能用 FAISS 返回的 "score" 字段：它对 inner_product 是
  余弦的严格递减函数，直接喂进融合器会把排序反转（见 _faiss_ip_score）。
- 图路必须全路常量分，不要按种子排名衰减：get_neighbors 不返回相关度，
  编造的序会被 min-max 拉到满量程而劫持首位（见 GRAPH_SEED_SCORE）。
- native 臂的边必须双向各写一次：ContextGraph 的邻接只按 source 存、
  get_neighbors 只走出边，单向写入等于只看得到出边，而 neug 臂的
  GraphStore.get_neighbors 默认 direction="both"（实测差异见 _flush）。
- neug 臂 reuse 冻结库前要过 _assert_vector_leg（走原生 vs.count()）：跨仓
  schema 迁移会让向量路静默返回空、三路塌缩成纯 bm25，而 run 看上去是成功的
  （见该方法 docstring）。
- 唯一变量是后端：抽取产物、embedding（text-embedding-v3 内容哈希缓存）、
  hybrid_alpha、top_k、切块策略、图路打分与遍历方向、融合器、_to_hits
  两配置完全一致。

实测结果、归因限制与三次静默失败事故（P0-4 / C2 / P0-5）记在
`results/locomo/semantica-neug/NOTES.md`（两臂同一份）。
离线回归：`probes/probe_semantica_arms.py`（54 项断言，无网络无 LLM）。

契约：harness/locomo/ADAPTER-CONTRACT.md
- 会话隔离：每 sample_id 一个独立库（neug 文件库 / native 内存图）（约束 1）
- search 只返回检索上下文，答题由 harness 统一做（约束 2）
- teardown 正常 close（约束 6）。引擎多实例坑：同进程关闭一个实例会破坏
  其他实例的向量查询，故多样本库全程保活、仅在 teardown 统一关
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from openai import OpenAI

from ..base import SearchHit, SearchTrace, SystemAdapter
from ..llm import DEFAULT_BASE_URL

# semantica（含 graph_store/neug provider）经 benchmark venv 的 editable 安装提供，
# 与 graphiti/mem0/cognee 三系统同惯例（裸导入，无 sys.path 注入）。

EMBED_MODEL = os.environ.get("BENCH_EMBED_MODEL", "text-embedding-v3")
EMBED_DIM = int(os.environ.get("BENCH_EMBED_DIM", "1024"))
CHUNK_CHARS = 1000   # 与 naive-rag 同口径：按 turn 边界切，不拆发言
EMBED_BATCH = 10     # DashScope 兼容端点单次批量上限（保守值）
HYBRID_ALPHA = 0.5   # 0=vector only, 1=graph only（Semantica 原生融合权重）
GRAPH_SEEDS = 3      # 图扩展的种子数（向量 top 命中）
# get_neighbors 不返回相关度，所以这一路**真实方差为 0**，只能给常量分。
# 常量分在 _rank_and_merge 里因 max==min 跳过 min-max 归一化，全路保持
# GRAPH_SEED_SCORE×hybrid_alpha + context_boost = 0.50×0.5+0.01 = 0.26，
# 落在向量路（首位恒为 1.0×(1-alpha) = 0.50）的中段。
#
# **不要按种子排名衰减给分**。曾试过 0.50/0.45/0.40 想给路内一点区分度，但
# min-max 只保序不保值：编造的三个分会被拉到满量程 {1.0, 0.5, 0.0}，
# ×alpha+boost 后图路首位变成 0.51（被多个种子共同命中的邻居再 ×1.2 =
# 0.61），**压过向量路首位 0.50**。等于宣称「头号图命中比头号向量命中更相
# 关」，而那个序完全是编的。真实 LoCoMo 实测代价（conv-30，40 题，
# probes/probe_graph_leg_score.py）：
#   衰减 0.05：90% 的问题 top-1 是实体节点（content 就是个名字，如
#              "Caroline"）、实体节点占 top-20 的 23.1%、ctx 均值 -19.7%
#   常量分  ：top-1 是实体节点 5.0%、占 7.4%、ctx 均值 15217.7（与 9/2
#              那次有效三路跑批的 15129.2 只差 0.6%）
# 全量 1540 题（同一份重摄入库，只差打分）：衰减分 accuracy 0.4481 vs
# 常量分 0.5961（-14.8pp），native 臂同步从 0.5136 掉到 0.4727。
# 该不变量由 probes/probe_semantica_arms.py::check_graph_leg_constant 用真实
# _rank_and_merge 钉住（常量分向量路首位胜、衰减分图路首位胜）。
#
# 常量分不等于整路全平：_rank_and_merge 自带的多源 boost（同一 node_id 被
# 多个种子命中时 max(...)*1.2）会把 0.26 抬到 0.31，即「被多个头部向量种子
# 共同指向的邻居」排在只被一个指向的前面。这是融合器自己的真实信号，比编造
# 的排名衰减可靠。
GRAPH_SEED_SCORE = 0.50


def _chunk_text(text: str) -> list[str]:
    """按行切块（与 naive-rag 同函数语义：目标长度、不拆发言）。

    `(text or "")` 与 naive_rag.py 的切块逐字相同，是为了与基线 adapter 同
    口径（契约要求切块策略一致），不是多余的容错：实测 272 个 session 无一
    为空、最短 1161 字符。另注意本函数对任何输入都至少产出一块
    （`"".split("\n") == [""]`，尾部 `if buf:` 恒真）。
    """
    lines = (text or "").split("\n")
    chunks, buf, size = [], [], 0
    for ln in lines:
        if size + len(ln) > CHUNK_CHARS and buf:
            chunks.append("\n".join(buf))
            buf, size = [], 0
        buf.append(ln)
        size += len(ln) + 1
    if buf:
        chunks.append("\n".join(buf))
    return chunks


def _norm_vec(v: list[float]) -> list[float]:
    """L2 归一化。

    native 臂的 FAISS flat 索引用 inner_product，只有在归一向量上才等价余弦；
    neug 臂的 HNSW 建索引时已 cosine_normalize，这里对它是幂等的。
    """
    arr = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(arr))
    return (arr / n).tolist() if n > 0 else arr.tolist()


def _faiss_ip_score(hit: dict) -> float:
    """把 FAISS inner_product 命中换算成与 neug 臂同标度的余弦分。

    FAISSSearch.search_similar 的 "score" 是 1/(1+max(0,dist))，而
    inner_product 下 FAISS 返回的 dist 本身就是余弦 —— 该变换是余弦的严格
    递减函数（实测 cos=1.0 -> 0.500、cos=0.0 -> 1.000），喂进 _rank_and_merge
    的 min-max 归一化后排序被整体反转。"distance" 字段才是余弦。
    neug 臂 NeugVectorStore.search 给的是 1 - cos_distance/2 = (1+cos)/2，
    这里取同一公式，两臂向量路分值可直接互比。

    只适用于 FAISS 命中：两臂的 hit dict 都有 "distance" 键但语义相反
    （neug = 余弦距离，0 为相同；FAISS inner_product = 余弦相似度，1 为
    相同），把本函数用在 neug 命中上会得到反向分。neug 的 "score" 已经
    是正确的，直接用。
    """
    cos = float(hit["distance"])
    return max(0.0, min(1.0, (1.0 + cos) / 2.0))


def _graph_leg(neighbors_of, vec_hits) -> list:
    """一跳图扩展路，两臂共用：全路常量分（见 GRAPH_SEED_SCORE 注释）。

    ``neighbors_of(node_id)`` 由各臂提供（GraphStore.get_neighbors /
    ContextGraph.get_neighbors），产出 ``(id, content)``。同一邻居被多个种子
    命中时保留多条，交给 _rank_and_merge 按 node_id 去重并做多源 boost
    （max(...)*1.2）——路内的区分度由那个 boost 提供，不在这里编。

    **每个种子的邻居必须排序后再产出**。全路同分意味着 _rank_and_merge 里
    大量命中是精确同分，而它尾部的 ``sorted(key=score, reverse=True)`` 是稳定
    排序 —— 同分之间谁进 top_k 完全由进入融合器时的**列表位置**决定，也就是
    由后端的邻居遍历顺序决定（neug 是 varlen 遍历序，native 是
    ``ContextGraph._adjacency`` 的插入序）。实测（conv-30，81 题，两臂逐题
    对比）：

      - 两臂的图路输入是同一个多重集：243 次种子扩展，邻居 id 的多重集与
        content **全部相同**，只有顺序不同（顺序也相同的仅 36/243）。
      - 向量路同样逐位相同（id 集合、顺序、分值差 <= 1.19e-07）。
      - 但融合后的 top-20：native 有 **76.4%** 的槽位处于精确同分并列，
        实体节点占 23.5%；把 neug 的 bm25 路摘掉后是 32.8%。**输入相同、
        输出只有 17/81 题的 id 集合相同** —— 差异全部来自 tie-break。

    排序把这个隐藏变量钉死：两臂从此对同一种子产出同一顺序，tie-break 不再
    依赖后端。按 node_id 排是中性的选择（不引入任何相关性假设，图路本来也
    没有相关性可给）；副作用是 chunk id（``conv-30:1:1``）字典序先于实体 id
    （``ent:conv-30:Jon``），实测让无 bm25 那一路的实体占比从 32.8% 降到
    20.1%、ctx 均值从 10745 升到 12786 —— 这是把 "yesterday" / "Lemme" 这类
    spacy NER 噪声实体换回真实 chunk，不是调参调出来的收益。
    """
    from semantica.context.context_retriever import RetrievedContext

    leg = []
    for h in vec_hits[:GRAPH_SEEDS]:
        for nb_id, content in sorted(neighbors_of(h["id"])):
            leg.append(RetrievedContext(
                content=content, score=GRAPH_SEED_SCORE,
                source=f"graph:{nb_id}", metadata={"node_id": nb_id},
                related_entities=[{"id": h["id"], "type": "seed"}]))
    return leg


def _to_hits(merged, top_k: int) -> list:
    """融合结果 -> SearchHit：滤空 payload、跨路去重、再截断，一次遍历。

    三项顺序不能反：先截断会在前 top_k 含空 content 或重复节点时少给命中。

    跨路去重是**结构上**必要的：_rank_and_merge 只在路内去重（vector/memory
    路按 content[:100]、graph 路按 node_id），合并时的过滤只剔除 graph 源那
    一份，不剔除 vector 源那一份 —— 同一节点从两路各进一次就会占掉两个
    top_k 名额。merged 已按分降序，首次出现即最高分。

    但**实测收益取决于语料规模，不要当成性能优化读**：
    - 3-session 合成语料（probes/probe_semantica_arms.py）：neug 臂 top_k=8
      里重复 3 个 —— 语料太小时图路邻居与向量命中高度重叠。
    - 真实 LoCoMo（conv-30，top_k=20，健康三路，25 题）：
      `cross_leg_dups_removed = 0`。图路平均 54.1 条命中，但它们在
      min-max 归一 + hybrid_alpha 加权后排名靠后，挤不进前 20，所以与
      vector 路（含折进去的 bm25）不相遇。
    即：在真实负载上这是正确性保险而非可测量的预算节省。保留是因为成本
    就是一次 set 查找，且小语料/高重叠数据上确实会触发。
    这不改变融合算法本身，只保证 adapter 输出是一份不重复的排序命中列表；
    两臂同一函数，约束 5 对称。
    """
    seen = set()
    hits = []
    for r in merged:
        if not r.content:
            continue
        nid = str(r.metadata["node_id"])
        if nid in seen:
            continue
        seen.add(nid)
        hits.append(SearchHit(id=nid, score=float(r.score), payload=r.content))
        if len(hits) == top_k:
            break
    return hits


class _SemanticaBase(SystemAdapter):
    """两臂共享：嵌入缓存、抽取产物、统一的摄入行构造。"""
    sub_scenario = "A"

    def __init__(self):
        self._work_dir: Path | None = None
        self._cache_dir: Path | None = None
        self._client = None
        self._embed_cache_path: Path | None = None
        self._embed_cache: dict[str, list[float]] = {}
        self._artifacts: dict[str, dict] = {}
        self._ingested: set[tuple[str, int]] = set()

    # ---- 生命周期 ----
    def setup(self, work_dir: str, cache_dir: str):
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("DASHSCOPE_API_KEY not set")
        self._client = OpenAI(
            base_url=os.environ.get("DASHSCOPE_BASE_URL", DEFAULT_BASE_URL),
            api_key=api_key, timeout=120,
        )
        self._work_dir = Path(work_dir)
        self._work_dir.mkdir(parents=True, exist_ok=True)
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._embed_cache_path = self._cache_dir / f"semantica_embed_{EMBED_MODEL}.jsonl"
        self._load_embed_cache()

    # ---- embedding（内容哈希磁盘缓存，协议 3/6，两臂共享）----
    @staticmethod
    def _content_key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]

    def _load_embed_cache(self):
        if not self._embed_cache_path.exists():
            return
        with self._embed_cache_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                self._embed_cache[rec["k"]] = rec["v"]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        keys = [self._content_key(t) for t in texts]
        out = [self._embed_cache.get(k) for k in keys]
        missing = [i for i, v in enumerate(out) if v is None]
        if not missing:
            return out
        pending = []
        for s in range(0, len(missing), EMBED_BATCH):
            batch_idx = missing[s:s + EMBED_BATCH]
            resp = self._client.embeddings.create(
                model=EMBED_MODEL, input=[texts[i][:8000] for i in batch_idx])
            for j, d in enumerate(resp.data):
                i = batch_idx[j]
                vec = [float(x) for x in d.embedding]
                out[i] = vec
                self._embed_cache[keys[i]] = vec
                pending.append(json.dumps({"k": keys[i], "v": vec}))
        # 整批一次追加：原先在向量循环里逐条 open("a")，每条一次句柄开关。
        # 缓存是可重建的幂等产物，进程中断只丢掉当前批、下次重新请求即可。
        with self._embed_cache_path.open("a") as f:
            f.write("\n".join(pending) + "\n")
        return out

    # ---- 抽取产物（按 (system, sample_id) 键控，两臂共享）----
    def _artifact_path(self, sample_id: str) -> Path:
        return self._cache_dir / f"semantica_extract_{sample_id}.json"

    def _get_extraction(self, sample_id: str, session_idx: int,
                        date_time: str, text: str) -> dict:
        """返回该 session 的抽取产物；缺失则跑原生管线并落盘共享。

        artifact 按 sample_id 常驻内存、只在新 session 抽取后写盘：原先每个
        session 都重读并重写整份 artifact，IO 随 session 数平方增长。
        """
        artifact = self._artifacts.get(sample_id)
        if artifact is None:
            path = self._artifact_path(sample_id)
            artifact = {"sample_id": sample_id, "method": "spacy+pattern",
                        "sessions": {}}
            if path.exists():
                artifact = json.loads(path.read_text())
            self._artifacts[sample_id] = artifact
        key = str(session_idx)
        if key in artifact["sessions"]:
            return artifact["sessions"][key]

        # 原生抽取：spacy NER + pattern 三元组（GraphBuilder 默认，零 LLM）
        from semantica.kg.graph_builder import GraphBuilder
        graph = GraphBuilder(resolve_conflicts=False).build(
            text, ner_method="spacy", extract_relations=False,
            extract_triplets=True)
        sess = {
            "date_time": date_time or "",
            "entities": [
                {"id": e.get("id"), "name": e.get("name"),
                 "type": e.get("type", "UNKNOWN")}
                for e in graph["entities"] if e.get("id")
            ],
            "relationships": [
                {"source": r.get("source"), "target": r.get("target"),
                 "type": r.get("type", "RELATED_TO")}
                for r in graph["relationships"]
                if r.get("source") and r.get("target")
            ],
        }
        artifact["sessions"][key] = sess
        self._write_artifact(sample_id, artifact)
        return sess

    def _write_artifact(self, sample_id: str, artifact: dict):
        path = self._artifact_path(sample_id)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(artifact, ensure_ascii=False, indent=1))
        os.replace(tmp, path)

    # ---- 摄入：两臂统一的行构造（切块 + 抽取产物 -> 节点/边）----
    def _build_rows(self, sample_id: str, session_idx: int,
                    date_time: str, text: str, sess_art: dict,
                    prev_head: str | None):
        chunks = _chunk_text(text)
        node_rows, edge_rows = [], []
        for ci, chunk in enumerate(chunks):
            node_rows.append({
                "id": f"{sample_id}:{session_idx}:{ci}", "etype": "chunk",
                "content": chunk,
                "extra": json.dumps({"date_time": date_time or "",
                                     "session_idx": session_idx},
                                    ensure_ascii=False),
            })
        edge_rows.extend(
            {"from_id": f"{sample_id}:{session_idx}:{ci}",
             "to_id": f"{sample_id}:{session_idx}:{ci + 1}",
             "edge_type": "NEXT", "weight": 1.0}
            for ci in range(len(chunks) - 1))
        if prev_head and chunks:
            edge_rows.append({"from_id": prev_head,
                              "to_id": f"{sample_id}:{session_idx}:0",
                              "edge_type": "FOLLOWS", "weight": 1.0})

        # 抽取实体节点（跨 session 同名实体同 id 自然合并）
        texts_lower = [c.lower() for c in chunks]
        for ent in sess_art.get("entities", []):
            name = str(ent["name"])
            node_rows.append({
                "id": f"ent:{sample_id}:{name}", "etype": str(ent["type"]),
                "content": name,
                "extra": json.dumps({"session_idx": session_idx},
                                    ensure_ascii=False),
            })
            # 实体 -> 提及它的块（确定性字符串包含链接）
            nl = name.lower()
            for ci, tl in enumerate(texts_lower):
                if nl in tl:
                    edge_rows.append({
                        "from_id": f"ent:{sample_id}:{name}",
                        "to_id": f"{sample_id}:{session_idx}:{ci}",
                        "edge_type": "MENTIONED", "weight": 1.0})
        # 三元组边（实体-实体）
        for rel in sess_art.get("relationships", []):
            edge_rows.append({
                "from_id": f"ent:{sample_id}:{rel['source']}",
                "to_id": f"ent:{sample_id}:{rel['target']}",
                "edge_type": str(rel["type"]), "weight": 1.0})
        return node_rows, edge_rows, chunks

    def ingest_session(self, sample_id: str, session_idx: int,
                       date_time: str, text: str):
        if (sample_id, session_idx) in self._ingested:
            return
        sess_art = self._get_extraction(sample_id, session_idx, date_time, text)
        prev_head = self._prev_head_for(sample_id)
        node_rows, edge_rows, chunks = self._build_rows(
            sample_id, session_idx, date_time, text, sess_art, prev_head)
        # 不检查 node_rows 是否为空：_chunk_text 对任何输入都至少产出一块，
        # _build_rows 又为每块无条件建行，所以这里恒非空。之前那句
        # `if not node_rows: return` 是不可达的静默兜底：真出现空行只可能是
        # 切块逻辑被改坏，那时应该炸，而不是悄悄少摄入一个 session、run 照样
        # "成功"（与 P0-4 / P0-5 同一类失败模式）。
        # 节点向量：块与实体同一口径（内容哈希缓存，两臂共享零新增调用）
        vecs = self._embed([r["content"] for r in node_rows])
        for row, vec in zip(node_rows, vecs):
            row["vec"] = _norm_vec(vec)
        self._flush(sample_id, node_rows, edge_rows)
        if chunks:
            self._set_prev_head(sample_id, f"{sample_id}:{session_idx}:0")
        self._ingested.add((sample_id, session_idx))

    def _resolve_store(self, sample_id: str | None, stores: dict) -> dict:
        """解析本 sample 的存储句柄，解析不到就抛。

        返回空命中会让 run 判为成功、accuracy 静默归零，在 summary.json 里与
        "检索确实没召回"无法区分（同仓 mem0-neug 就是这么跑出 0.0 的）。
        """
        if sample_id is None:
            if len(stores) != 1:
                raise RuntimeError(
                    "sample_id required when multiple samples ingested "
                    f"(have {len(stores)})")
            sample_id = next(iter(stores))
        if sample_id not in stores:
            raise RuntimeError(
                f"no store for sample_id={sample_id!r}; "
                f"ingested={sorted(stores)}")
        return stores[sample_id]

    # ---- 子类钩子 ----
    def _prev_head_for(self, sample_id: str) -> str | None:
        raise NotImplementedError

    def _set_prev_head(self, sample_id: str, head: str):
        raise NotImplementedError

    def _flush(self, sample_id: str, node_rows: list[dict], edge_rows: list[dict]):
        raise NotImplementedError

    def reuse_ingested(self, sessions):
        raise NotImplementedError

    def teardown(self):
        pass


class SemanticaNeuGAdapter(_SemanticaBase):
    """with NeuG：一个嵌入式库承载向量 + 全文 + 图三路检索。"""
    name = "semantica-neug"

    def __init__(self):
        super().__init__()
        # sample_id -> {"gs": GraphStore(neug), "vs": VectorStore(neug)}；两门面经
        # §1.2 注册表共享单 engine，向量+bm25 由后端 §1.4 融合。
        self._stores: dict[str, dict] = {}
        self._prev_head: dict[str, str] = {}
        self._retriever = None

    def setup(self, work_dir: str, cache_dir: str):
        super().setup(work_dir, cache_dir)
        from semantica.context.context_retriever import ContextRetriever
        self._retriever = ContextRetriever(hybrid_alpha=HYBRID_ALPHA)

    def teardown(self):
        # 引擎多实例坑：仅在此统一关闭，关闭后不再有任何查询。gs 与 vs 共享单
        # engine（§1.2 注册表 refcount==2），两个都 close 才把 refcount 归 0、
        # 真正释放引擎一次。
        for st in self._stores.values():
            st["vs"].close()
            st["gs"].close()
        self._stores.clear()

    def reuse_ingested(self, sessions):
        for s in sessions:
            sid = s["sample_id"]
            if sid in self._stores:
                continue
            self._assert_vector_leg(sid, self._ensure_store(sid))

    def _assert_vector_leg(self, sample_id: str, st: dict):
        """复用冻结库前校验向量路活着，否则整个 run 会静默退化成纯 bm25。

        run_eval 的复用闸门只比对 session 数（`.ingest_complete` 里就一个
        `{"sessions": N}`），不带存储 schema 指纹。而后端迁移过 DDL：新增
        `has_vec BOOL DEFAULT false`，`NeugVectorStore.search` 用它做 HNSW
        预过滤。旧库在 connect 时被 `ALTER TABLE ADD IF NOT EXISTS` 补上了列，
        但存量行全是默认 false，于是 `WHERE n.has_vec` 匹配 0 行、`search`
        返回空列表——不抛异常，与「确实没召回」无法区分；图路也因拿不到向量
        种子而一并死掉，三路塌缩成一路。

        2026-09-09 的 neug 臂就是这样跑出一份纯 bm25 结果（accuracy 0.5929 /
        p50 0.52ms），10 个库 `has_vec=true` 的行数全为 0，与旧基线的差值被
        误读成本次代码改动的收益。修法：删掉 `work_run0` 重新摄入（嵌入与
        抽取产物都在 `results/locomo/cache/semantica/` 下，0 次 API 调用）。
        与 `_resolve_store` 同理：宁可炸，不要静默出数。
        """
        # vs.count() = NeugVectorStore.count() = WHERE has_vec=true 的行数，正是
        # 向量路 HNSW 预过滤的谓词。走原生门面，不再伸手 _store_backend 发裸 Cypher。
        if st["vs"].count() == 0:
            raise RuntimeError(
                f"{self.name}: store for {sample_id!r} has no rows with an "
                "embedding (vs.count()==0), so the vector leg (and the graph "
                "leg seeded from it) would silently return nothing and the run "
                "would measure bm25 only. The frozen store predates the has_vec "
                f"schema migration - delete {self._work_dir} and re-ingest.")

    def _prev_head_for(self, sample_id):
        return self._prev_head.get(sample_id)

    def _set_prev_head(self, sample_id, head):
        self._prev_head[sample_id] = head

    def _ensure_store(self, sample_id: str) -> dict:
        st = self._stores.get(sample_id)
        if st is not None:
            return st
        from semantica.graph_store.graph_store import GraphStore
        from semantica.vector_store.vector_store import VectorStore

        db_path = str(self._work_dir / f"semantica_{sample_id}.db")
        gs = GraphStore(backend="neug", db_path=db_path, vector_dim=EMBED_DIM)
        gs.connect()
        # Option A：向量+bm25 走原生 VectorStore(backend=neug) 门面（§1.3 派发 →
        # §1.4 后端内聚融合），不再手工拼 NeugVectorStore/NeugFTS。§1.2 注册表让
        # gs 与 vs 共享同一个 engine（refcount==2），无需手工共享 store。
        vs = VectorStore(backend="neug",
                         config={"db_path": db_path, "dimension": EMBED_DIM})
        st = {"gs": gs, "vs": vs}
        self._stores[sample_id] = st
        return st

    def _flush(self, sample_id, node_rows, edge_rows):
        # 每 session 一次 COPY，故意不按 sample 合并——与 perf 臂的 E4（合并成
        # 单次 COPY，省 22.7%）相反，因为两边表规模差三个数量级：COPY 的固定
        # 开销 ∝ 表内已有行数（每次 seal 整表重写 checkpoint），perf 是单库
        # 51661 节点 / 2.8M 边，locomo 是每 sample 一个库、106-269 节点 /
        # 268-1356 边（10 库合计 1996 节点）。这个量级下重写成本可忽略，而
        # 逐 session 落盘让摄入进度可中断续跑。摄入耗时也不是本赛道的对比
        # 指标（summary.json 只记 accuracy 与检索延时）。
        backend = self._ensure_store(sample_id)["gs"]._store_backend
        backend.copy_nodes(node_rows)
        if edge_rows:
            backend.copy_edges(edge_rows)

    # ---- 检索：向量+bm25（后端 §1.4 融合）+ 图扩展，跨池走原生 _rank_and_merge ----
    def search(self, question: str, top_k: int = 20, sample_id: str | None = None):
        st = self._resolve_store(sample_id, self._stores)

        from semantica.context.context_retriever import RetrievedContext

        # 归一化对 neug 是幂等的（cosine_normalize 默认 true，余弦距离本身
        # 尺度不变），做了才能让两臂交给后端的查询向量逐位相同。
        qvec = _norm_vec(self._embed([question])[0])
        results: list[RetrievedContext] = []

        # 路 1+2：HNSW 向量 + bm25 全文，经原生 VectorStore 门面由后端 §1.4 内聚
        # 融合——传 _neug_raw_query 触发 bm25 路，命中自带 metadata['retrieval_routes']
        # 溯源（['vector'] / ['bm25'] / 两者=共识）。qvec 由本臂预算好（与 native 臂
        # 同一 _embed，约束 5 对称），不在后端重复 embed。
        # content 一律直接下标取，不用 .get(..., "")：向量命中来自 _metadata_from_row、
        # bm25-only 命中来自 fts hit，后端给每个命中都写了 content，缺了就说明返回
        # 形状变了。此时空串会被 _to_hits 当空 payload 滤掉、一路静默消失、run 照样
        # "成功"——正是 P0-4 的失败模式。宁可炸。
        vec_hits = st["vs"].search_vectors(qvec, k=top_k, _neug_raw_query=question)
        for h in vec_hits:
            results.append(RetrievedContext(
                content=h["metadata"]["content"], score=h["score"],
                source=f"vector:{h['id']}",
                metadata={"node_id": h["id"],
                          "retrieval_routes": h["metadata"].get("retrieval_routes")},
            ))

        # 路 3：图扩展（与 native 臂共用 _graph_leg，全路常量分同参；种子取自融合命中）
        results.extend(_graph_leg(
            lambda nid: ((n["id"], n["properties"]["content"])
                         for n in st["gs"].get_neighbors(nid, depth=1)),
            vec_hits))

        # Semantica 原生融合：把「向量+bm25 融合池」与「图池」分池归一化 +
        # hybrid_alpha 加权 + 去重（bm25 已在后端折进向量池，这里不再单列一路）。
        # 融合改变了向量池的分值量程，图路占比随之变化——两池实际配比记在
        # results/locomo/semantica-neug/NOTES.md（p2-3 重跑后更新）。
        merged = self._retriever._rank_and_merge(results, question)
        hits = _to_hits(merged, top_k)
        trace = SearchTrace(extra={
            "vector_hits": len(vec_hits),
            "bm25_fused": sum(1 for h in vec_hits
                              if "bm25" in (h["metadata"].get("retrieval_routes") or [])),
            "graph_hits": sum(1 for r in results if r.source.startswith("graph:")),
            "merged": len(merged),
        })
        return hits, trace


class SemanticaNativeAdapter(_SemanticaBase):
    """without NeuG 对照：内存 ContextGraph + 原生 FAISS，向量+图两路。

    原生栈无全文路，不接 fts（也不写客户端关键词兜底，决策 B3）。检索与
    neug 臂同构：自拼两路命中后裸调 _rank_and_merge，不走 retrieve()。
    """
    name = "semantica-native"

    def __init__(self):
        super().__init__()
        # sample_id -> {"cg": ContextGraph, "vs": VectorStore, "rt": ContextRetriever}
        self._stores: dict[str, dict] = {}
        self._prev_head: dict[str, str] = {}

    def reuse_ingested(self, sessions):
        # 内存栈不跨进程持久：复用 run 时基于共享缓存（抽取产物 + 嵌入）
        # 零 API 调用重建，语义等价于读回已摄入的库。
        for s in sessions:
            if (s["sample_id"], s["session_idx"]) not in self._ingested:
                self.ingest_session(s["sample_id"], s["session_idx"],
                                    s.get("date_time", ""), s["text"])

    def _prev_head_for(self, sample_id):
        return self._prev_head.get(sample_id)

    def _set_prev_head(self, sample_id, head):
        self._prev_head[sample_id] = head

    def _ensure_store(self, sample_id: str) -> dict:
        st = self._stores.get(sample_id)
        if st is not None:
            return st
        from semantica.context.context_graph import ContextGraph
        from semantica.context.context_retriever import ContextRetriever
        from semantica.vector_store.vector_store import VectorStore

        cg = ContextGraph()
        vs = VectorStore(backend="faiss", config={"dimension": EMBED_DIM})
        # 故意不传 knowledge_graph/vector_store：_rank_and_merge 尾部的语义
        # 重排以 self.vector_store 为开关，一旦非空就对每个结果再 embed 一次
        # content[:500]。截断键与摄入时的完整 chunk 不同（实测 chunk 671-673
        # 字符），哈希缓存必 miss —— 实测每题 22 次 embed（neug 臂 1 次），
        # 其中 7 次是真实 API 调用且全部计入检索延时。融合器只当纯函数用。
        rt = ContextRetriever(hybrid_alpha=HYBRID_ALPHA)
        st = {"cg": cg, "vs": vs, "rt": rt}
        self._stores[sample_id] = st
        return st

    def _flush(self, sample_id, node_rows, edge_rows):
        st = self._ensure_store(sample_id)
        cg, vs = st["cg"], st["vs"]
        cg.add_nodes([
            {"id": r["id"], "type": r["etype"], "content": r["content"]}
            for r in node_rows
        ])
        if edge_rows:
            # 每条边双向各写一次。ContextGraph._add_internal_edge 只做
            # `_adjacency[source_id].append`，get_neighbors 又只遍历
            # `_adjacency[current_id]`（变量名就叫 outgoing_edges），所以单向
            # 写入 == 图路只看得到出边；而 neug 臂的 GraphStore.get_neighbors
            # 默认 direction="both"，是无向遍历。两边必须同构（约束 5）。
            #
            # 这个差异不是小数点后的事：_build_rows 的 MENTIONED 是
            # entity->chunk，单向写入下 chunk 种子**根本走不到**提到它的实体。
            # 实测 conv-30 五个问题、15 个向量种子里 10 个是 chunk 节点：
            #   neug（无向）    299 个邻居，67 个实体，chunk 种子触达实体 32 次
            #   native 单向     244 个邻居，19 个实体，chunk 种子触达实体  0 次
            #   native 双向     299 个邻居，67 个实体，chunk 种子触达实体 32 次
            # 双向写入后与 neug 逐位相同。与 perf 臂同法（那边同样双向写入）。
            # 边类型只有 NEXT/FOLLOWS/MENTIONED/related_to/located_in/
            # works_for，不含 skos:broader|narrower，故翻倍写入不会触发
            # add_edges 里的 validate_skos_hierarchy。
            rows = []
            for e in edge_rows:
                # weight 直接下标：_build_rows 的四个建边点（NEXT / FOLLOWS /
                # MENTIONED / 抽取关系）都显式写了 weight，缺了就是行构造变了。
                for src, dst in ((e["from_id"], e["to_id"]),
                                 (e["to_id"], e["from_id"])):
                    rows.append({"source_id": src, "target_id": dst,
                                 "type": e["edge_type"], "weight": e["weight"]})
            cg.add_edges(rows)
        vecs = np.asarray([r["vec"] for r in node_rows], dtype=np.float32)
        backend_store = vs._backend_store
        if backend_store.index is None:
            # inner_product + 归一向量 == 余弦（与 neug HNSW cosine 同口径）
            backend_store.create_index(index_type="flat", metric="inner_product")
        backend_store.add_vectors(
            vecs, ids=[r["id"] for r in node_rows],
            metadata=[{"content": r["content"], "node_id": r["id"]}
                      for r in node_rows])

    # ---- 检索：向量 + 图两路，与 neug 臂同构（无全文路）----
    def search(self, question: str, top_k: int = 20, sample_id: str | None = None):
        st = self._resolve_store(sample_id, self._stores)

        from semantica.context.context_retriever import RetrievedContext

        # 必须归一：FAISS inner_product 不是尺度不变的，distance == cos*|q|，
        # 未归一会让 _faiss_ip_score 的 (1+cos)/2 映射失真（排序不受影响，
        # 但分值不再与 neug 臂可比，也可能被 clamp 吃掉）。
        qvec = np.asarray(_norm_vec(self._embed([question])[0]),
                          dtype=np.float32)
        results: list[RetrievedContext] = []

        # 路 1：FAISS flat + inner_product（归一向量上等价余弦，同 neug HNSW 口径）
        # content 直接下标：这份 metadata 就是本类 _flush 里自己写进去的
        # （{"content": r["content"], "node_id": r["id"]}），缺了即后端改了返回
        # 形状，宁可炸也不要让 _to_hits 把整路静默滤空（见 neug 臂同款注释）。
        vec_hits = st["vs"].search_vectors(qvec, k=top_k)
        for h in vec_hits:
            results.append(RetrievedContext(
                content=h["metadata"]["content"],
                score=_faiss_ip_score(h),
                source=f"vector:{h['id']}", metadata={"node_id": h["id"]},
            ))

        # 路 2：图扩展（与 neug 臂共用 _graph_leg，全路常量分同参）
        results.extend(_graph_leg(
            lambda nid: ((n["id"], n["content"])
                         for n in st["cg"].get_neighbors(nid, hops=1)),
            vec_hits))

        merged = st["rt"]._rank_and_merge(results, question)
        hits = _to_hits(merged, top_k)
        trace = SearchTrace(extra={
            "vector_hits": len(vec_hits),
            "graph_hits": sum(1 for r in results if r.source.startswith("graph:")),
            "merged": len(merged),
            "fts": "N/A (native stack has no full-text route)",
        })
        return hits, trace


# ---- semantica-neo4j 臂（server 对照）----
# 连接参数与 graphiti-neo4j 臂同源（同一本地实例 bolt://localhost:7687），
# 环境变量可覆盖。
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "testpass")
# 共享服务端上的隔离标记：本臂所有节点带此 label，setup 精确清理、绝不触碰
# graphiti-neo4j 的数据（那边靠 group_id 过滤 + 自己的 label 集）。
NEO4J_MARK = "SemanticaBench"


class SemanticaNeo4jAdapter(_SemanticaBase):
    """server 对照：原生 FAISS 向量 + 真实 neo4j 图服务端，向量 + 图两路（无全文）。

    能力面与 native 臂完全一致（向量 + 图，无 bm25 全文路），唯一变量是图后端从
    内存 ContextGraph 换成真实 neo4j server（bolt RPC）——回答 maintainer "为何不
    与 Semantica 默认后端对比"：neug（嵌入式单引擎，且独有 bm25 路）对 neo4j
    （默认 server 后端）在同能力面上的质量对比。

    与 native 臂的差异都是 neo4j 后端本身的性质（非配置差，约束 5 的唯一变量仍是
    "后端"）：
    - 图 ingest 走 facade add_nodes/add_edges（neo4j 无 COPY）；每 sample 图规模
      106-269 节点 / 268-1356 边，逐行往返可接受（质量赛道不计 ingest 延时）。
    - 边**单写**：Neo4jStore.get_neighbors 默认 direction="both"（无向 varlen），
      与 neug 臂同构；native 臂的双写只是 ContextGraph 出边单向存储的补偿。
    - 图腿需 app-id→internal-id 桥接：Neo4jStore.get_neighbors 用
      ``WHERE id(start)=$node_id``（neo4j 内部整数 id），而种子来自 FAISS（存的
      是 app 字符串 id）。facade 只在 create_relationship 里做该映射、
      get_neighbors 不做，故复用 facade 自己维护的 ``_app_node_id_map`` 桥接
      （probe_neo4j_arm 10/10：直接传 app 字符串 id 返回空，桥接后逐跳正确）。
    """
    name = "semantica-neo4j"

    def __init__(self):
        super().__init__()
        # sample_id -> {"gs": GraphStore(neo4j), "vs": VectorStore(faiss)}
        self._stores: dict[str, dict] = {}
        self._prev_head: dict[str, str] = {}
        self._retriever = None
        # sample_id -> 已建过 neo4j 节点的 app id 集合。neo4j 的 CREATE 不去重，
        # 靠它跳过复现实体的重复建点，对齐 native(self.nodes[id] upsert)/neug(PK
        # 去重)，见 _flush 注释。
        self._seen_nodes: dict[str, set] = {}

    def _new_graph_store(self):
        from semantica.graph_store.graph_store import GraphStore
        return GraphStore(backend="neo4j", uri=NEO4J_URI, user=NEO4J_USER,
                          password=NEO4J_PASSWORD)

    def setup(self, work_dir: str, cache_dir: str):
        super().setup(work_dir, cache_dir)
        from semantica.context.context_retriever import ContextRetriever
        self._retriever = ContextRetriever(hybrid_alpha=HYBRID_ALPHA)
        # 共享服务端 fresh start：分批精确清掉本臂 marker 数据（scope 到
        # :SemanticaBench，绝不触碰 graphiti-neo4j 或其它臂的节点）。
        gs = self._new_graph_store()
        gs.connect()
        try:
            self._clear_marker(gs)
        finally:
            gs.close()
        # marker 清空 <=> 图空 <=> 去重状态清零：三者同一不变量。setup 是 fresh
        # start 点，必须同步重置，否则同实例二次 run 会把建点全跳过、图空召回零。
        self._seen_nodes.clear()

    @staticmethod
    def _clear_marker(gs):
        # 先分批删边、再删点：单事务 DETACH DELETE 在大图上会撑爆事务内存池
        # （与 graphiti perf-neo4j 臂同一手法，但 scope 到本臂 marker）。
        while True:
            rows = gs.query(
                f"MATCH (n:{NEO4J_MARK})-[r]-() WITH r LIMIT 100000 DELETE r "
                "RETURN count(*) AS c")
            if not rows or int(rows[0]["c"]) == 0:
                break
        while True:
            rows = gs.query(
                f"MATCH (n:{NEO4J_MARK}) WITH n LIMIT 50000 DELETE n "
                "RETURN count(*) AS c")
            if not rows or int(rows[0]["c"]) == 0:
                break

    def teardown(self):
        # 关掉每个 sample 的 bolt 连接；数据留在服务端（下次 setup 会清）。
        for st in self._stores.values():
            st["gs"].close()
        self._stores.clear()

    def reuse_ingested(self, sessions):
        # setup 已清库：与 native 臂同法，从共享缓存（抽取产物 + 嵌入）零 API 重建。
        for s in sessions:
            if (s["sample_id"], s["session_idx"]) not in self._ingested:
                self.ingest_session(s["sample_id"], s["session_idx"],
                                    s.get("date_time", ""), s["text"])

    def _prev_head_for(self, sample_id):
        return self._prev_head.get(sample_id)

    def _set_prev_head(self, sample_id, head):
        self._prev_head[sample_id] = head

    def _ensure_store(self, sample_id: str) -> dict:
        st = self._stores.get(sample_id)
        if st is not None:
            return st
        from semantica.vector_store.vector_store import VectorStore

        gs = self._new_graph_store()
        gs.connect()
        vs = VectorStore(backend="faiss", config={"dimension": EMBED_DIM})
        st = {"gs": gs, "vs": vs}
        self._stores[sample_id] = st
        return st

    def _flush(self, sample_id, node_rows, edge_rows):
        st = self._ensure_store(sample_id)
        gs, vs = st["gs"], st["vs"]
        # 图：facade add_nodes（labels 带 marker 供精确清理）；app 字符串 id 存进
        # properties["id"]、内容存 properties["content"]，_app_node_id_map 记录
        # app→internal（add_edges 的端点解析与图腿桥接都靠它）。
        #
        # 节点去重（对齐 native/neug，约束 5）：native 的 add_nodes 按 id upsert、
        # neug 的 copy_nodes 按 PK 去重，同一 app id 恒一个节点、跨 session 的边全
        # 累积到它上面。neo4j 的 CREATE 不去重，复现实体（ent:{sample}:{name}）逐
        # session 会建多个节点，_app_node_id_map 覆盖成最后一个、早先 session 的边
        # 挂在孤立节点上图腿走不到 -> 召回系统性偏少。故按已见 app id 跳过重复建点：
        # 每 id 只 CREATE 一次，map 稳定指向它，后续 session 的边经 add_edges 端点
        # 映射自然累积到同一节点，与 native/neug 拓扑逐位一致。
        #
        # 边不去重：Neo4jStore.get_neighbors 用 `RETURN DISTINCT id(neighbor),
        # neighbor`、native get_neighbors 用 visited 集合，两边都把平行关系（复现
        # 三元组）折叠成单个邻居，重复边对图腿输出无可观测影响。
        seen_nodes = self._seen_nodes.setdefault(sample_id, set())
        new_rows = []
        for r in node_rows:
            if r["id"] not in seen_nodes:
                seen_nodes.add(r["id"])
                new_rows.append(r)
        if new_rows:
            gs.add_nodes([
                {"id": r["id"], "type": r["etype"], "content": r["content"],
                 "labels": [str(r["etype"]), NEO4J_MARK]}
                for r in new_rows
            ])
        if edge_rows:
            # 单写：Neo4jStore.get_neighbors direction="both" 天然无向（同 neug 臂），
            # 不需 native 臂那样的双向补偿。
            gs.add_edges([
                {"source_id": e["from_id"], "target_id": e["to_id"],
                 "type": e["edge_type"], "weight": e["weight"]}
                for e in edge_rows
            ])
        # 向量：与 native 臂逐位相同的 FAISS 路（flat + inner_product，归一向量上
        # 等价余弦）；metadata 带 content/node_id 供检索直接下标。
        # 故意**不**去重：native 臂每 session 也把全部 node_rows（含复现实体）塞进
        # 同一 VectorStore(backend=faiss)，重复 id 由下游 _rank_and_merge(content
        # [:100]) 与 _to_hits(node_id) 收敛。两臂走同一份 add_vectors、去重行为逐位
        # 一致；此处擅自去重反会制造与 native 的差异（约束 5）。
        vecs = np.asarray([r["vec"] for r in node_rows], dtype=np.float32)
        backend_store = vs._backend_store
        if backend_store.index is None:
            backend_store.create_index(index_type="flat", metric="inner_product")
        backend_store.add_vectors(
            vecs, ids=[r["id"] for r in node_rows],
            metadata=[{"content": r["content"], "node_id": r["id"]}
                      for r in node_rows])

    # ---- 检索：FAISS 向量 + neo4j 图两路，跨池走原生 _rank_and_merge（同 native）----
    def search(self, question: str, top_k: int = 20, sample_id: str | None = None):
        st = self._resolve_store(sample_id, self._stores)

        from semantica.context.context_retriever import RetrievedContext

        qvec = np.asarray(_norm_vec(self._embed([question])[0]),
                          dtype=np.float32)
        results: list[RetrievedContext] = []

        # 路 1：FAISS（与 native 臂同一实现，无 bm25 融合——这正是与 neug 臂的能力差）
        vec_hits = st["vs"].search_vectors(qvec, k=top_k)
        for h in vec_hits:
            results.append(RetrievedContext(
                content=h["metadata"]["content"],
                score=_faiss_ip_score(h),
                source=f"vector:{h['id']}", metadata={"node_id": h["id"]},
            ))

        # 路 2：neo4j 图扩展（与 native/neug 共用 _graph_leg，全路常量分同参）。
        # neighbors_of 做 app-id→internal-id 桥接：种子是 FAISS 存的 app 字符串 id，
        # Neo4jStore.get_neighbors 要内部整数 id；回来再把 properties["id"]/content
        # 映射成 (app_id, content) 交给 _graph_leg（与 neug 臂的 (id, content) 同形）。
        gs = st["gs"]
        idmap = gs._app_node_id_map

        def neighbors_of(app_id):
            internal = idmap.get(app_id, app_id)
            for n in gs.get_neighbors(internal, depth=1):
                props = n.get("properties", {}) or {}
                yield (props.get("id", n.get("id")), props.get("content", ""))

        results.extend(_graph_leg(neighbors_of, vec_hits))

        merged = self._retriever._rank_and_merge(results, question)
        hits = _to_hits(merged, top_k)
        trace = SearchTrace(extra={
            "vector_hits": len(vec_hits),
            "graph_hits": sum(1 for r in results if r.source.startswith("graph:")),
            "merged": len(merged),
            "fts": "N/A (neo4j arm: FAISS vector + neo4j graph, no full-text route)",
        })
        return hits, trace
