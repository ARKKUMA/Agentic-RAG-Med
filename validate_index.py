"""
validate_index.py  —  ChromaDB 索引质量验证
用法：python validate_index.py
"""

import sys, json, time, textwrap
sys.stdout.reconfigure(encoding='utf-8')

import chromadb
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ── 配置 ─────────────────────────────────────────────────────────
DB_DIR     = Path('d:/Rag-Med/pipeline_output/chroma_db')
COLLECTION = 'pmc_full'
MODEL      = 'BAAI/bge-base-en-v1.5'
BGE_PREFIX = 'Represent this sentence for searching relevant passages: '
# ─────────────────────────────────────────────────────────────────

PASS = '  ✓'
FAIL = '  ✗'
WARN = '  !'

results = []

def check(name, ok, detail=''):
    tag = PASS if ok else FAIL
    print(f'{tag}  {name}')
    if detail:
        print(f'      {detail}')
    results.append((name, ok))

def section(title):
    print(f'\n{"─"*60}')
    print(f'  {title}')
    print(f'{"─"*60}')


# ══════════════════════════════════════════════════════════════════
# 0. 连接
# ══════════════════════════════════════════════════════════════════
section('0. 连接 ChromaDB')

try:
    client     = chromadb.PersistentClient(path=str(DB_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION, metadata={'hnsw:space': 'cosine'}
    )
    n_total = collection.count()
    check('ChromaDB 连接成功', True, f'DB_DIR={DB_DIR}')
    check('集合存在', True, f'collection={COLLECTION}')
except Exception as e:
    check('ChromaDB 连接', False, str(e))
    sys.exit(1)

print(f'\n  加载嵌入模型 {MODEL} ...')
try:
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model  = SentenceTransformer(MODEL, device=device)
    if device == 'cuda':
        model.half()
    check('模型加载成功', True, f'device={device}')
except Exception as e:
    check('模型加载', False, str(e))
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════
# 1. 基础统计验证
# ══════════════════════════════════════════════════════════════════
section('1. 基础统计验证')

check('向量总数 > 0', n_total > 0, f'{n_total:,} 个向量')

# 抽取 200 个样本检查元数据
sample = collection.get(limit=200, include=['metadatas', 'documents'])
metas  = sample['metadatas']
docs   = sample['documents']

# 必填字段完整性
required_str = ['chunk_type', 'pmc_id', 'journal']
required_int = ['pub_year', 'token_count']
for field in required_str + required_int:
    missing = sum(1 for m in metas if not m.get(field))
    check(f'元数据字段 [{field}] 完整', missing == 0,
          f'缺失 {missing}/200' if missing else '200/200 完整')

# 文档内容非空
empty_docs = sum(1 for d in docs if not d or not d.strip())
check('文档内容非空', empty_docs == 0,
      f'{empty_docs} 条空文档' if empty_docs else '全部非空')

# pub_year 合理范围 1950-2025
bad_year = sum(1 for m in metas if not (1950 <= int(m.get('pub_year', 0) or 0) <= 2025))
check('发表年份在合理范围内 [1950-2025]', bad_year == 0,
      f'{bad_year} 条异常年份' if bad_year else '全部正常')

# token_count 合理范围 1-1024
bad_tok = sum(1 for m in metas if not (1 <= int(m.get('token_count', 0) or 0) <= 1024))
check('token_count 在合理范围 [1-1024]', bad_tok == 0,
      f'{bad_tok} 条异常' if bad_tok else '全部正常')

# 统计摘要
chunk_types = {}
for m in metas:
    ct = m.get('chunk_type', 'unknown')
    chunk_types[ct] = chunk_types.get(ct, 0) + 1
imrad_types = {}
for m in metas:
    it = m.get('imrad_type', '') or 'N/A'
    imrad_types[it] = imrad_types.get(it, 0) + 1

print(f'\n  chunk_type 分布（样本 200）:')
for k, v in sorted(chunk_types.items(), key=lambda x: -x[1]):
    print(f'    {k:<20} {v:>4}')
print(f'\n  imrad_type 分布（样本 200）:')
for k, v in sorted(imrad_types.items(), key=lambda x: -x[1]):
    print(f'    {k:<20} {v:>4}')


# ══════════════════════════════════════════════════════════════════
# 2. 相似性检索验证（自相似性）
# ══════════════════════════════════════════════════════════════════
section('2. 相似性检索验证')

# 从索引中随机取 5 条文档，查询自身，期望 rank-1 就是自己
test_items = [(sample['ids'][i], docs[i]) for i in range(0, 200, 40)]

print('  自相似性测试（从索引取文本查自身，rank-1 应返回自身）:')
self_sim_pass = 0
for doc_id, text in test_items:
    query_vec = model.encode(
        [BGE_PREFIX + text[:512]],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).tolist()
    res = collection.query(
        query_embeddings=query_vec,
        n_results=3,
        include=['distances', 'documents'],
    )
    top_id   = res['ids'][0][0]
    top_dist = res['distances'][0][0]
    top_sim  = 1 - top_dist
    is_self  = (top_id == doc_id)
    if is_self:
        self_sim_pass += 1
    status = PASS if is_self else WARN
    print(f'  {status}  sim={top_sim:.4f}  '
          f'{"rank-1 = 自身" if is_self else f"rank-1 ≠ 自身 ({top_id[:20]}...)"}')
    print(f'        文本: {text[:80].strip()}...')

check('自相似性：rank-1 命中率 ≥ 80%',
      self_sim_pass >= 4,
      f'{self_sim_pass}/5 命中')

# 语义相关性测试
print()
semantic_queries = [
    ('CRISPR gene editing therapy',          ['gene', 'editing', 'CRISPR', 'therapy']),
    ('COVID-19 vaccine efficacy trial',      ['vaccine', 'COVID', 'efficacy', 'trial']),
    ('machine learning drug discovery',      ['machine learning', 'drug', 'model']),
    ('single cell RNA sequencing protocol',  ['RNA', 'sequencing', 'cell', 'protocol']),
]
print('  语义相关性测试（top-3 结果是否包含关键词）:')
for query, keywords in semantic_queries:
    qvec = model.encode(
        [BGE_PREFIX + query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).tolist()
    res = collection.query(
        query_embeddings=qvec,
        n_results=3,
        include=['documents', 'distances'],
    )
    combined = ' '.join(res['documents'][0]).lower()
    hits = [kw for kw in keywords if kw.lower() in combined]
    ok   = len(hits) >= len(keywords) // 2
    sims = [f'{1-d:.3f}' for d in res['distances'][0]]
    check(f'查询 [{query[:40]}]',
          ok,
          f'关键词命中 {len(hits)}/{len(keywords)}  sims={sims}')
    if not ok:
        print(f'        top-1: {res["documents"][0][0][:100]}')


# ══════════════════════════════════════════════════════════════════
# 3. 边界情况验证
# ══════════════════════════════════════════════════════════════════
section('3. 边界情况验证')

# 3a. 空查询（单个空格）
try:
    qvec = model.encode(
        [BGE_PREFIX + ' '],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).tolist()
    res = collection.query(query_embeddings=qvec, n_results=1, include=['distances'])
    check('空查询不崩溃', True, f'返回 {len(res["ids"][0])} 条结果')
except Exception as e:
    check('空查询不崩溃', False, str(e))

# 3b. 超长查询（截断到 512 tokens ≈ 2000 chars）
long_text = ('The treatment of patients with advanced non-small-cell lung cancer '
             'using checkpoint inhibitors combined with chemotherapy regimens ') * 20
try:
    qvec = model.encode(
        [BGE_PREFIX + long_text[:2000]],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).tolist()
    res = collection.query(query_embeddings=qvec, n_results=1, include=['distances'])
    check('超长查询不崩溃', True, f'输入 {len(long_text[:2000])} 字符，返回 {len(res["ids"][0])} 条')
except Exception as e:
    check('超长查询不崩溃', False, str(e))

# 3c. 元数据过滤：methods + pub_year >= 2020
try:
    qvec = model.encode(
        [BGE_PREFIX + 'RNA sequencing single cell protocol'],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).tolist()
    res = collection.query(
        query_embeddings=qvec,
        n_results=5,
        where={'$and': [{'imrad_type': {'$eq': 'methods'}},
                        {'pub_year':   {'$gte': 2020}}]},
        include=['metadatas', 'distances'],
    )
    n_res = len(res['ids'][0])
    years_ok = all(m['pub_year'] >= 2020 for m in res['metadatas'][0])
    imrad_ok = all(m['imrad_type'] == 'methods' for m in res['metadatas'][0])
    check('元数据过滤返回结果', n_res > 0, f'返回 {n_res} 条')
    check('过滤结果 pub_year >= 2020', years_ok,
          f'年份: {[m["pub_year"] for m in res["metadatas"][0]]}')
    check('过滤结果 imrad_type == methods', imrad_ok,
          f'类型: {[m["imrad_type"] for m in res["metadatas"][0]]}')
except Exception as e:
    check('元数据过滤查询', False, str(e))

# 3d. 向量维度验证
try:
    peek = collection.get(limit=1, include=['embeddings'])
    emb_list = peek['embeddings']
    if emb_list is not None and len(emb_list) > 0:
        dim = len(emb_list[0])
        check('向量维度 = 768', dim == 768, f'实际维度: {dim}')
    else:
        check('向量维度检查', False, '无法获取 embedding（ChromaDB 默认不存储，跳过）')
except Exception as e:
    check('向量维度检查', False, str(e))


# ══════════════════════════════════════════════════════════════════
# 汇总
# ══════════════════════════════════════════════════════════════════
section('验证汇总')

passed = sum(1 for _, ok in results if ok)
failed = sum(1 for _, ok in results if not ok)
total  = len(results)

print(f'  总计: {total} 项检查   ✓ {passed} 通过   ✗ {failed} 失败')
print(f'  ChromaDB 向量总数: {n_total:,}')
print()

if failed:
    print('  失败项:')
    for name, ok in results:
        if not ok:
            print(f'    ✗  {name}')
else:
    print('  所有检查通过！索引质量正常。')

sys.exit(0 if failed == 0 else 1)
