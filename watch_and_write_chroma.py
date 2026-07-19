"""
watch_and_write_chroma.py

监听 emb_cache/ 目录，发现新的 .npy 文件即写入 ChromaDB，写入后删除缓存。
与 Phase 1 并行运行，GPU encode 和 ChromaDB 写入完全解耦。

用法：
    python watch_and_write_chroma.py           # 正常监听模式
    python watch_and_write_chroma.py --rebuild # 从 ChromaDB 重建 checkpoint 后再监听

--rebuild：checkpoint 文件丢失时，逐批取第一个 chunk_id 查询 ChromaDB，
           重建已完成批次记录，之后继续正常监听。
"""

import os, json, time, signal, argparse
import numpy as np
import pandas as pd
import chromadb
from pathlib import Path

# ── 配置（与 notebook 保持一致）────────────────────────────────
BATCH_DIR    = Path('/root/autodl-tmp/batches_full')
DB_DIR       = Path('/root/autodl-tmp/pipeline_output/chroma_db')
COLLECTION   = 'pmc_full'
CACHE_DIR    = BATCH_DIR.parent / 'emb_cache'
CKPT_PATH    = BATCH_DIR / f'lc_embed_checkpoint_{COLLECTION}.json'
CHROMA_BATCH = 5000   # 每次写入 ChromaDB 的条数
POLL_INTERVAL = 10    # 轮询间隔（秒）
IDLE_TIMEOUT  = 300   # 无新文件超过此秒数后自动退出（5 分钟）
# ─────────────────────────────────────────────────────────────

STR_FIELDS = ['doc_id','chunk_type','section_title','section_path','imrad_type',
              'split_strategy','pmc_id','pmid','doi','journal','article_type',
              'abstract_source','source_url','source_title']
INT_FIELDS = ['pub_year','token_count','chunk_index','total_chunks']

def _batch_make_metadata(df: pd.DataFrame) -> list[dict]:
    tmp = {}
    for f in STR_FIELDS:
        tmp[f] = df[f].fillna('').astype(str).tolist() if f in df.columns else ['']*len(df)
    for f in INT_FIELDS:
        tmp[f] = pd.to_numeric(df[f], errors='coerce').fillna(0).astype(int).tolist() \
                 if f in df.columns else [0]*len(df)
    all_f = STR_FIELDS + INT_FIELDS
    return [{f: tmp[f][i] for f in all_f} for i in range(len(df))]

def _load_checkpoint() -> set:
    if CKPT_PATH.exists():
        return set(json.loads(CKPT_PATH.read_text()))
    return set()

def _save_checkpoint(done: set):
    CKPT_PATH.write_text(json.dumps(sorted(done)))

def _log(msg: str):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)

def _write_one(npy_file: Path, collection, chroma_done: set) -> int:
    """写入一个 .npy 文件到 ChromaDB，成功后删除。返回写入 chunk 数，失败返回 0。"""
    stem         = npy_file.stem.replace('_emb', '')
    parquet_file = BATCH_DIR / f'{stem}.parquet'

    if str(parquet_file) in chroma_done:
        npy_file.unlink()
        return 0

    if not parquet_file.exists():
        _log(f'  ⚠️  找不到 {parquet_file.name}，跳过')
        return 0

    # 加载 embedding（float16 → float32）
    embs = np.load(npy_file).astype(np.float32)

    # 与 Phase 1 保持相同顺序（不排序）
    df        = pd.read_parquet(parquet_file)
    texts     = df['text'].tolist()
    ids       = df['chunk_id'].tolist()
    metadatas = _batch_make_metadata(df)

    if len(embs) != len(df):
        _log(f'  ⚠️  {npy_file.name}: 向量数 {len(embs)} ≠ chunk 数 {len(df)}，跳过')
        return 0

    # 分块写入 ChromaDB（带重试）
    t_write = time.time()
    for start in range(0, len(df), CHROMA_BATCH):
        end = min(start + CHROMA_BATCH, len(df))
        for attempt in range(5):
            try:
                collection.add(
                    ids        = ids[start:end],
                    embeddings = embs[start:end].tolist(),
                    documents  = texts[start:end],
                    metadatas  = metadatas[start:end],
                )
                break
            except Exception as e:
                if attempt == 4:
                    _log(f'  ChromaDB 写入失败（已重试 5 次）: {e}')
                    return 0
                wait = 5 * (2 ** attempt)
                _log(f'  写入失败 ({attempt+1}/5)，{wait}s 后重试: {e}')
                time.sleep(wait)

    write_s = time.time() - t_write
    chroma_done.add(str(parquet_file))
    _save_checkpoint(chroma_done)
    npy_file.unlink()
    _log(f'  {parquet_file.name}  {len(df):,} chunks  {write_s:.1f}s  .npy 已删除')
    return len(df)


def _rebuild_checkpoint(collection) -> set:
    """逐批取第一个 chunk_id 查 ChromaDB，重建已完成批次的 checkpoint。"""
    batch_files = sorted(BATCH_DIR.glob('batch_*.parquet'))
    _log(f'开始重建 checkpoint，共 {len(batch_files):,} 个批次...')
    done = set()
    for i, bf in enumerate(batch_files):
        try:
            first_id = pd.read_parquet(bf, columns=['chunk_id'])['chunk_id'].iloc[0]
            result   = collection.get(ids=[str(first_id)], include=[])
            if result['ids']:
                done.add(str(bf))
        except Exception as e:
            _log(f'  ⚠️  {bf.name} 检查失败: {e}')
        if (i + 1) % 100 == 0 or (i + 1) == len(batch_files):
            _log(f'  [{i+1}/{len(batch_files)}]  已确认完成: {len(done):,}')
    _save_checkpoint(done)
    _log(f'checkpoint 重建完成，已完成批次: {len(done):,} / {len(batch_files):,}')
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rebuild', action='store_true',
                        help='从 ChromaDB 重建 checkpoint（checkpoint 丢失时使用）')
    args = parser.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _log('=== ChromaDB 写入守护进程启动 ===')
    _log(f'CACHE_DIR     : {CACHE_DIR}')
    _log(f'DB_DIR        : {DB_DIR}')
    _log(f'POLL_INTERVAL : {POLL_INTERVAL}s   IDLE_TIMEOUT : {IDLE_TIMEOUT}s')

    # 初始化 ChromaDB
    DB_DIR.mkdir(parents=True, exist_ok=True)
    client     = chromadb.PersistentClient(path=str(DB_DIR))
    collection = client.get_or_create_collection(
        name     = COLLECTION,
        metadata = {'hnsw:space': 'cosine'},
    )
    _log(f'ChromaDB 连接成功，当前向量数: {collection.count():,}')

    if args.rebuild:
        chroma_done = _rebuild_checkpoint(collection)
        print('─' * 60)
    else:
        chroma_done = _load_checkpoint()
    processed     = 0
    t_last_new    = time.time()   # 最近一次发现新文件的时间
    running       = True

    def _handle_sigint(sig, frame):
        nonlocal running
        _log('收到 Ctrl+C，等待当前批次完成后退出...')
        running = False

    signal.signal(signal.SIGINT, _handle_sigint)
    _log(f'已写入 ChromaDB: {len(chroma_done):,}  开始监听...')
    print('─' * 60)

    t0 = time.time()
    total_written = 0

    while running:
        # 扫描新的 .npy 文件（排除已处理）
        npy_files = sorted(
            f for f in CACHE_DIR.glob('*_emb.npy')
            if str(BATCH_DIR / (f.stem.replace('_emb', '') + '.parquet'))
               not in chroma_done
        )

        if npy_files:
            t_last_new = time.time()
            for npy in npy_files:
                if not running:
                    break
                t_batch = time.time()
                stem = npy.stem.replace('_emb', '')
                _log(f'[{processed+1}] 发现 {npy.name}，开始写入...')
                n = _write_one(npy, collection, chroma_done)
                if n > 0:
                    processed     += 1
                    total_written += n
                    elapsed        = time.time() - t0
                    speed          = total_written / elapsed if elapsed > 0 else 0
                    _log(f'  [{processed}]  累计={total_written:,}  {speed:.0f} chunks/s')
        else:
            idle_s = time.time() - t_last_new
            if idle_s > IDLE_TIMEOUT:
                _log(f'已 {IDLE_TIMEOUT}s 无新缓存文件，自动退出。')
                break
            remaining_idle = IDLE_TIMEOUT - idle_s
            print(f'\r  等待新缓存文件... (空闲 {idle_s:.0f}s / {IDLE_TIMEOUT}s)',
                  end='', flush=True)
            time.sleep(POLL_INTERVAL)

    print()
    print('─' * 60)
    _log(f'退出。本次共写入 {processed} 个批次，{total_written:,} chunks')
    _log(f'ChromaDB 当前向量总数: {collection.count():,}')


if __name__ == '__main__':
    main()
