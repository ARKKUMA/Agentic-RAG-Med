import time, sys, threading, queue
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer

BATCH_DIR    = Path(r'd:\Rag-Med\pipeline_output\batches_full')
TEST_BATCHES = 5
EMBED_BATCH  = 512
PREFETCH     = 2      # 提前预读批次数

files = sorted(BATCH_DIR.glob('batch_*.parquet'))[:TEST_BATCHES]

print('Loading model...')
model = SentenceTransformer(
    'BAAI/bge-base-en-v1.5',
    device='cuda' if torch.cuda.is_available() else 'cpu',
    model_kwargs={'torch_dtype': torch.float16},
)
print(f'Device: {next(model.parameters()).device}  dtype: {next(model.parameters()).dtype}')

# 预读线程
def _reader(files, q):
    for f in files:
        q.put((f, pd.read_parquet(f, columns=['text'])))
    q.put(None)

pq = queue.Queue(maxsize=PREFETCH)
threading.Thread(target=_reader, args=(files, pq), daemon=True).start()

total_chunks = 0
times_encode = []
t_wall_start = time.time()

i = 0
while True:
    item = pq.get()
    if item is None:
        break
    bf, df = item
    texts = df['text'].tolist()
    n = len(texts)

    t0 = time.time()
    embs = model.encode(
        texts,
        batch_size           = EMBED_BATCH,
        normalize_embeddings = True,
        show_progress_bar    = False,
        convert_to_numpy     = True,
    )
    elapsed = time.time() - t0

    total_chunks   += n
    times_encode.append(elapsed)
    print(f'  batch {i+1}: {n:,} chunks  encode={elapsed:.1f}s  {n/elapsed:.0f} chunks/s')
    i += 1

wall_total    = time.time() - t_wall_start
avg_speed     = total_chunks / sum(times_encode)   # pure encode speed
wall_speed    = total_chunks / wall_total           # end-to-end speed (encode + IO overlap)

total_batches     = len(sorted(BATCH_DIR.glob('batch_*.parquet')))
total_chunks_est  = total_batches * (total_chunks / TEST_BATCHES)
eta_encode_h      = total_chunks_est / avg_speed  / 3600
eta_wall_h        = total_chunks_est / wall_speed / 3600

print()
print(f'Pure encode speed  : {avg_speed:,.0f} chunks/s')
print(f'End-to-end speed   : {wall_speed:,.0f} chunks/s  (encode + IO prefetch)')
print(f'Total batches      : {total_batches}')
print(f'Est. total chunks  : {total_chunks_est:,.0f}')
print(f'ETA (encode only)  : {eta_encode_h:.1f} h')
print(f'ETA (wall, +10%DB) : {eta_wall_h * 1.1:.1f} h  <- use this')