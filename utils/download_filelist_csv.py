#!/usr/bin/env python3
"""
爬取 PMC oa_comm 下所有 filelist.csv 文件并合并。

NCBI FTP 结构：
  https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_comm/xml/
    ├── oa_comm_xml.PMC000xxxxxx.baseline.2026-01-23.filelist.csv
    ├── oa_comm_xml.PMC000xxxxxx.baseline.2026-01-23.tar.gz
    ├── oa_comm_xml.incr.2026-01-24.filelist.csv
    ├── oa_comm_xml.incr.2026-01-24.tar.gz
    └── ...

用法：
  # 下载所有 filelist.csv 到 ./filelist_csv/ 并合并
  python download_filelist_csv.py

  # 自定义输出目录，不合并
  python download_filelist_csv.py -o ./my_dir --no-merge

  # 只下载 baseline（不含增量）
  python download_filelist_csv.py --baseline-only

  # dry-run：只列不下
  python download_filelist_csv.py --dry-run
"""

import argparse
import csv
import io
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin

import requests

DEFAULT_BASE = "https://ftp.ncbi.nlm.nih.gov/pub/pmc/deprecated/oa_bulk/oa_comm/xml/"
# 兼容两种常见文件名模式：
#   oa_comm_xml.PMC000xxxxxx.baseline.2026-01-23.filelist.csv  (带包名前缀)
#   filelist.csv  (简单名)
FILELIST_RE = re.compile(r'href="([^"?][^"]*filelist[^"]*\.csv)"', re.IGNORECASE)
HEADERS = {"User-Agent": "pmc-filelist-downloader/1.0"}
CHUNK = 1 << 20  # 1 MiB


def scrape_filelist_urls(base_url, session, baseline_only=False, incr_only=False,
                         verbose=False):
    """从目录索引页解析所有 filelist.csv 链接。"""
    if verbose:
        print(f"[scan] {base_url}")

    r = session.get(base_url, headers=HEADERS, timeout=60)
    r.raise_for_status()

    names = sorted(set(FILELIST_RE.findall(r.text)))
    if verbose:
        print(f"[scan] 发现 {len(names)} 个 filelist.csv")

    urls = []
    for name in names:
        if baseline_only and "incr" in name.lower():
            continue
        if incr_only and "baseline" in name.lower():
            continue
        urls.append(urljoin(base_url, name))

    return sorted(urls)


def download_one(url, out_dir, session, retries=5):
    """下载单个 filelist.csv，支持断点续传与重试。"""
    fname = url.rsplit("/", 1)[-1]
    dest = os.path.join(out_dir, fname)

    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return fname, "skip-exists", dest

    for attempt in range(1, retries + 1):
        try:
            resume_at = os.path.getsize(dest) if os.path.exists(dest) else 0
            hdrs = dict(HEADERS)
            if resume_at:
                hdrs["Range"] = f"bytes={resume_at}-"

            with session.get(url, headers=hdrs, stream=True, timeout=60) as r:
                if r.status_code == 416:
                    break
                mode = "ab" if (resume_at and r.status_code == 206) else "wb"
                if mode == "wb":
                    resume_at = 0
                r.raise_for_status()

                with open(dest, mode) as f:
                    for chunk in r.iter_content(CHUNK):
                        if chunk:
                            f.write(chunk)

            return fname, "ok", dest

        except requests.RequestException as e:
            wait = min(2 ** attempt, 60)
            print(f"  [{fname}] attempt {attempt}/{retries} failed: {e} -> retry in {wait}s")
            time.sleep(wait)

    return fname, "FAILED", None


def merge_csvs(csv_paths, out_path):
    """合并多个 filelist.csv 为一个，去重，保留表头一次。"""
    seen_ids = set()
    header = None
    rows = []

    for path in sorted(csv_paths):
        if not path or not os.path.exists(path):
            continue
        try:
            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if header is None:
                    header = reader.fieldnames
                for row in reader:
                    # 用文件路径作去重键
                    key = row.get("File", "") or row.get("file", "")
                    if key and key in seen_ids:
                        continue
                    seen_ids.add(key)
                    rows.append(row)
        except Exception as e:
            print(f"  [warn] 读取 {path} 失败: {e}")

    if not header or not rows:
        print("[warn] 没有有效数据可合并。")
        return 0

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


def main():
    ap = argparse.ArgumentParser(description="PMC oa_comm filelist.csv 批量下载器")
    ap.add_argument("--base", default=DEFAULT_BASE,
                    help="目录索引 URL（默认 deprecated/oa_bulk/oa_comm/xml/）")
    ap.add_argument("-o", "--out", default="./filelist_csv",
                    help="filelist.csv 存放目录（默认 ./filelist_csv）")
    ap.add_argument("--merged", default="../data/oa_comm_filelist_merged.csv",
                    help="合并后输出路径（默认 ../data/oa_comm_filelist_merged.csv，相对 utils/ 运行）")
    ap.add_argument("--no-merge", action="store_true",
                    help="只下载，不合并")
    ap.add_argument("--baseline-only", action="store_true",
                    help="只下载 baseline 包的 filelist.csv")
    ap.add_argument("--incr-only", action="store_true",
                    help="只下载增量包的 filelist.csv")
    ap.add_argument("-j", "--jobs", type=int, default=4,
                    help="并发数（默认 4）")
    ap.add_argument("--dry-run", action="store_true",
                    help="只列出链接，不下载")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="详细输出")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    session = requests.Session()

    print(f"[1/3] 扫描目录: {args.base}")
    try:
        urls = scrape_filelist_urls(
            args.base, session,
            baseline_only=args.baseline_only,
            incr_only=args.incr_only,
            verbose=args.verbose,
        )
    except requests.RequestException as e:
        print(f"[error] 无法访问 {args.base}: {e}", file=sys.stderr)
        sys.exit(1)

    if not urls:
        print("[warn] 未找到任何 filelist.csv。请检查 --base URL 或网络连接。",
              file=sys.stderr)
        sys.exit(1)

    print(f"[1/3] 找到 {len(urls)} 个 filelist.csv：")
    for u in urls:
        print(f"  - {u.rsplit('/', 1)[-1]}")

    if args.dry_run:
        print("\n[dry-run] 退出，未下载任何文件。")
        return

    print(f"\n[2/3] 下载到 {args.out}/ (并发={args.jobs})")
    total = len(urls)
    results = []

    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(download_one, u, args.out, session): u for u in urls}
        for i, fut in enumerate(as_completed(futs), 1):
            fname, status, dest = fut.result()
            print(f"  [{i:>3}/{total}] {fname}: {status}")
            results.append((fname, status, dest))

    failed = [r for r in results if r[1] == "FAILED"]
    success_paths = [r[2] for r in results if r[2] is not None]
    print(f"\n完成：{total - len(failed)}/{total} 成功" +
          (f"，{len(failed)} 失败" if failed else ""))

    if args.no_merge:
        return

    print(f"\n[3/3] 合并 CSV -> {args.merged}")
    n = merge_csvs(success_paths, args.merged)
    print(f"  合并完成：共 {n:,} 条记录写入 {args.merged}")

    if failed:
        print("\n失败列表（重跑会自动跳过已下载的文件）：")
        for fname, _, __ in failed:
            print(f"  - {fname}")


if __name__ == "__main__":
    main()
