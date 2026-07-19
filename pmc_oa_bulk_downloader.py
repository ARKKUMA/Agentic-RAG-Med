#!/usr/bin/env python3
"""
PMC oa_bulk 批量下载器 / PMC oa_bulk bulk downloader

特点 / Features:
  - 默认只爬取 oa_comm/xml/ 目录（商用许可 XML）
  - 运行时解析目录索引（不写死文件名）/ scrapes the Apache index at runtime
  - 断点续传 / resumable downloads via HTTP Range
  - 失败重试 + 指数退避 / retry with exponential backoff
  - 可选 md5 校验 / optional md5 verification (.md5 sibling files)
  - 可选自动解包 / optional auto-extract

用法示例 / Usage:
  # 下载 oa_comm XML（默认行为）
  python pmc_oa_bulk_downloader.py -o ./pmc_xml

  # 下完后自动解包
  python pmc_oa_bulk_downloader.py -o ./pmc_xml --extract

  # 先看会下哪些（dry-run），不实际下载
  python pmc_oa_bulk_downloader.py --dry-run

  # 切换到其他目录（如 oa_noncomm）
  python pmc_oa_bulk_downloader.py --base https://ftp.ncbi.nlm.nih.gov/pub/pmc/deprecated/oa_bulk/oa_noncomm/xml/ -o ./pmc_xml
"""

import argparse
import hashlib
import os
import re
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin

import requests

DEFAULT_BASE = "https://ftp.ncbi.nlm.nih.gov/pub/pmc/deprecated/oa_bulk/oa_comm/xml/"
ARCHIVE_RE = re.compile(r'href="([^"?][^"]*\.(?:tar\.gz|tgz))"', re.IGNORECASE)
# 子目录链接：以 / 结尾，排除父目录、绝对根路径与排序查询
DIR_RE = re.compile(r'href="([^"?:/][^":]*/)"', re.IGNORECASE)
CHUNK = 1 << 20  # 1 MiB
HEADERS = {"User-Agent": "pmc-oa-bulk-downloader/1.0"}


def _fmt_match(url, fmt):
    """格式匹配：看文件名 token 或路径里的 xml/ txt/ 子目录。"""
    if fmt is None:
        return True
    low = url.lower()
    if fmt == "xml":
        return ".xml.tar.gz" in low or "_xml." in low or "/xml/" in low
    if fmt == "txt":
        return ".txt.tar.gz" in low or "_txt." in low or "/txt/" in low
    return True


def list_archives(base_url, fmt, license_filter, session, verbose=False,
                  depth=0, max_depth=3, _seen=None):
    """递归解析目录索引，返回匹配的压缩包绝对 URL 列表。
    既支持扁平结构（顶层直接放 .tar.gz），也支持嵌套结构
    （oa_comm/xml/...tar.gz 这类子目录）。"""
    if _seen is None:
        _seen = set()
    if base_url in _seen or depth > max_depth:
        return []
    _seen.add(base_url)

    r = session.get(base_url, headers=HEADERS, timeout=60)
    r.raise_for_status()

    files = sorted(set(ARCHIVE_RE.findall(r.text)))
    subdirs = sorted(set(DIR_RE.findall(r.text)))
    if verbose:
        print(f"[scan] {base_url}  -> {len(files)} archive(s), "
              f"{len(subdirs)} subdir(s)")
        for d in subdirs:
            print(f"        dir: {d}")

    selected = []
    for name in files:
        url = urljoin(base_url, name)
        if not _fmt_match(url, fmt):
            continue
        if license_filter and license_filter.lower() not in url.lower():
            continue
        selected.append(url)

    # 顶层没找到文件，或允许递归时，进子目录继续找
    for d in subdirs:
        # license 过滤可在目录层提前剪枝（子目录名常含 comm/noncomm/other）
        sub_url = urljoin(base_url, d)
        selected += list_archives(sub_url, fmt, license_filter, session,
                                  verbose, depth + 1, max_depth, _seen)
    return selected


def remote_md5(url, session):
    """尝试拿同名 .md5；拿不到返回 None。支持两种常见格式。"""
    try:
        r = session.get(url + ".md5", headers=HEADERS, timeout=30)
        if r.status_code != 200:
            return None
        text = r.text.strip()
        # 格式1: "MD5(filename)= <hash>"
        m = re.search(r"=\s*([0-9a-fA-F]{32})", text)
        if m:
            return m.group(1).lower()
        # 格式2: "<hash>  filename"
        m = re.match(r"([0-9a-fA-F]{32})", text)
        if m:
            return m.group(1).lower()
    except requests.RequestException:
        pass
    return None


def file_md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def download_one(url, out_dir, session, verify=True, retries=5):
    fname = url.rsplit("/", 1)[-1]
    dest = os.path.join(out_dir, fname)
    expected = remote_md5(url, session) if verify else None

    # 已存在且校验通过则跳过
    if os.path.exists(dest) and expected and file_md5(dest) == expected:
        return fname, "skip-verified"

    for attempt in range(1, retries + 1):
        try:
            resume_at = os.path.getsize(dest) if os.path.exists(dest) else 0
            hdrs = dict(HEADERS)
            if resume_at:
                hdrs["Range"] = f"bytes={resume_at}-"

            with session.get(url, headers=hdrs, stream=True, timeout=120) as r:
                if r.status_code == 416:  # range 不可满足，多半已下完
                    break
                # 服务器忽略 Range 返回 200 => 从头写
                mode = "ab" if (resume_at and r.status_code == 206) else "wb"
                if mode == "wb":
                    resume_at = 0
                r.raise_for_status()

                total = r.headers.get("Content-Length")
                total = int(total) + resume_at if total else None
                done = resume_at
                last = time.time()
                with open(dest, mode) as f:
                    for chunk in r.iter_content(CHUNK):
                        if not chunk:
                            continue
                        f.write(chunk)
                        done += len(chunk)
                        if time.time() - last > 2:
                            pct = f"{done/total*100:5.1f}%" if total else f"{done/1e6:.0f}MB"
                            print(f"  [{fname}] {pct}", end="\r", flush=True)
                            last = time.time()
            print(" " * 60, end="\r")

            if expected:
                if file_md5(dest) == expected:
                    return fname, "ok-verified"
                # 校验失败，删了重来
                os.remove(dest)
                raise ValueError("md5 mismatch")
            return fname, "ok"

        except (requests.RequestException, ValueError) as e:
            wait = min(2 ** attempt, 60)
            print(f"  [{fname}] attempt {attempt}/{retries} failed: {e} -> retry in {wait}s")
            time.sleep(wait)

    return fname, "FAILED"


def extract(out_dir):
    for fn in sorted(os.listdir(out_dir)):
        if not fn.endswith(".tar.gz"):
            continue
        path = os.path.join(out_dir, fn)
        print(f"extracting {fn} ...")
        with tarfile.open(path, "r:gz") as tf:
            tf.extractall(out_dir)  # PMC 包内是相对路径，安全


def main():
    ap = argparse.ArgumentParser(description="PMC oa_bulk batch downloader")
    ap.add_argument("--base", default=DEFAULT_BASE,
                    help="目录索引 URL（默认 oa_comm/xml/）")
    ap.add_argument("--fmt", choices=["xml", "txt", "all"], default="xml",
                    help="下载格式（默认 xml）")
    ap.add_argument("--license", dest="license_filter", default=None,
                    help="额外的 license 过滤子串（默认不过滤，已由 --base 锁定 oa_comm）")
    ap.add_argument("-o", "--out", default="./pmc_oa_bulk", help="输出目录")
    ap.add_argument("-j", "--jobs", type=int, default=2,
                    help="并发文件数（单文件很大，2~3 即可）")
    ap.add_argument("--no-verify", action="store_true", help="跳过 md5 校验")
    ap.add_argument("--extract", action="store_true", help="下完后解包")
    ap.add_argument("--dry-run", action="store_true", help="只列出将下载的文件")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="打印目录扫描过程（调试用）")
    ap.add_argument("--max-depth", type=int, default=3,
                    help="子目录递归深度上限")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    session = requests.Session()

    fmt = None if args.fmt == "all" else args.fmt
    urls = list_archives(args.base, fmt, args.license_filter, session,
                         verbose=args.verbose, max_depth=args.max_depth)
    urls = sorted(set(urls))

    if not urls:
        print("没有匹配到任何文件。建议加 -v --dry-run 看看目录里到底有什么，"
              "或检查 --base / --fmt / --license / --max-depth", file=sys.stderr)
        sys.exit(1)

    total_n = len(urls)
    print(f"匹配到 {total_n} 个压缩包：")
    for u in urls:
        print("  -", u.rsplit("/", 1)[-1])
    if args.dry_run:
        return

    results = []
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(download_one, u, args.out, session,
                          not args.no_verify): u for u in urls}
        for i, fut in enumerate(as_completed(futs), 1):
            fname, status = fut.result()
            print(f"[{i}/{total_n}] {fname}: {status}")
            results.append((fname, status))

    failed = [f for f, s in results if s == "FAILED"]
    print(f"\n完成：{total_n - len(failed)}/{total_n} 成功")
    if failed:
        print("失败列表（重跑脚本会自动断点续传）：")
        for f in failed:
            print("  -", f)

    if args.extract and not failed:
        extract(args.out)


if __name__ == "__main__":
    main()
