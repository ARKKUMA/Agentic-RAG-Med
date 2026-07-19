"""
批量解压 pmc_xml 目录下的 tar.gz 文件
用法: python extract_pmc.py [--output OUTPUT_DIR] [--workers N] [--skip-existing]
      python extract_pmc.py --files-from oa_comm_filelist_clean.txt  # 只解压清单中的文件
"""

import argparse
import tarfile
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# ── 默认配置 ──────────────────────────────────────────────────
DEFAULT_INPUT_DIR = Path(r"d:\Rag-Med\pmc_xml")
DEFAULT_OUTPUT_DIR = Path(r"d:\Rag-Med\pmc_xml_extracted")
DEFAULT_WORKERS = 4
# ──────────────────────────────────────────────────────────────

print_lock = Lock()


def log(msg: str):
    with print_lock:
        print(msg, flush=True)


def extract_one(archive: Path, output_dir: Path, skip_existing: bool) -> tuple[str, bool, str]:
    """解压单个 tar.gz，返回 (文件名, 是否成功, 错误信息)"""
    # 以 archive stem 作为子目录（去掉 .tar.gz）
    stem = archive.name.removesuffix(".tar.gz")
    dest = output_dir / stem

    if skip_existing and dest.exists() and any(dest.iterdir()):
        return archive.name, True, "skipped (already exists)"

    try:
        dest.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(dest, filter="data")
        return archive.name, True, "ok"
    except Exception as e:
        return archive.name, False, str(e)


def main():
    parser = argparse.ArgumentParser(description="批量解压 pmc_xml 目录下的 tar.gz 文件")
    parser.add_argument("--input", default=str(DEFAULT_INPUT_DIR), help="tar.gz 所在目录")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR), help="解压输出目录")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="并行线程数 (默认 4)")
    parser.add_argument("--skip-existing", action="store_true", help="跳过已解压的文件")
    parser.add_argument("--files-from", metavar="FILE",
                        help="只解压此文件中列出的文件名（每行一个文件名，由 dry-run 生成）")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if args.files_from:
        allowed = {
            line.strip()
            for line in Path(args.files_from).read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        }
        archives = sorted(
            p for p in input_dir.glob("*.tar.gz") if p.name in allowed
        )
        missing = allowed - {p.name for p in archives}
        if missing:
            print(f"警告: 清单中有 {len(missing)} 个文件在 {input_dir} 中不存在:")
            for m in sorted(missing):
                print(f"  {m}")
    else:
        archives = sorted(input_dir.glob("*.tar.gz"))

    if not archives:
        print(f"在 {input_dir} 中未找到任何 .tar.gz 文件")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    total = len(archives)
    print(f"找到 {total} 个 tar.gz 文件")
    print(f"输出目录: {output_dir}")
    print(f"并行线程: {args.workers}")
    print(f"跳过已有: {args.skip_existing}")
    print("-" * 60)

    done = 0
    failed = []
    skipped = 0
    start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(extract_one, arc, output_dir, args.skip_existing): arc
            for arc in archives
        }
        for future in as_completed(futures):
            name, ok, msg = future.result()
            done += 1
            elapsed = time.time() - start
            speed = done / elapsed if elapsed > 0 else 0
            eta = (total - done) / speed if speed > 0 else 0

            if msg.startswith("skipped"):
                skipped += 1
                status = "SKIP"
            elif ok:
                status = "OK  "
            else:
                failed.append((name, msg))
                status = "FAIL"

            log(f"[{done:>4}/{total}] {status} | {name[:55]:<55} | ETA {eta:>5.0f}s")

    print("-" * 60)
    total_time = time.time() - start
    print(f"完成: {done} 个  |  跳过: {skipped} 个  |  失败: {len(failed)} 个  |  耗时: {total_time:.1f}s")

    if failed:
        print("\n失败列表:")
        for name, err in failed:
            print(f"  {name}: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
