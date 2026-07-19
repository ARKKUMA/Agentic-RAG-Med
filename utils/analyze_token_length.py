"""
PMC 摘要 Token 长度量化分析
tokenizer : tiktoken cl100k_base（DeepSeek/Qwen3/GLM 系列 BPE 近似值）
用法: python analyze_token_length.py [--input DIR] [--limit N] [--embed-limit 512]
"""

import argparse
import random
import statistics
import json
from pathlib import Path
from xml.etree import ElementTree as ET
import tiktoken

# ── 全文 section 提取（用于正文分块分析）──────────────────
def _all_text(el):
    return "".join(el.itertext()).strip() if el is not None else ""

def get_abstract(root):
    el = root.find(".//abstract")
    return _all_text(el) if el is not None else ""

def get_body_sections(root):
    """返回 [(title, text), ...] 列表"""
    secs = []
    for sec in root.findall(".//body//sec"):
        title_el = sec.find("title")
        title = title_el.text.strip() if title_el is not None and title_el.text else "untitled"
        secs.append((title, _all_text(sec)))
    return secs

# ── 统计辅助 ──────────────────────────────────────────────
def percentile(sorted_data, p):
    if not sorted_data:
        return 0
    idx = (len(sorted_data) - 1) * p / 100
    lo, hi = int(idx), min(int(idx) + 1, len(sorted_data) - 1)
    return sorted_data[lo] + (sorted_data[hi] - sorted_data[lo]) * (idx - lo)

def build_histogram(data, bins):
    """返回 [(label, count), ...]"""
    hist = []
    edges = [0] + bins + [float("inf")]
    labels = (
        [f"<{bins[0]}"] +
        [f"{bins[i]}–{bins[i+1]-1}" for i in range(len(bins)-1)] +
        [f"≥{bins[-1]}"]
    )
    for i, label in enumerate(labels):
        lo, hi = edges[i], edges[i+1]
        cnt = sum(1 for x in data if lo <= x < hi)
        hist.append((label, cnt))
    return hist

# ── 分割策略建议 ──────────────────────────────────────────
def split_strategy(lengths, embed_limit):
    n = len(lengths)
    sorted_l = sorted(lengths)

    pcts = {p: percentile(sorted_l, p) for p in [50, 75, 90, 95, 99, 100]}
    need_split = sum(1 for x in lengths if x > embed_limit)
    need_split_pct = need_split / n * 100

    lines = []
    lines.append(f"\n## 2. 分割策略分析（嵌入模型上限 {embed_limit} tokens）\n")
    lines.append(f"- 超出上限的摘要数：**{need_split} / {n}（{need_split_pct:.1f}%）**")
    lines.append(f"- 50th 分位数：{pcts[50]:.0f} tokens")
    lines.append(f"- 75th 分位数：{pcts[75]:.0f} tokens")
    lines.append(f"- 90th 分位数：{pcts[90]:.0f} tokens")
    lines.append(f"- 95th 分位数：{pcts[95]:.0f} tokens")
    lines.append(f"- 99th 分位数：{pcts[99]:.0f} tokens")
    lines.append(f"- 最大值：{pcts[100]:.0f} tokens\n")

    p95 = pcts[95]
    p99 = pcts[99]

    lines.append("### 策略判断\n")

    if p95 <= embed_limit:
        lines.append(f"> **95th 分位数（{p95:.0f}）≤ {embed_limit}**：")
        lines.append("> 绝大多数摘要无需切割，只有约 5% 的长尾需要处理。")
        lines.append("> **推荐方案 A**：截断策略——超出部分直接截断，损失信息可忽略不计。\n")
        if p99 > embed_limit * 1.5:
            lines.append(f"> **99th 分位数（{p99:.0f}）> {embed_limit*1.5:.0f}**：")
            lines.append("> 存在极端长尾（1%），建议对超出 2× 上限的摘要改用滑动窗口分块，")
            lines.append("> 其余超限摘要截断即可。\n")
    elif p95 <= embed_limit * 1.5:
        lines.append(f"> **95th 分位数（{p95:.0f}）在 {embed_limit}–{embed_limit*1.5:.0f} 区间**：")
        lines.append("> 相当数量的摘要超限，截断会损失尾部信息。")
        lines.append("> **推荐方案 B**：滑动窗口分块，步长 = 上限 × 0.8，overlap = 上限 × 0.2。")
        lines.append(f"> 示例：上限 {embed_limit}，步长 {int(embed_limit*0.8)}，重叠 {int(embed_limit*0.2)} tokens。\n")
    else:
        lines.append(f"> **95th 分位数（{p95:.0f}）> {embed_limit*1.5:.0f}**：")
        lines.append("> 大量摘要严重超限，建议：")
        lines.append("> 1. 换用上限更大的嵌入模型（如 bge-m3 支持 8192 tokens）；")
        lines.append("> 2. 或对摘要先做句子级分块再嵌入，每块 ≤ 上限。\n")

    lines.append("### 三级处理流程\n")
    lines.append("```")
    lines.append(f"输入摘要")
    lines.append(f"  ├─ ≤ {embed_limit} tokens          → 直接嵌入（无需处理）")
    lines.append(f"  ├─ {embed_limit+1}–{embed_limit*2} tokens  → 滑动窗口分为 2 块，各取独立向量后取平均")
    lines.append(f"  └─ > {embed_limit*2} tokens          → 按句子边界分块，每块 ≤ {embed_limit} tokens")
    lines.append("```\n")

    return "\n".join(lines), pcts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",       default=r"d:\Rag-Med\pmc_xml_extracted")
    ap.add_argument("--limit",       type=int, default=8000, help="扫描文件数上限")
    ap.add_argument("--embed-limit", type=int, default=512,  help="嵌入模型 token 上限")
    ap.add_argument("--output",      default=r"d:\Rag-Med\pmc_token_analysis.md")
    args = ap.parse_args()

    enc = tiktoken.get_encoding("cl100k_base")
    print("Tokenizer: tiktoken cl100k_base（GPT-4 / DeepSeek / Qwen 近似）")

    xml_files = list(Path(args.input).rglob("*.xml"))
    print(f"找到 {len(xml_files)} 个 XML，随机取 {args.limit} 个分析…")
    random.seed(42)
    sample = random.sample(xml_files, min(args.limit, len(xml_files)))

    abs_lengths = []       # 摘要 token 数
    body_lengths = []      # 全文 token 数
    sec_lengths  = []      # 单个 section token 数（用于分块参考）
    examples = {"short": [], "mid": [], "long": []}  # 各区间示例

    for i, path in enumerate(sample, 1):
        if i % 1000 == 0:
            print(f"  {i}/{len(sample)}…", flush=True)
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError:
            continue

        abstract = get_abstract(root)
        if not abstract:
            continue

        abs_tok = len(enc.encode(abstract))
        abs_lengths.append(abs_tok)

        # 收集示例
        if abs_tok < 100 and len(examples["short"]) < 3:
            title_el = root.find(".//article-title")
            examples["short"].append({
                "tokens": abs_tok,
                "title": _all_text(title_el)[:80],
                "abstract": abstract[:300],
            })
        elif 100 <= abs_tok <= 512 and len(examples["mid"]) < 3:
            title_el = root.find(".//article-title")
            examples["mid"].append({
                "tokens": abs_tok,
                "title": _all_text(title_el)[:80],
                "abstract": abstract[:300],
            })
        elif abs_tok > 512 and len(examples["long"]) < 3:
            title_el = root.find(".//article-title")
            examples["long"].append({
                "tokens": abs_tok,
                "title": _all_text(title_el)[:80],
                "abstract": abstract[:300],
            })

        # 正文
        secs = get_body_sections(root)
        if secs:
            body_tok = sum(len(enc.encode(t)) for _, t in secs)
            body_lengths.append(body_tok)
            for _, t in secs:
                st = len(enc.encode(t))
                if st > 0:
                    sec_lengths.append(st)

    # ── 统计 ──
    sorted_abs  = sorted(abs_lengths)
    n = len(sorted_abs)
    print(f"\n有效摘要：{n} 篇")

    pcts_abs = {p: percentile(sorted_abs, p) for p in [25, 50, 75, 90, 95, 99, 100]}
    mean_abs = statistics.mean(abs_lengths)
    stdev_abs = statistics.stdev(abs_lengths)

    sorted_body = sorted(body_lengths)
    pcts_body = {p: percentile(sorted_body, p) for p in [50, 75, 90, 95, 99]}

    sorted_sec = sorted(sec_lengths)
    pcts_sec = {p: percentile(sorted_sec, p) for p in [50, 75, 90, 95, 99]}

    EMBED_LIMIT = args.embed_limit
    hist_bins = [50, 100, 150, 200, 250, 300, 400, 512, 700, 1000]
    hist = build_histogram(abs_lengths, hist_bins)

    # ── 报告 ──
    strategy_text, _ = split_strategy(abs_lengths, EMBED_LIMIT)

    bar_max = max(c for _, c in hist)
    def bar(c): return "█" * int(c / bar_max * 30)

    report_lines = [
        "# PMC oa_comm 文本 Token 长度分析报告",
        "",
        f"> Tokenizer：`tiktoken cl100k_base`（DeepSeek V3/V4、Qwen3、GLM-4/5 BPE 近似值）  ",
        f"> 嵌入模型 Token 上限：`{EMBED_LIMIT}`  ",
        f"> 分析样本：{n} 篇（含有效摘要）",
        "",
        "---",
        "",
        "## 1. 摘要 Token 长度统计",
        "",
        "### 1.1 描述性统计",
        "",
        "| 指标 | 值 |",
        "|------|----|",
        f"| 样本量 | {n} 篇 |",
        f"| 均值（mean） | {mean_abs:.1f} tokens |",
        f"| 标准差（std） | {stdev_abs:.1f} tokens |",
        f"| 25th 分位数 | {pcts_abs[25]:.0f} tokens |",
        f"| 中位数（50th） | {pcts_abs[50]:.0f} tokens |",
        f"| 75th 分位数 | {pcts_abs[75]:.0f} tokens |",
        f"| 90th 分位数 | {pcts_abs[90]:.0f} tokens |",
        f"| **95th 分位数** | **{pcts_abs[95]:.0f} tokens** |",
        f"| **99th 分位数** | **{pcts_abs[99]:.0f} tokens** |",
        f"| 最大值 | {pcts_abs[100]:.0f} tokens |",
        "",
        "### 1.2 Token 长度分布直方图",
        "",
        "```",
        f"{'区间':<14} {'篇数':>6}  分布",
    ]
    for label, cnt in hist:
        pct = cnt / n * 100
        report_lines.append(f"{label:<14} {cnt:>6} ({pct:>4.1f}%)  {bar(cnt)}")
    report_lines.append("```")

    over_limit = sum(1 for x in abs_lengths if x > EMBED_LIMIT)
    report_lines += [
        "",
        f"> 超出 {EMBED_LIMIT} tokens 的摘要：**{over_limit} 篇（{over_limit/n*100:.1f}%）**",
        "",
        "### 1.3 各区间典型样本",
        "",
        "**短摘要（< 100 tokens）**",
        "",
    ]
    for ex in examples["short"]:
        report_lines += [
            f"- **[{ex['tokens']} tokens]** {ex['title']}",
            f"  > {ex['abstract'][:200]}…",
            "",
        ]
    report_lines += ["**中等摘要（100–512 tokens，主体区间）**", ""]
    for ex in examples["mid"]:
        report_lines += [
            f"- **[{ex['tokens']} tokens]** {ex['title']}",
            f"  > {ex['abstract'][:200]}…",
            "",
        ]
    report_lines += [f"**长摘要（> {EMBED_LIMIT} tokens，需切割）**", ""]
    for ex in examples["long"]:
        report_lines += [
            f"- **[{ex['tokens']} tokens]** {ex['title']}",
            f"  > {ex['abstract'][:200]}…",
            "",
        ]

    report_lines.append(strategy_text)

    report_lines += [
        "---",
        "",
        "## 3. 正文（body）Token 长度参考",
        "",
        "| 指标 | 值 |",
        "|------|----|",
        f"| 正文中位数 | {pcts_body[50]:.0f} tokens |",
        f"| 正文 90th | {pcts_body[90]:.0f} tokens |",
        f"| 正文 95th | {pcts_body[95]:.0f} tokens |",
        f"| 正文 99th | {pcts_body[99]:.0f} tokens |",
        f"| 单章节中位数 | {pcts_sec[50]:.0f} tokens |",
        f"| 单章节 90th | {pcts_sec[90]:.0f} tokens |",
        f"| 单章节 95th | {pcts_sec[95]:.0f} tokens |",
        "",
        "> 正文若需分块嵌入，建议**按章节（section）为单位**，",
        f"> 单章节 90th 分位数为 {pcts_sec[90]:.0f} tokens，",
        f"> 超出 {EMBED_LIMIT} tokens 的章节再做滑动窗口二次切割。",
        "",
        "---",
        "",
        "## 4. 结论与建议",
        "",
        "| 对象 | 结论 |",
        "|------|------|",
        f"| 摘要嵌入 | 95th={pcts_abs[95]:.0f} tokens，{'大部分无需切割，仅5%长尾需处理' if pcts_abs[95] <= EMBED_LIMIT else '超过上限，需切割'} |",
        f"| 长尾策略 | 99th={pcts_abs[99]:.0f} tokens，{'建议滑动窗口或截断' if pcts_abs[99] > EMBED_LIMIT else '截断即可'} |",
        f"| 正文分块 | 优先按章节切，单节 90th={pcts_sec[90]:.0f} tokens |",
        f"| Tokenizer | cl100k_base（≈DeepSeek/Qwen3/GLM BPE），实际部署时替换为目标模型 tokenizer |",
    ]

    out = Path(args.output)
    out.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\n报告保存至 {out}")

    # 控制台摘要
    print(f"\n摘要统计: mean={mean_abs:.0f}  p50={pcts_abs[50]:.0f}"
          f"  p95={pcts_abs[95]:.0f}  p99={pcts_abs[99]:.0f}  max={pcts_abs[100]:.0f}")
    print(f"超出 {EMBED_LIMIT} tokens: {over_limit}/{n} ({over_limit/n*100:.1f}%)")
    print(f"正文单章节 p90={pcts_sec[90]:.0f}  p95={pcts_sec[95]:.0f}")


if __name__ == "__main__":
    main()
