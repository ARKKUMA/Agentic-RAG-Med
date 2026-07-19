"""
PMC XML 数据质量分析脚本
用法: python analyze_pmc_xml.py [--input DIR] [--sample N] [--output report.json]
"""

import argparse
import json
import sys
import random
from pathlib import Path
from xml.etree import ElementTree as ET
from collections import Counter, defaultdict

NS = {
    "xlink": "http://www.w3.org/1999/xlink",
    "mml":   "http://www.w3.org/1998/Math/MathML",
}

# JATS XML 字段提取 ──────────────────────────────────────────────

def _text(root, xpath):
    el = root.find(xpath)
    return el.text.strip() if el is not None and el.text else None

def _all_text(el):
    """递归拼接元素下的全部文本（含子标签）"""
    return "".join(el.itertext()).strip() if el is not None else None

def extract_fields(tree):
    root = tree.getroot()
    front = root.find(".//front")
    if front is None:
        return None

    fields = {}

    # ── IDs ──
    for id_el in root.findall(".//article-id"):
        id_type = id_el.get("pub-id-type", "")
        if id_type == "pmid":
            fields["pmid"] = id_el.text.strip() if id_el.text else None
        elif id_type == "pmc":
            fields["pmc_id"] = id_el.text.strip() if id_el.text else None
        elif id_type == "doi":
            fields["doi"] = id_el.text.strip() if id_el.text else None

    # ── Title ──
    title_el = root.find(".//article-title")
    fields["title"] = _all_text(title_el)

    # ── Journal ──
    jt = root.find(".//journal-title")
    fields["journal"] = jt.text.strip() if jt is not None and jt.text else None

    # ── Pub date（优先 epub，其次 ppub，其次 collection）──
    pub_date = None
    for ptype in ("epub", "ppub", "collection"):
        pd = root.find(f".//pub-date[@pub-type='{ptype}']")
        if pd is not None:
            year  = _text(pd, "year")
            month = _text(pd, "month") or "1"
            day   = _text(pd, "day")   or "1"
            if year:
                pub_date = f"{year}-{int(month):02d}-{int(day):02d}"
                break
    fields["pub_date"] = pub_date

    # ── Abstract ──
    abs_el = root.find(".//abstract")
    fields["abstract"] = _all_text(abs_el)

    # ── Body ──
    body_el = root.find(".//body")
    body_text = _all_text(body_el)
    fields["body_len"] = len(body_text) if body_text else 0

    # ── Keywords ──
    kwds = [_all_text(k) for k in root.findall(".//kwd")]
    fields["keywords"] = [k for k in kwds if k] or None

    # ── Article type ──
    fields["article_type"] = root.get("article-type")

    return fields


# 分析逻辑 ────────────────────────────────────────────────────────

def analyze(xml_files, sample_n):
    if sample_n and len(xml_files) > sample_n:
        random.seed(42)
        xml_files = random.sample(xml_files, sample_n)

    total = len(xml_files)
    field_names = ["pmid", "pmc_id", "doi", "title", "journal", "pub_date",
                   "abstract", "keywords", "article_type"]

    missing   = defaultdict(int)   # 字段缺失计数
    short_abs = 0                  # 极短 abstract（< 50 字符）
    parse_err = 0                  # XML 解析失败
    no_body   = 0                  # body 为空
    journals  = Counter()
    pub_years = Counter()
    article_types = Counter()

    for path in xml_files:
        try:
            tree = ET.parse(path)
        except ET.ParseError:
            parse_err += 1
            continue

        fields = extract_fields(tree)
        if fields is None:
            for f in field_names:
                missing[f] += 1
            continue

        for f in field_names:
            val = fields.get(f)
            if not val:
                missing[f] += 1

        if fields.get("abstract"):
            if len(fields["abstract"]) < 50:
                short_abs += 1
        if fields.get("body_len", 0) == 0:
            no_body += 1
        if fields.get("journal"):
            journals[fields["journal"]] += 1
        if fields.get("pub_date"):
            pub_years[fields["pub_date"][:4]] += 1
        if fields.get("article_type"):
            article_types[fields["article_type"]] += 1

    # ── 缺失率 ──
    missing_rate = {f: round(missing[f] / total * 100, 2) for f in field_names}

    return {
        "total_analyzed": total,
        "parse_errors":   parse_err,
        "missing_rate_%": missing_rate,
        "short_abstract_count": short_abs,
        "short_abstract_rate_%": round(short_abs / total * 100, 2),
        "no_body_count": no_body,
        "no_body_rate_%": round(no_body / total * 100, 2),
        "top_journals":  journals.most_common(20),
        "pub_year_dist": dict(sorted(pub_years.items())),
        "article_types": dict(article_types.most_common()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  default=r"d:\Rag-Med\pmc_xml_extracted",
                    help="解压后的根目录")
    ap.add_argument("--sample", type=int, default=5000,
                    help="随机抽样数量（0=全量，但很慢）")
    ap.add_argument("--output", default=r"d:\Rag-Med\pmc_quality_report.json",
                    help="报告输出路径")
    args = ap.parse_args()

    input_dir = Path(args.input)
    print(f"扫描 {input_dir} …", flush=True)

    xml_files = list(input_dir.rglob("*.xml"))
    print(f"找到 {len(xml_files)} 个 XML 文件", flush=True)

    if not xml_files:
        print("未找到任何 XML 文件，请确认解压目录正确")
        sys.exit(1)

    sample_n = args.sample if args.sample > 0 else None
    if sample_n:
        print(f"随机抽样 {min(sample_n, len(xml_files))} 个分析 …", flush=True)

    report = analyze(xml_files, sample_n)

    out = Path(args.output)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告已保存到 {out}")

    # ── 打印摘要 ──
    print("\n===== 数据质量摘要 =====")
    print(f"分析样本:  {report['total_analyzed']} 篇")
    print(f"解析失败:  {report['parse_errors']} 篇")
    print(f"\n字段缺失率:")
    for f, r in report["missing_rate_%"].items():
        flag = " ⚠" if r > 1 else ""
        print(f"  {f:<15} {r:>6.2f}%{flag}")
    print(f"\n极短摘要(<50字):  {report['short_abstract_rate_%']:.2f}%")
    print(f"无正文(body=0):   {report['no_body_rate_%']:.2f}%")
    print(f"\nTop 10 期刊:")
    for j, c in report["top_journals"][:10]:
        print(f"  {c:>6}  {j}")
    print(f"\n发表年份分布:")
    for y, c in sorted(report["pub_year_dist"].items())[-10:]:
        print(f"  {y}: {c}")
    print(f"\n文章类型:")
    for t, c in report["article_types"].items():
        print(f"  {c:>6}  {t}")


if __name__ == "__main__":
    main()
