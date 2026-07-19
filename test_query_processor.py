"""
test_query_processor.py — 查询处理器本地测试
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from retrieval import MedicalQueryProcessor

LOG_PATH  = "d:/Rag-Med/logs/query_processor.jsonl"
processor = MedicalQueryProcessor(log_path=LOG_PATH)

TEST_QUERIES = [
    # 缩写 + 时间过滤
    "What is the effect of metformin on T2DM patients in the last 5 years?",
    # 中文查询
    "二甲双胍对心血管疾病有何影响？",
    # 多实体 + IMRaD 倾向
    "mechanism of aspirin in preventing MI and stroke",
    # 复杂过滤：时间 + 研究设计
    "COVID-19 vaccine efficacy trial results since 2021",
    # 纯缩写
    "HbA1c LDL HDL in T2DM patients after statin therapy",
    # 边界：无实体
    "how does exercise affect mental health?",
]

SEP = "=" * 65

for q in TEST_QUERIES:
    result = processor.process(q)
    print(SEP)
    print(result.summary())

print(SEP)
print(f"\n共处理 {len(TEST_QUERIES)} 条查询，全部完成。")
