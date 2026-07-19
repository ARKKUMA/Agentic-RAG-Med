"""
umls_builder.py — 从 UMLS REST API 构建医学同义词词典
用法：python -m retrieval.umls_builder --api-key YOUR_KEY
      python -m retrieval.umls_builder --api-key YOUR_KEY --verify  # 只测试连接

输出：retrieval/vocabulary/umls_synonyms.json
      retrieval/vocabulary/umls_build.log
"""

import os
import json
import time
import logging
import argparse
import requests
from pathlib import Path

# ── 配置 ──────────────────────────────────────────────────────────
BASE_URL   = 'https://uts-ws.nlm.nih.gov/rest'
OUT_DIR    = Path(__file__).parent / 'vocabulary'
OUT_FILE   = OUT_DIR / 'umls_synonyms.json'
LOG_FILE   = OUT_DIR / 'umls_build.log'

# 只保留这几个权威词库的同义词（过滤噪音）
VOCAB_WHITELIST = {
    'MSH',    # MeSH — 医学主题词表
    'SNOMEDCT_US',  # SNOMED CT
    'RXNORM', # 药物标准名
    'NCI',    # NCI Thesaurus
    'ICD10CM',# ICD-10-CM
    'HPO',    # Human Phenotype Ontology
}

# 每个 term 最多取多少个同义词
MAX_SYNONYMS_PER_TERM = 15
# API 请求间隔（避免被限速）
REQUEST_DELAY = 0.3  # 秒
# ─────────────────────────────────────────────────────────────────

# 要从 UMLS 扩充的种子术语（覆盖疾病、药物、生物标志物三类）
SEED_TERMS = [
    # 心血管
    'myocardial infarction', 'heart failure', 'hypertension',
    'atrial fibrillation', 'coronary artery disease', 'stroke',
    'deep vein thrombosis', 'pulmonary embolism', 'atherosclerosis',
    # 代谢
    'type 2 diabetes mellitus', 'type 1 diabetes mellitus',
    'obesity', 'hyperlipidemia', 'non-alcoholic fatty liver disease',
    'metabolic syndrome', 'hypothyroidism', 'hyperthyroidism',
    # 肿瘤
    'lung cancer', 'breast cancer', 'colorectal cancer',
    'hepatocellular carcinoma', 'prostate cancer', 'leukemia',
    'lymphoma', 'melanoma', 'ovarian cancer', 'pancreatic cancer',
    # 神经 / 精神
    "alzheimer's disease", "parkinson's disease", 'multiple sclerosis',
    'epilepsy', 'depression', 'anxiety disorder', 'schizophrenia',
    'bipolar disorder', 'traumatic brain injury', 'dementia',
    # 呼吸
    'chronic obstructive pulmonary disease', 'asthma',
    'acute respiratory distress syndrome', 'pneumonia', 'tuberculosis',
    # 感染
    'sepsis', 'HIV infection', 'COVID-19', 'urinary tract infection',
    'hepatitis B', 'hepatitis C',
    # 肾 / 肝
    'chronic kidney disease', 'acute kidney injury',
    'liver cirrhosis', 'non-alcoholic steatohepatitis',
    # 药物
    'metformin', 'aspirin', 'atorvastatin', 'warfarin', 'insulin',
    'lisinopril', 'amlodipine', 'methotrexate', 'ibuprofen',
    'chemotherapy', 'immunotherapy', 'radiotherapy',
    'pembrolizumab', 'nivolumab', 'trastuzumab', 'rituximab',
    # 生物标志物
    'HbA1c', 'LDL cholesterol', 'HDL cholesterol', 'C-reactive protein',
    'troponin', 'creatinine', 'hemoglobin', 'platelet count',
    'brain natriuretic peptide', 'PSA', 'BRCA1', 'BRCA2',
    # 手术 / 操作
    'coronary artery bypass grafting', 'percutaneous coronary intervention',
    'organ transplantation', 'dialysis', 'mechanical ventilation',
]


class UMLSClient:
    """UMLS REST API 客户端。"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()
        self.session.params = {'apiKey': api_key}

    def verify(self) -> bool:
        """验证 API Key 是否有效。"""
        try:
            r = self.session.get(
                f'{BASE_URL}/search/current',
                params={'string': 'aspirin', 'pageSize': 1},
                timeout=10,
            )
            r.raise_for_status()
            data = r.json()
            return len(data.get('result', {}).get('results', [])) > 0
        except Exception as e:
            logging.error(f'API Key 验证失败: {e}')
            return False

    def search_cui(self, term: str) -> str | None:
        """根据术语名称搜索 CUI（Concept Unique Identifier）。"""
        try:
            r = self.session.get(
                f'{BASE_URL}/search/current',
                params={
                    'string':   term,
                    'searchType': 'exact',
                    'pageSize': 1,
                },
                timeout=10,
            )
            r.raise_for_status()
            results = r.json()['result']['results']
            if results and results[0].get('ui', 'NONE') != 'NONE':
                return results[0]['ui']
            # exact 没找到则用 normalizedString 再试
            r2 = self.session.get(
                f'{BASE_URL}/search/current',
                params={
                    'string':   term,
                    'searchType': 'normalizedString',
                    'pageSize': 1,
                },
                timeout=10,
            )
            r2.raise_for_status()
            results2 = r2.json()['result']['results']
            if results2 and results2[0].get('ui', 'NONE') != 'NONE':
                return results2[0]['ui']
            return None
        except Exception as e:
            logging.warning(f'  搜索 [{term}] 失败: {e}')
            return None

    def get_synonyms(self, cui: str) -> list[str]:
        """根据 CUI 获取所有白名单词库中的同义词名称。"""
        synonyms: list[str] = []
        page = 1
        while True:
            try:
                r = self.session.get(
                    f'{BASE_URL}/content/current/CUI/{cui}/atoms',
                    params={'pageSize': 25, 'pageNumber': page},
                    timeout=10,
                )
                if r.status_code == 404:
                    break
                r.raise_for_status()
                data   = r.json()
                atoms  = data.get('result', [])
                if not atoms:
                    break
                for atom in atoms:
                    vocab = atom.get('rootSource', '')
                    name  = atom.get('name', '').strip()
                    if vocab in VOCAB_WHITELIST and name and name not in synonyms:
                        synonyms.append(name)
                # 分页
                total_pages = data.get('pageCount', 1)
                if page >= total_pages or len(synonyms) >= MAX_SYNONYMS_PER_TERM:
                    break
                page += 1
                time.sleep(REQUEST_DELAY)
            except Exception as e:
                logging.warning(f'  获取 CUI={cui} atoms 失败 (page={page}): {e}')
                break
        return synonyms[:MAX_SYNONYMS_PER_TERM]


def build(api_key: str) -> dict[str, list[str]]:
    """
    遍历 SEED_TERMS，查询 UMLS，构建同义词词典。
    已有缓存的直接跳过（断点续跑）。
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            logging.FileHandler(LOG_FILE, encoding='utf-8'),
            logging.StreamHandler(),
        ],
    )

    client = UMLSClient(api_key)

    # 验证 Key
    logging.info('验证 API Key ...')
    if not client.verify():
        raise ValueError('API Key 无效，请检查后重试')
    logging.info('API Key 验证通过')

    # 加载已有缓存（支持断点续跑）
    result: dict[str, list[str]] = {}
    if OUT_FILE.exists():
        result = json.loads(OUT_FILE.read_text(encoding='utf-8'))
        logging.info(f'加载已有缓存：{len(result)} 条术语')

    total  = len(SEED_TERMS)
    done   = 0
    skip   = 0
    failed = 0

    for i, term in enumerate(SEED_TERMS):
        if term in result:
            skip += 1
            continue

        logging.info(f'[{i+1:>3}/{total}]  {term}')

        # 1. 搜索 CUI
        cui = client.search_cui(term)
        time.sleep(REQUEST_DELAY)

        if cui is None:
            logging.warning(f'  未找到 CUI，跳过')
            result[term] = []
            failed += 1
        else:
            # 2. 获取同义词
            syns = client.get_synonyms(cui)
            # 去掉与种子词本身完全相同的（不区分大小写）
            syns = [s for s in syns if s.lower() != term.lower()]
            result[term] = syns
            done += 1
            logging.info(f'  CUI={cui}  同义词={len(syns)}  示例: {syns[:3]}')
            time.sleep(REQUEST_DELAY)

        # 每 10 条保存一次（防中断丢失）
        if (i + 1) % 10 == 0:
            OUT_FILE.write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding='utf-8',
            )

    # 最终保存
    OUT_FILE.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )

    logging.info('─' * 50)
    logging.info(f'完成：新增 {done}，跳过缓存 {skip}，未找到 {failed}')
    logging.info(f'词典已保存至 {OUT_FILE}')
    return result


def main():
    parser = argparse.ArgumentParser(description='从 UMLS 构建医学同义词词典')
    parser.add_argument('--api-key', required=True, help='UMLS API Key')
    parser.add_argument('--verify',  action='store_true',
                        help='只验证 API Key，不构建词典')
    args = parser.parse_args()

    if args.verify:
        client = UMLSClient(args.api_key)
        ok = client.verify()
        print('API Key 有效 ✓' if ok else 'API Key 无效 ✗')
        return

    build(args.api_key)


if __name__ == '__main__':
    main()
