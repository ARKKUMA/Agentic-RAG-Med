"""
medical_vocab.py — 医学词汇库
包含：缩写展开、同义词词典、实体识别正则模式、过滤条件关键词
数据来源参考：UMLS、MeSH、DrugBank、ICD-10
"""

# ══════════════════════════════════════════════════════════════════
# 1. 医学缩写 → 全称（含中文缩写映射）
# ══════════════════════════════════════════════════════════════════
ABBREVIATIONS: dict[str, list[str]] = {
    # 心血管
    "mi":    ["myocardial infarction", "heart attack"],
    "chd":   ["coronary heart disease", "coronary artery disease"],
    "cad":   ["coronary artery disease"],
    "hf":    ["heart failure", "cardiac failure"],
    "af":    ["atrial fibrillation"],
    "dvt":   ["deep vein thrombosis"],
    "pe":    ["pulmonary embolism"],
    "htn":   ["hypertension", "high blood pressure"],
    "bp":    ["blood pressure"],
    "lvef":  ["left ventricular ejection fraction"],
    "mace":  ["major adverse cardiovascular events"],
    # 代谢
    "t2dm":  ["type 2 diabetes mellitus", "type 2 diabetes"],
    "t1dm":  ["type 1 diabetes mellitus", "type 1 diabetes"],
    "dm":    ["diabetes mellitus", "diabetes"],
    "hba1c": ["glycated hemoglobin", "hemoglobin a1c"],
    "bmi":   ["body mass index"],
    "nafld": ["non-alcoholic fatty liver disease"],
    "nash":  ["non-alcoholic steatohepatitis"],
    # 肿瘤
    "nsclc": ["non-small cell lung cancer"],
    "sclc":  ["small cell lung cancer"],
    "crc":   ["colorectal cancer"],
    "hcc":   ["hepatocellular carcinoma", "liver cancer"],
    "rcc":   ["renal cell carcinoma", "kidney cancer"],
    "tnbc":  ["triple-negative breast cancer"],
    "pdl1":  ["programmed death-ligand 1"],
    "pd1":   ["programmed cell death protein 1"],
    "car-t": ["chimeric antigen receptor t-cell therapy"],
    # 神经
    "ad":    ["alzheimer's disease", "alzheimer disease"],
    "pd":    ["parkinson's disease", "parkinson disease"],
    "ms":    ["multiple sclerosis"],
    "tbi":   ["traumatic brain injury"],
    "ptsd":  ["post-traumatic stress disorder"],
    # 呼吸
    "copd":  ["chronic obstructive pulmonary disease"],
    "ards":  ["acute respiratory distress syndrome"],
    "osa":   ["obstructive sleep apnea"],
    # 感染
    "hiv":   ["human immunodeficiency virus"],
    "tb":    ["tuberculosis"],
    "uti":   ["urinary tract infection"],
    "sepsis":["septicemia", "bloodstream infection"],
    # 实验室 / 生物标志物
    "ldl":   ["low-density lipoprotein", "ldl cholesterol"],
    "hdl":   ["high-density lipoprotein", "hdl cholesterol"],
    "crp":   ["c-reactive protein"],
    "bnp":   ["brain natriuretic peptide", "b-type natriuretic peptide"],
    "egfr":  ["estimated glomerular filtration rate",
              "epidermal growth factor receptor"],
    "alt":   ["alanine aminotransferase"],
    "ast":   ["aspartate aminotransferase"],
    "ck":    ["creatine kinase"],
    "tnf":   ["tumor necrosis factor"],
    "il6":   ["interleukin-6"],
    "il-6":  ["interleukin-6"],
    # 研究设计
    "rct":   ["randomized controlled trial"],
    "rr":    ["relative risk", "risk ratio"],
    "or":    ["odds ratio"],
    "hr":    ["hazard ratio"],
    "ci":    ["confidence interval"],
    "nnt":   ["number needed to treat"],
    # 中文缩写
    "二甲双胍": ["metformin"],
    "心梗":    ["myocardial infarction", "heart attack"],
    "高血压":  ["hypertension"],
    "糖尿病":  ["diabetes mellitus"],
}

# ══════════════════════════════════════════════════════════════════
# 2. 同义词词典（疾病 / 药物 / 手术）
# ══════════════════════════════════════════════════════════════════
SYNONYMS: dict[str, list[str]] = {
    # 疾病同义词
    "heart attack":              ["myocardial infarction", "acute MI", "cardiac infarction"],
    "myocardial infarction":     ["heart attack", "acute MI"],
    "stroke":                    ["cerebrovascular accident", "CVA", "brain attack", "ischemic stroke"],
    "high blood pressure":       ["hypertension", "arterial hypertension"],
    "hypertension":              ["high blood pressure", "arterial hypertension"],
    "diabetes":                  ["diabetes mellitus", "DM", "hyperglycemia"],
    "type 2 diabetes":           ["T2DM", "non-insulin-dependent diabetes", "adult-onset diabetes"],
    "cancer":                    ["malignancy", "neoplasm", "carcinoma", "tumor", "tumour"],
    "breast cancer":             ["breast carcinoma", "mammary carcinoma"],
    "lung cancer":               ["pulmonary carcinoma", "bronchogenic carcinoma"],
    "alzheimer":                 ["alzheimer's disease", "AD", "dementia", "cognitive decline"],
    "alzheimer's disease":       ["AD", "dementia", "senile dementia"],
    "parkinson":                 ["parkinson's disease", "PD", "parkinsonism"],
    "depression":                ["major depressive disorder", "MDD", "clinical depression"],
    "anxiety":                   ["anxiety disorder", "generalized anxiety disorder", "GAD"],
    "obesity":                   ["overweight", "adiposity", "excess body weight"],
    "heart failure":             ["cardiac failure", "congestive heart failure", "CHF"],
    "atrial fibrillation":       ["AF", "AFib", "irregular heartbeat"],
    "covid-19":                  ["SARS-CoV-2", "coronavirus", "COVID", "novel coronavirus"],
    "sepsis":                    ["blood poisoning", "septicemia", "systemic infection"],
    "kidney disease":            ["renal disease", "nephropathy", "chronic kidney disease", "CKD"],
    "liver disease":             ["hepatic disease", "hepatopathy", "liver disorder"],
    # 药物同义词
    "metformin":                 ["glucophage", "biguanide", "dimethylbiguanide"],
    "aspirin":                   ["acetylsalicylic acid", "ASA", "salicylate"],
    "atorvastatin":              ["lipitor", "statin"],
    "rosuvastatin":              ["crestor", "statin"],
    "statin":                    ["HMG-CoA reductase inhibitor", "lipid-lowering agent"],
    "warfarin":                  ["coumadin", "anticoagulant", "vitamin k antagonist"],
    "insulin":                   ["insulin therapy", "exogenous insulin"],
    "methotrexate":              ["MTX", "antimetabolite"],
    "ibuprofen":                 ["NSAID", "non-steroidal anti-inflammatory"],
    "lisinopril":                ["ACE inhibitor", "angiotensin-converting enzyme inhibitor"],
    "amlodipine":                ["calcium channel blocker", "CCB"],
    "chemotherapy":              ["cytotoxic therapy", "antineoplastic therapy", "chemo"],
    "immunotherapy":             ["immune checkpoint inhibitor", "biologic therapy", "checkpoint blockade"],
    "radiotherapy":              ["radiation therapy", "irradiation", "RT"],
    # 操作 / 手术
    "surgery":                   ["operation", "surgical procedure", "intervention"],
    "bypass surgery":            ["CABG", "coronary artery bypass grafting"],
    "angioplasty":               ["PCI", "percutaneous coronary intervention", "stenting"],
    "biopsy":                    ["tissue sampling", "histological sampling"],
    # 结局指标
    "mortality":                 ["death", "survival", "fatality"],
    "morbidity":                 ["disease burden", "complications"],
    "efficacy":                  ["effectiveness", "therapeutic effect", "clinical benefit"],
    "side effect":               ["adverse event", "adverse drug reaction", "toxicity"],
}

# ══════════════════════════════════════════════════════════════════
# 3. 实体识别正则模式
# ══════════════════════════════════════════════════════════════════
import re

ENTITY_PATTERNS: dict[str, str] = {
    "drug": (
        r'\b(aspirin|metformin|atorvastatin|rosuvastatin|warfarin|insulin|'
        r'lisinopril|amlodipine|methotrexate|ibuprofen|paracetamol|acetaminophen|'
        r'omeprazole|pantoprazole|clopidogrel|rivaroxaban|apixaban|dabigatran|'
        r'ramipril|losartan|valsartan|simvastatin|pravastatin|fluoxetine|'
        r'sertraline|amoxicillin|azithromycin|doxycycline|ciprofloxacin|'
        r'prednisolone|dexamethasone|hydrocortisone|levothyroxine|'
        r'chemotherapy|immunotherapy|radiotherapy|pembrolizumab|nivolumab|'
        r'trastuzumab|bevacizumab|rituximab|imatinib|osimertinib)\b'
    ),
    "disease": (
        r'\b(diabetes|hypertension|cancer|carcinoma|lymphoma|leukemia|'
        r'melanoma|stroke|infarction|heart.?failure|alzheimer|parkinson|'
        r'depression|anxiety|obesity|asthma|copd|fibrosis|cirrhosis|'
        r'hepatitis|sepsis|pneumonia|tuberculosis|HIV|AIDS|COVID|'
        r'arthritis|osteoporosis|atherosclerosis|thrombosis|embolism|'
        r'nephropathy|neuropathy|retinopathy|dementia|epilepsy|'
        r'schizophrenia|bipolar|psoriasis|eczema|colitis|crohn)\b'
    ),
    "procedure": (
        r'\b(surgery|biopsy|transplant|dialysis|chemotherapy|radiotherapy|'
        r'angioplasty|bypass|catheterization|endoscopy|colonoscopy|'
        r'MRI|CT.?scan|ultrasound|echocardiography|mammography|'
        r'vaccination|immunization|screening|PCI|CABG|stenting)\b'
    ),
    "biomarker": (
        r'\b(HbA1c|LDL|HDL|CRP|BNP|troponin|creatinine|albumin|'
        r'hemoglobin|hematocrit|platelet|neutrophil|lymphocyte|'
        r'TNF|IL-?6|IL-?1|VEGF|PSA|CA-?125|CEA|AFP|'
        r'BRCA[12]|EGFR|ALK|ROS1|KRAS|BRAF|PD-?L?1)\b'
    ),
    "anatomy": (
        r'\b(heart|liver|kidney|lung|brain|colon|breast|prostate|'
        r'pancreas|thyroid|adrenal|spleen|stomach|intestine|'
        r'artery|vein|neuron|hepatic|renal|cardiac|pulmonary|'
        r'cerebral|gastrointestinal|musculoskeletal)\b'
    ),
}

# 编译为 re 对象（不区分大小写）
COMPILED_PATTERNS: dict[str, re.Pattern] = {
    entity_type: re.compile(pattern, re.IGNORECASE)
    for entity_type, pattern in ENTITY_PATTERNS.items()
}

# ══════════════════════════════════════════════════════════════════
# 4. 时间范围关键词
# ══════════════════════════════════════════════════════════════════
import datetime

_CURRENT_YEAR = datetime.datetime.now().year

TIME_PATTERNS: list[tuple] = [
    # (正则, 提取年份范围的函数)
    (re.compile(r'last\s+(\d+)\s+years?', re.I),
     lambda m: (_CURRENT_YEAR - int(m.group(1)), _CURRENT_YEAR)),
    (re.compile(r'past\s+(\d+)\s+years?', re.I),
     lambda m: (_CURRENT_YEAR - int(m.group(1)), _CURRENT_YEAR)),
    (re.compile(r'since\s+(20\d{2})', re.I),
     lambda m: (int(m.group(1)), _CURRENT_YEAR)),
    (re.compile(r'after\s+(20\d{2})', re.I),
     lambda m: (int(m.group(1)) + 1, _CURRENT_YEAR)),
    (re.compile(r'before\s+(20\d{2})', re.I),
     lambda m: (1900, int(m.group(1)) - 1)),
    (re.compile(r'between\s+(20\d{2})\s+and\s+(20\d{2})', re.I),
     lambda m: (int(m.group(1)), int(m.group(2)))),
    (re.compile(r'in\s+(20\d{2})', re.I),
     lambda m: (int(m.group(1)), int(m.group(1)))),
    (re.compile(r'recent(?:ly)?', re.I),
     lambda m: (_CURRENT_YEAR - 5, _CURRENT_YEAR)),
]

# ══════════════════════════════════════════════════════════════════
# 5. 中文医学词汇 → 英文翻译（用于中文查询预处理）
# ══════════════════════════════════════════════════════════════════
ZH_TO_EN: dict[str, str] = {
    "二甲双胍": "metformin",
    "阿司匹林": "aspirin",
    "他汀":     "statin",
    "胰岛素":   "insulin",
    "华法林":   "warfarin",
    "心血管":   "cardiovascular",
    "心脏":     "heart",
    "心肌梗死": "myocardial infarction",
    "心梗":     "myocardial infarction",
    "心力衰竭": "heart failure",
    "心衰":     "heart failure",
    "高血压":   "hypertension",
    "血压":     "blood pressure",
    "糖尿病":   "diabetes mellitus",
    "2型糖尿病":"type 2 diabetes",
    "血糖":     "blood glucose",
    "糖化血红蛋白": "HbA1c",
    "癌症":     "cancer",
    "肺癌":     "lung cancer",
    "乳腺癌":   "breast cancer",
    "肝癌":     "hepatocellular carcinoma",
    "结直肠癌": "colorectal cancer",
    "脑卒中":   "stroke",
    "卒中":     "stroke",
    "中风":     "stroke",
    "阿尔茨海默": "alzheimer",
    "帕金森":   "parkinson",
    "抑郁":     "depression",
    "焦虑":     "anxiety",
    "肥胖":     "obesity",
    "血脂":     "lipid",
    "胆固醇":   "cholesterol",
    "低密度脂蛋白": "LDL",
    "高密度脂蛋白": "HDL",
    "甘油三酯": "triglyceride",
    "肾病":     "kidney disease",
    "肝病":     "liver disease",
    "炎症":     "inflammation",
    "免疫":     "immune",
    "化疗":     "chemotherapy",
    "放疗":     "radiotherapy",
    "手术":     "surgery",
    "临床试验": "clinical trial",
    "随机对照": "randomized controlled",
    "系统综述": "systematic review",
    "荟萃分析": "meta-analysis",
    "发病率":   "incidence",
    "死亡率":   "mortality",
    "不良反应": "adverse event",
    "副作用":   "side effect",
    "疗效":     "efficacy",
    "预后":     "prognosis",
    "风险":     "risk",
    "影响":     "effect",
    "治疗":     "treatment",
    "预防":     "prevention",
    "诊断":     "diagnosis",
    "机制":     "mechanism",
}

# ══════════════════════════════════════════════════════════════════
# 6. 研究设计关键词 → imrad_type / chunk_type 过滤提示
# ══════════════════════════════════════════════════════════════════
STUDY_DESIGN_HINTS: dict[str, str] = {
    # 关键词 → 偏好的 imrad_type
    "mechanism":    "methods",
    "protocol":     "methods",
    "how to":       "methods",
    "procedure":    "methods",
    "technique":    "methods",
    "result":       "results",
    "outcome":      "results",
    "finding":      "results",
    "efficacy":     "results",
    "effect":       "results",
    "association":  "results",
    "background":   "introduction",
    "review":       "introduction",
    "discuss":      "discussion",
    "implication":  "discussion",
    "conclusion":   "discussion",
}

# ══════════════════════════════════════════════════════════════════
# 7. 运行时合并 UMLS 词典（若文件存在则自动加载）
# ══════════════════════════════════════════════════════════════════
import json as _json
from pathlib import Path as _Path

_UMLS_FILE = _Path(__file__).parent / 'vocabulary' / 'umls_synonyms.json'

def _load_umls_synonyms() -> dict[str, list[str]]:
    """加载 UMLS 构建的同义词文件，合并进 SYNONYMS。"""
    if not _UMLS_FILE.exists():
        return {}
    try:
        data = _json.loads(_UMLS_FILE.read_text(encoding='utf-8'))
        # 过滤掉空列表
        return {k: v for k, v in data.items() if v}
    except Exception:
        return {}

def _merge_umls(base: dict, umls: dict) -> dict:
    """将 UMLS 同义词合并进静态词典，已有条目追加（不覆盖）。"""
    merged = {k: list(v) for k, v in base.items()}
    for term, syns in umls.items():
        term_lower = term.lower()
        if term_lower in merged:
            # 追加 UMLS 中尚未收录的同义词
            existing_lower = {s.lower() for s in merged[term_lower]}
            merged[term_lower].extend(
                s for s in syns if s.lower() not in existing_lower
            )
        else:
            merged[term_lower] = syns
    return merged

# 启动时执行一次合并
_umls_data = _load_umls_synonyms()
if _umls_data:
    SYNONYMS = _merge_umls(SYNONYMS, _umls_data)
    _umls_count = sum(len(v) for v in _umls_data.values())
    print(f'[medical_vocab] UMLS 词典已加载：{len(_umls_data)} 个术语，'
          f'{_umls_count} 条同义词')
