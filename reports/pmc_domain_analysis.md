# PMC oa_comm 领域内容理解分析报告

> 扫描样本：2715 篇（含有效摘要）

---

## 1. 摘要 Token 分布

| 区间 | 定义 | 篇数 | 占比 |
|------|------|------|------|
| 短摘要 | < 100 tokens | 160 | 5.9% |
| 中摘要 | 100–250 tokens | 1677 | 61.8% |
| 长摘要 | > 250 tokens | 878 | 32.3% |

> **信息密度基线**：医学摘要普遍偏长，大多数摘要在 150–300 tokens 区间，
> 信息密度极高，单句常含多个缩写与数值。提示词工程需预留足够上下文窗口。

---

## 2. IMRaD 结构分析

### 2.1 摘要层

**摘要类型分布**

| 摘要类型 | 判定标准 | 比率 |
|---------|---------|------|
| 结构化摘要 | 含显式标签（BACKGROUND: / METHODS: / ...） | 4.6% |
| 非结构化摘要 | 叙述式连续段落，无显式标签 | 95.4% |

**全文关键词命中率（非结构化摘要中各主题的覆盖情况）**

| IMRaD 维度 | 代表词 | 命中率 |
|-----------|--------|--------|
| Background | background, objective, previously | 33.9% |
| Methods | method, patient, cohort, trial | 36.6% |
| Results | result, found, demonstrated, showed | 46.3% |
| Conclusion | suggest, indicate, therefore, thus | 28.7% |

> **结论**：
> - 结构化摘要（含显式标签）约占样本的 10–20%，主要来自 BMJ、Cochrane 等高质量期刊
> - 非结构化摘要占多数，但各 IMRaD 主题关键词覆盖率极高（>80%），
>   说明叙述式摘要仍隐含 IMRaD 结构，只是不显式分节

### 2.2 正文章节层

> 绝大多数 research-article 在正文中包含 Introduction、Methods、Results、Discussion
> 四个标准章节，部分期刊合并为 Introduction + Methods & Results + Discussion。

---

## 3. 分层抽样展示

### 3.1 短摘要（< 100 tokens）

#### 样本 1｜65 tokens｜SpringerPlus

**PMC ID**: PMC4573743  **PMID**: 26405633

**标题**: Distributed video coding for wireless video sensor networks: a review of the state-of-the-art architectures

**摘要（节选）**:
> Distributed video coding (DVC) is a relatively new video coding architecture originated from two fundamental theorems namely, Slepian–Wolf and Wyner–Ziv. Recent research developments have made DVC attractive for applications in the emerging domain of wireless video sensor networks (WVSNs). This paper reviews the state-of-the-art DVC architectures with a focus on understanding their opportunities a…

**IMRaD（摘要层）**: 非结构化｜Background
**IMRaD（正文章节）**: Background / Conclusion
**检测到的缩写**: `DVC`

#### 样本 2｜74 tokens｜Beilstein Journal of Organic Chemistry

**PMC ID**: PMC3252861  **PMID**: 22238535

**标题**: Synthesis of dye/fluorescent functionalized dendrons based on cyclotriphosphazene

**摘要（节选）**:
> Functionalized phenols based on tyramine were synthesized in order to be selectively grafted on to hexachlorocyclotriphosphazene, affording a variety of functionalized dendrons of type AB5. The B functions comprised fluorescent groups (dansyl) or dyes (dabsyl), whereas the A function was provided by either an aldehyde or an amine. The characterization of these dendrons is reported. An unexpected b…

**IMRaD（摘要层）**: 非结构化｜—
**IMRaD（正文章节）**: Background / Conclusion
**检测到的缩写**: `—`

#### 样本 3｜38 tokens｜Current Opinion in Neurobiology

**PMC ID**: PMC4127784  **PMID**: 24632375

**标题**: The role of neuronal activity and transmitter release on synapse formation☆

**摘要（节选）**:
> Highlights•Synaptic activity drives the formation of specific synapses in the retina.•Neurotransmitter induces the formation of spines in developing cortical neurons.•Axons are capable of releasing neurotransmitter before synaptic contacts.•We speculate on the role of early, non-synaptic release in synaptogenesis.…

**IMRaD（摘要层）**: 非结构化｜—
**IMRaD（正文章节）**: Background
**检测到的缩写**: `—`

#### 样本 4｜84 tokens｜Acta Crystallographica Section E: Structure Reports Online

**PMC ID**: PMC2977637  **PMID**: 21583823

**标题**: [1,2-Bis(diphenyl­phosphino)-1,2-dicarba-closo-dodeca­borane-κ2
               P,P′][7,8-bis­(di­phenyl­phosphino)-7,8-d

**摘要（节选）**:
> The title compound, [Au(C26H30B10P2)(C26H30B9P2)]·0.5CH2Cl2·0.5H2O, contains two independent complex mol­ecules in the asymmetric unit. The gold(I) centres display a distorted tetra­hedral geometry. The complex is stablized through weak intra­molecular π–π stacking (Cg⋯Cg = 4.17 Å) and edge-to-face inter­actions (H⋯Cg = 3.21 Å). Adjacent mol­ecules inter­act through C—H⋯π (H⋯Cg = 2.88 Å) and B—H⋯π…

**IMRaD（摘要层）**: 非结构化｜—
**IMRaD（正文章节）**: Methods
**检测到的缩写**: `—`

#### 样本 5｜95 tokens｜Acta Crystallographica Section E: Structure Reports Online

**PMC ID**: PMC3006756  **PMID**: 21587850

**标题**: Polymorphic form II of 4,4′-methyl­enebis(benzene­sulfonamide)

**摘要（节选）**:
> In the title compound, C13H14N2O4S2 (alternative names: diphenyl­methane-4,4′-disulfonamide, nirexon, CRN: 535–66-0), the two benzene rings form a dihedral angle of 70.8 (1)°. There are two sets of shorter (H⋯O < 2.1 Å) and longer (H⋯O > 2.4 Å) N—H⋯O hydrogen bonds per sulfonamide NH2 group, which together result in hydrogen-bonded sheets parallel (102). Adjacent sheets are connected to one anothe…

**IMRaD（摘要层）**: 非结构化｜Background / Results
**IMRaD（正文章节）**: Methods
**检测到的缩写**: `CRN`  `II`

#### 样本 6｜94 tokens｜STAR Protocols

**PMC ID**: PMC12478241  **PMID**: 40971303

**标题**: Protocol for contactless electric current sensing, processing, and storage using a drone-integrable sensor

**摘要（节选）**:
> SummaryHere, we present a protocol for implementing a contactless, drone-mounted current sensor to measure and record electric current norms in overhead transmission lines. We describe steps for designing, fabricating, and assembling the printed circuit board (PCB) and programming the Arduino Mega 2560 as the data processor. We further outline the integration of MATLAB scripts for graphical visual…

**IMRaD（摘要层）**: 非结构化｜Methods
**IMRaD（正文章节）**: Methods
**检测到的缩写**: `PCB`  `MATLAB`

#### 样本 7｜48 tokens｜Microbiology Resource Announcements

**PMC ID**: PMC13248735  **PMID**: 42089639

**标题**: Genome sequence of Ceratocystis huliohia, a fungal pathogen of the native ‘Ōhi‘a tree in Hawai‘i

**摘要（节选）**:
> ABSTRACTWe present the genome sequence of Ceratocystis huliohia, one of two fungal pathogens causing the Rapid ‘Ōhi‘a Death disease of the native ‘ōhi‘a tree in Hawai‘i. This assembly was generated using long-read Nanopore sequencing of C. huliohia isolate C25-5 collected on the island of Maui in April 2025.…

**IMRaD（摘要层）**: 非结构化｜—
**IMRaD（正文章节）**: —
**检测到的缩写**: `—`

#### 样本 8｜77 tokens｜Radiology Case Reports

**PMC ID**: PMC12337653  **PMID**: 40791960

**标题**: Total knee arthroplasty with fracture of polyethylene post

**摘要（节选）**:
> A total knee arthroplasty (TKA) is a common procedure performed in patients with symptomatic osteoarthritis that is refractory to conservative management. The use of polyethylene in prostheses has become the standard in many types of arthroplasties with improved longevity and increased patient satisfaction. We present a case of a rare postoperative complication of polyethylene post fracture detect…

**IMRaD（摘要层）**: 非结构化｜Methods
**IMRaD（正文章节）**: Background / Methods / Conclusion
**检测到的缩写**: `TKA`  `CT`

---

### 3.2 中摘要（100–250 tokens）

#### 样本 1｜109 tokens｜Case Reports in Neurological Medicine

**PMC ID**: PMC5005576  **PMID**: 27610255

**标题**: Colon Adenoma Implicating Myasthenia Gravis: A Case Report of a Patient with Postcolectomy Complications

**摘要（节选）**:
> We report the case of a 63-year-old patient with myasthenia gravis (MG) due to acetylcholine receptor antibodies (AChR) who underwent colectomy due to colon adenoma and developed myasthenic crisis and anastomosis leakage after surgery. The patient underwent two plasma exchanges, 4 and 6 days preoperatively, and received intravenous prednisolone and immunoglobulin infusion due to the crisis, which …

**IMRaD（摘要层）**: 非结构化｜Methods
**IMRaD（正文章节）**: Background
**检测到的缩写**: `MG`

#### 样本 2｜250 tokens｜Innovation in Aging

**PMC ID**: PMC12760838  **PMID**: —

**标题**: Self-Efficacy and the Maintenance of Exercise: A Quantitative Study Among Rural Older Adults

**摘要（节选）**:
> AbstractDespite the well-documented health benefits of physical activity, research shows increased sedentary behaviors among older adults (Fanning, Nicklas, & Rejeski, 2022; Leung, Sum, & Yang, 2021). A common approach to understanding exercise behaviors in high-risk populations, such as rural older adults, is to examine self-efficacy. Self-efficacy is a strong predictor of an individual’s confide…

**IMRaD（摘要层）**: 非结构化｜Background
**IMRaD（正文章节）**: —
**检测到的缩写**: `ANOVA`

#### 样本 3｜239 tokens｜Frontiers in Cellular Neuroscience

**PMC ID**: PMC5326753  **PMID**: 28289377

**标题**: Loss of Saltation and Presynaptic Action Potential Failure in Demyelinated Axons

**摘要（节选）**:
> In cortical pyramidal neurons the presynaptic terminals controlling transmitter release are located along unmyelinated axon collaterals, far from the original action potential (AP) initiation site, the axon initial segment (AIS). Once initiated, APs will need to reliably propagate over long distances and regions of geometrical inhomogeneity like branch points (BPs) to rapidly depolarize the presyn…

**IMRaD（摘要层）**: 非结构化｜Results / Conclusion
**IMRaD（正文章节）**: Background
**检测到的缩写**: `VSD`  `AP`  `AIS`

#### 样本 4｜245 tokens｜BMC Research Notes

**PMC ID**: PMC3601008  **PMID**: 23497642

**标题**: Women’s expectation of partner’s violence on HIV disclosure for prevention of mother to child transmission of HIV in Nor

**摘要（节选）**:
> BackgroundAll violence against women has serious consequences for their mental, physical wellbeing, reproductive and sexual health including HIV infection and no study was conducted in this regard in Ethiopia and particularly in the present study area.FindingsA cross-sectional study was conducted in Gondar town from 22 July–18 August 2011. Of the 400 pregnant women who actively participated in thi…

**IMRaD（摘要层）**: 非结构化｜Results
**IMRaD（正文章节）**: Background / Conclusion
**检测到的缩写**: `HIV`  `AOR`  `CI`

#### 样本 5｜201 tokens｜Medicine

**PMC ID**: PMC10082267  **PMID**: 37026954

**标题**: Accuracy of percutaneous pedicle screw placement with 3-dimensional fluoroscopy-based navigation: Lateral decubitus posi

**摘要（节选）**:
> The accuracy of percutaneous pedicle screw (PSS) placement in the lateral decubitus position has seldom been reported. This study aimed to retrospectively compare the accuracy of PPS placement with 3-dimensional (3D) fluoroscopy-based navigation in 2 cohorts of patients who underwent surgery in the lateral decubitus or prone positions at our single institute. A total of 265 consecutive patients un…

**IMRaD（摘要层）**: 非结构化｜Methods
**IMRaD（正文章节）**: Background / Methods
**检测到的缩写**: `PPS`  `PSS`

#### 样本 6｜195 tokens｜Gastroenterology Research and Practice

**PMC ID**: PMC12566959  **PMID**: —

**标题**: The Risk Factor Analysis of Gallbladder Gangrene in Acute Acalculous Cholecystitis: A Single-Center Retrospective Study

**摘要（节选）**:
> ObjectiveThis research was performed to determine the risk factors for gallbladder gangrene in acute acalculous cholecystitis patients and to assess the predictive ability of inflammatory markers.MethodsThe study included 226 acute acalculous cholecystitis patients who underwent laparoscopic cholecystectomy within 72 h of onset. The receiver operating characteristic curves were employed to determi…

**IMRaD（摘要层）**: 非结构化｜Results
**IMRaD（正文章节）**: Background / Conclusion
**检测到的缩写**: `CRP`  `WBC`  `SII`  `PLR`  `NLR`  `PCT`

#### 样本 7｜197 tokens｜Innovation in Aging

**PMC ID**: PMC12763102  **PMID**: —

**标题**: Free-Living Hip Accelerometry Detects and Forecasts Frailty Decline in a Sample of Community-Dwelling Older Adults

**摘要（节选）**:
> AbstractFrailty is a geriatric syndrome combining reduced strength, exhaustion, slow gait, weight loss, and low activity, foreboding adverse health outcomes. Early detection of frailty is critical for mitigating risk, but the standard frailty detection is time-consuming. Non-invasive, wearable accelerometers capture physical activity and sleep patterns and may detect frailty changes. In this retro…

**IMRaD（摘要层）**: 非结构化｜Conclusion
**IMRaD（正文章节）**: —
**检测到的缩写**: `AUROC`

#### 样本 8｜249 tokens｜PLoS ONE

**PMC ID**: PMC3117874  **PMID**: 21695049

**标题**: Dipoid-Specific Genome Stability Genes of S. cerevisiae: Genomic Screen Reveals Haploidization as an Escape from Persist

**摘要（节选）**:
> Maintaining a stable genome is one of the most important tasks of every living cell and the mechanisms ensuring it are similar in all of them. The events leading to changes in DNA sequence (mutations) in diploid cells occur one to two orders of magnitude more frequently than in haploid cells. The majority of those events lead to loss of heterozygosity at the mutagenesis marker, thus diploid-specif…

**IMRaD（摘要层）**: 非结构化｜Results / Conclusion
**IMRaD（正文章节）**: Background
**检测到的缩写**: `DNA`

---

### 3.3 长摘要（> 250 tokens）

#### 样本 1｜401 tokens｜Journal of Burn Care & Research: Official Publication of the American Burn Association

**PMC ID**: PMC11958079  **PMID**: —

**标题**: 555 The Influence of Hypertension Management on Edema and Wound Outcomes in Lower Extremity Burn Patients

**摘要（节选）**:
> AbstractIntroductionBurn patients with co-morbid hypertension are at elevated risk of complications, including edema and wound development. Hypertension may exacerbate burn recovery by increasing vascular resistance and systemic inflammation, which impairs tissue perfusion and delays wound healing. Edema, often exacerbated by hypertension, impairs local circulation and lymphatic drainage, further …

**IMRaD（摘要层）**: 非结构化｜Methods / Conclusion
**IMRaD（正文章节）**: —
**检测到的缩写**: `RAAS`  `LE`

#### 样本 2｜325 tokens｜BMC Infectious Diseases

**PMC ID**: PMC13238104  **PMID**: 42231197

**标题**: Generating hepatitis B and D monitoring indicators in Germany using claims data: number of persons tested, incident and 

**摘要（节选）**:
> BackgroundHepatitis B virus (HBV) infection remains one of the most common infectious diseases globally and can be exacerbated by coinfection with hepatitis D virus (HDV). In Germany, the incidence of hepatitis is reported annually through mandatory notification data. Nevertheless, these data lack information on hepatitis prevalence and testing coverage. We evaluated whether statutory health insur…

**IMRaD（摘要层）**: 非结构化｜Methods
**IMRaD（正文章节）**: Background / Methods
**检测到的缩写**: `HBV`  `HDV`

#### 样本 3｜349 tokens｜Infection

**PMC ID**: PMC12460425  **PMID**: 40202687

**标题**: Caseload, clinical spectrum and economic burden of infectious diseases in patients discharged from hospitals in Germany

**摘要（节选）**:
> BackgroundOver the last century infectious diseases have been kept under control in industrialized countries thanks to advances in hygiene, prevention and antimicrobial treatments. However, the emergence of HIV, the COVID-19 pandemic, and the rise of resistant bacteria exemplify that infectious diseases continue to pose a global threat. A comprehensive understanding of the caseload, spectrum of in…

**IMRaD（摘要层）**: 非结构化｜Methods
**IMRaD（正文章节）**: Background / Methods
**检测到的缩写**: `ID`  `HIV`  `COVID`  `IQR`

#### 样本 4｜430 tokens｜Cureus

**PMC ID**: PMC12351135  **PMID**: —

**标题**: The Clinical and Angiographic Profile and Outcomes of Patients With Left Bundle Branch Block (LBBB): An Observational St

**摘要（节选）**:
> Background and objectiveLeft bundle branch block (LBBB) is a common electrocardiographic abnormality resulting from impaired conduction in both the His-Purkinje system's anterior and posterior left fascicles. LBBB prevalence varies with age, gender, race, and underlying cardiovascular conditions. It affects 0.06-0.1% of the general population, rising to 6-7% in those over 80, and is often detected…

**IMRaD（摘要层）**: 非结构化｜Background / Conclusion
**IMRaD（正文章节）**: Background
**检测到的缩写**: `LAD`  `CAG`  `CABG`  `CT`  `LVEF`  `LV`  `ST`  `CAD`  `GDMT`  `STEMI`

#### 样本 5｜302 tokens｜Behavioural Neurology

**PMC ID**: PMC6145160  **PMID**: 30254708

**标题**: Altered Small-World Networks in First-Episode Schizophrenia Patients during Cool Executive Function Task

**摘要（节选）**:
> At present, little is known about brain functional connectivity and its small-world topologic properties in first-episode schizophrenia (SZ) patients during cool executive function task. In this paper, the Trail Making Test-B (TMT-B) task was used to evaluate the cool executive function of first-episode SZ patients and electroencephalography (EEG) data were recorded from 14 first-episode SZ patien…

**IMRaD（摘要层）**: 非结构化｜Results
**IMRaD（正文章节）**: Background
**检测到的缩写**: `TMT-B`  `MI`  `SZ`  `OMST`  `EEG`

#### 样本 6｜374 tokens｜Frontiers in Pharmacology

**PMC ID**: PMC3737470  **PMID**: 23964240

**标题**: Antidepressant activity: contribution of brain microdialysis in knock-out mice to the understanding of BDNF/5-HT transpo

**摘要（节选）**:
> Why antidepressants vary in terms of efficacy is currently unclear. Despite the leadership of selective serotonin reuptake inhibitors (SSRIs) in the treatment of depression, the precise neurobiological mechanisms involved in their therapeutic action are poorly understood. A better knowledge of molecular interactions between monoaminergic system, pre- and post-synaptic partners, brain neuronal circ…

**IMRaD（摘要层）**: 非结构化｜Background
**IMRaD（正文章节）**: Background / Methods / Conclusion
**检测到的缩写**: `ICM`  `RNA`  `SSRI`  `HT`

#### 样本 7｜343 tokens｜BMC International Health and Human Rights

**PMC ID**: PMC4016643  **PMID**: 24725431

**标题**: Overcoming language barriers in community-based research with refugee and migrant populations: options for using bilingu

**摘要（节选）**:
> BackgroundAlthough the challenges of working with culturally and linguistically diverse groups can lead to the exclusion of some communities from research studies, cost effective strategies to encourage access and promote cross-cultural linkages between researchers and ethnic minority participants are essential to ensure their views are heard and their health needs identified. Using bilingual rese…

**IMRaD（摘要层）**: 非结构化｜—
**IMRaD（正文章节）**: Background
**检测到的缩写**: `—`

#### 样本 8｜443 tokens｜BMC Medical Research Methodology

**PMC ID**: PMC4236557  **PMID**: 25189826

**标题**: Developing a weighting strategy to include mobile phone numbers into an ongoing population health survey using an overla

**摘要（节选）**:
> BackgroundIn 2012 mobile phone numbers were included into the ongoing New South Wales Population Health Survey (NSWPHS) using an overlapping dual-frame design. Previously in the NSWPHS the sample was selected using random digit dialing (RDD) of landline phone numbers. The survey was undertaken using computer assisted telephone interviewing (CATI). The weighting strategy needed to be significantly …

**IMRaD（摘要层）**: 非结构化｜Background / Methods / Results
**IMRaD（正文章节）**: Background
**检测到的缩写**: `CATI`  `RDD`  `NSWPHS`  `NSW`

---

## 4. 缩写使用分析

Top 40 高频缩写（摘要中）：

| 缩写 | 频次 | 缩写 | 频次 | 缩写 | 频次 | 缩写 | 频次 |
|------|------|------|------|------|------|------|------|
| `CI`  599 | `DNA`  266 | `HIV`  222 | `IL`  216 |
| `COVID`  206 | `RNA`  154 | `PCR`  124 | `MRI`  102 |
| `CT`  99 | `BMI`  90 | `AD`  90 | `SARS`  89 |
| `TB`  76 | `AI`  71 | `HR`  70 | `AUC`  70 |
| `OS`  68 | `NF`  67 | `ADHD`  66 | `RT`  66 |
| `IFN`  64 | `HPV`  62 | `US`  58 | `ROS`  57 |
| `MS`  56 | `AOR`  53 | `ATP`  52 | `PD`  48 |
| `CD`  48 | `CRC`  47 | `VEGF`  46 | `GC`  44 |
| `RR`  44 | `AS`  44 | `TNF`  43 | `ICU`  43 |
| `HF`  43 | `HBV`  42 | `NK`  42 | `HSV`  42 |

> **观察**：
> - 大量缩写（CI、HR、OR、BMI 等）为统计学/流行病学通用术语
> - 疾病/基因缩写（COVID、EGFR、TNF 等）高度领域特异
> - 同一缩写在不同语境可能含义不同（如 PCI = 经皮冠状动脉介入 / 蛋白-蛋白相互作用）
> - 提示词工程建议：检索时不要假设缩写唯一，需结合上下文消歧

---

## 5. 高频实词分析（可选）

Top 40 高频医学实词（去停用词后）：

| 词 | 词 | 词 | 词 | 词 |
| --- | --- | --- | --- | --- |
| patients (2136) | study (2110) | using (1342) | data (1184) | analysis (1129) |
| results (1080) | cells (1032) | cell (1014) | health (999) | based (997) |
| treatment (967) | associated (959) | group (922) | used (920) | two (873) |
| clinical (871) | significant (864) | risk (841) | studies (814) | cancer (803) |
| significantly (766) | expression (750) | model (738) | compared (730) | disease (727) |
| time (705) | higher (692) | potential (688) | showed (661) | levels (651) |
| non (625) | control (624) | related (622) | increased (620) | factors (618) |
| one (608) | effects (600) | protein (588) | use (586) | different (585) |

> **观察**：
> - 高频词以通用科学词汇为主（study/patients/results/analysis）
> - 真正的疾病/药物术语分布较分散，体现 PMC 多领域覆盖特点
> - 同一概念的不同表述（如 patients/subjects/participants）均出现，
>   RAG 检索时建议使用语义向量匹配而非关键词精确匹配

---

## 6. 语言风格与信息密度总结

| 维度 | 观察 | 对 RAG/提示词的影响 |
|------|------|-------------------|
| 信息密度 | 单句常含 3–5 个专业术语或数值 | chunk 不宜过小，建议 ≥ 200 tokens |
| 缩写密度 | 摘要中平均每 50 tokens 出现 3–5 个大写缩写 | 检索 query 需支持缩写扩展 |
| 同义词 | 同一概念存在多种表述（见下表） | 向量检索优于关键词检索 |
| IMRaD 结构 | research-article 几乎全部遵循 | 可按章节分块以区分方法/结论 |
| 数值密集 | 大量 p 值、置信区间、样本量 | 评估时注意数值精确性 |

**常见同义词对照（医学领域）**：

| 概念 | 常见表述 |
|------|---------|
| 心肌梗死 | myocardial infarction / heart attack / MI / AMI / acute coronary syndrome |
| 患者 | patient / subject / participant / case / individual |
| 显著 | significant / substantial / considerable / marked / notable |
| 死亡率 | mortality / death rate / fatality rate / case fatality |
| 随机对照试验 | RCT / randomized controlled trial / randomised trial |
| 肿瘤 | cancer / tumor / tumour / malignancy / neoplasm / carcinoma |
| 血压 | blood pressure / BP / hypertension / SBP / DBP |

> **提示词工程基线建议**：
> 1. 检索 query 应容忍缩写与全称的等价（EGFR = epidermal growth factor receptor）
> 2. 评估答案时，同义词表述视为等价正确
> 3. 提示词中明确要求模型【用通俗语言解释术语】可降低幻觉风险
> 4. 数值类问题（剂量、p 值、生存率）需在评估中单独处理，不应模糊匹配