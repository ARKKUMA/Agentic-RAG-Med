# PMC oa_comm 文本 Token 长度分析报告

> Tokenizer：`tiktoken cl100k_base`（DeepSeek V3/V4、Qwen3、GLM-4/5 BPE 近似值）  
> 嵌入模型 Token 上限：`512`  
> 分析样本：7248 篇（含有效摘要）

---

## 1. 摘要 Token 长度统计

### 1.1 描述性统计

| 指标 | 值 |
|------|----|
| 样本量 | 7248 篇 |
| 均值（mean） | 332.5 tokens |
| 标准差（std） | 134.6 tokens |
| 25th 分位数 | 244 tokens |
| 中位数（50th） | 323 tokens |
| 75th 分位数 | 409 tokens |
| 90th 分位数 | 496 tokens |
| **95th 分位数** | **560 tokens** |
| **99th 分位数** | **704 tokens** |
| 最大值 | 1912 tokens |

### 1.2 Token 长度分布直方图

```
区间                 篇数  分布
<50                82 ( 1.1%)  █
50–99             131 ( 1.8%)  █
100–149           241 ( 3.3%)  ███
150–199           555 ( 7.7%)  ███████
200–249           931 (12.8%)  ████████████
250–299          1116 (15.4%)  ███████████████
300–399          2204 (30.4%)  ██████████████████████████████
400–511          1370 (18.9%)  ██████████████████
512–699           540 ( 7.5%)  ███████
700–999            71 ( 1.0%)  
≥1000               7 ( 0.1%)  
```

> 超出 512 tokens 的摘要：**609 篇（8.4%）**

### 1.3 各区间典型样本

**短摘要（< 100 tokens）**

- **[97 tokens]** Optimizing Treatment Precision: Role of Adaptive Radiotherapy in Modern Anal Can
  > Simple SummaryAnal cancer that is not metastatic is treated with definitive chemoradiation. The introduction of intensity-modulated radiation therapy significantly reduced treatment toxicity; adaptive…

- **[20 tokens]** Making Human Neurons from Stem Cells after Spinal Cord Injury
  > A new study by Yan and colleagues makes an important contribution to research on human spinal cord stem cells.…

- **[71 tokens]** Bis(4-acetyl­anilinium) hexa­chlorido­stannate(IV)
  > In the title compound, (C8H10NO)2[SnCl6], the SnIV atom exists in an octa­hedral coordination environment. In the crystal, inter­molecular N—H⋯O and N—H⋯Cl hydrogen bonds link the cations and anions i…

**中等摘要（100–512 tokens，主体区间）**

- **[171 tokens]** Predicting Suitable Habitat for Glipa (Coleoptera: Mordellidae: Mordellinae) Und
  > Simple SummaryBased on 297 geographic distribution records and seven bioclimatic variables, this study predicted the potential distribution of Glipa under current and future climate change scenarios b…

- **[385 tokens]** Ankle–Brachial Index Predicts Long-Term Renal Outcomes in Acute Stroke Patients
  > Renal dysfunction is common after stroke. We aimed to investigate the clinical predictability of the ankle–brachial index (ABI) and brachial-ankle pulse wave velocity (baPWV) on poststroke renal deter…

- **[330 tokens]** In vitro wound healing potential of cyclohexane extract of Onosma dichroantha Bo
  > Onosma dichroantha Boiss. is a biennial herb used in traditional medicine in Iran for healing wounds and burns. Our previous study demonstrated that cyclohexane extract of O. dichroantha Boiss. enhanc…

**长摘要（> 512 tokens，需切割）**

- **[517 tokens]** Case report: Salivary duct carcinoma in a patient with a germline CDH1 pathogeni
  > IntroductionRecently, an entity known as salivary duct carcinoma with rhabdoid features (SDC-RF) has been associated with somatic CDH1 mutations. Here we present the first known case report of convent…

- **[574 tokens]** Clinical Presentation, Diagnostic Delays, and Treatment Outcomes in Postural Ort
  > Background:  Postural orthostatic tachycardia syndrome (POTS) is a heterogeneous disorder of autonomic regulation characterised by unexplained orthostatic tachycardia in the absence of postural hypote…

- **[846 tokens]** Impact of a trace mineral injection at weaning on growth, behavior, and inflamma
  > AbstractTwo experiments evaluated the effects of an injectable trace mineral (ITM) solution at weaning on trace mineral (TM) status, inflammatory and antioxidant responses, grazing behavior, response …


## 2. 分割策略分析（嵌入模型上限 512 tokens）

- 超出上限的摘要数：**609 / 7248（8.4%）**
- 50th 分位数：323 tokens
- 75th 分位数：409 tokens
- 90th 分位数：496 tokens
- 95th 分位数：560 tokens
- 99th 分位数：704 tokens
- 最大值：1912 tokens

### 策略判断

> **95th 分位数（560）在 512–768 区间**：
> 相当数量的摘要超限，截断会损失尾部信息。
> **推荐方案 B**：滑动窗口分块，步长 = 上限 × 0.8，overlap = 上限 × 0.2。
> 示例：上限 512，步长 409，重叠 102 tokens。

### 三级处理流程

```
输入摘要
  ├─ ≤ 512 tokens          → 直接嵌入（无需处理）
  ├─ 513–1024 tokens  → 滑动窗口分为 2 块，各取独立向量后取平均
  └─ > 1024 tokens          → 按句子边界分块，每块 ≤ 512 tokens
```

---

## 3. 正文（body）Token 长度参考

| 指标 | 值 |
|------|----|
| 正文中位数 | 12170 tokens |
| 正文 90th | 25257 tokens |
| 正文 95th | 31279 tokens |
| 正文 99th | 52182 tokens |
| 单章节中位数 | 416 tokens |
| 单章节 90th | 2003 tokens |
| 单章节 95th | 2997 tokens |

> 正文若需分块嵌入，建议**按章节（section）为单位**，
> 单章节 90th 分位数为 2003 tokens，
> 超出 512 tokens 的章节再做滑动窗口二次切割。

---

## 4. 结论与建议

| 对象 | 结论 |
|------|------|
| 摘要嵌入 | 95th=560 tokens，超过上限，需切割 |
| 长尾策略 | 99th=704 tokens，建议滑动窗口或截断 |
| 正文分块 | 优先按章节切，单节 90th=2003 tokens |
| Tokenizer | cl100k_base（≈DeepSeek/Qwen3/GLM BPE），实际部署时替换为目标模型 tokenizer |