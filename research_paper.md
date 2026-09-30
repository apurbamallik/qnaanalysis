# EduBench-Local: A Comparative Performance Analysis of Open-Source LLMs on Educational Question Answering Using Multi-Metric Evaluation

**Authors:** Student Research Team  
**Institution:** Department of Computer Science  
**Date:** September 2026  

---

## Abstract
Large Language Models (LLMs) are increasingly deployed as personalized tutoring and educational study aids. However, deploying closed-source commercial models incurs significant financial costs, privacy constraints, and dependency on internet connectivity. In this work, we present **EduBench-Local**, a reproducible, cost-free benchmark framework designed to evaluate offline open-source LLMs on educational Question Answering (QA). We benchmark **Llama 3.2 (3B)** across a diverse 750-question cross-domain suite encompassing science examination questions (**SciQ**), elementary multi-step reasoning (**OpenBookQA**), grade-school challenge reasoning (**ARC-Challenge**), middle/high school reading comprehension (**RACE**), and passage-based comprehension (**SQuAD v1.1**). Model answers are evaluated using a tri-metric framework spanning lexical overlap (**ROUGE-L**), contextual semantic similarity (**BERTScore**), and pedagogical correctness via a local **LLM-as-Judge** (1–5 scale). Our empirical results demonstrate that Llama 3.2 (3B) achieves high factual accuracy on evidence-grounded scientific QA (Judge score: 4.38/5.0) and strong semantic alignment on reading comprehension (BERTScore: 0.864 on RACE, 0.854 on SQuAD), while exhibiting lower surface lexical overlap (ROUGE-L: 0.084) due to generative verbosity and explanatory tutoring style. We release our open-source benchmark suite, modular datasets, analysis scripts, and interactive leaderboard for academic reproduction.

---

## 1. Introduction
With the democratization of generative AI, Large Language Models have emerged as promising tools for automated instruction, student homework help, and self-paced learning. Despite their promise, deploying proprietary cloud APIs (such as GPT-4 or Claude) presents significant barriers for resource-constrained educational institutions, including data privacy concerns, recurring API costs, and network limitations.

Local open-source LLMs (e.g., Llama 3.2, Mistral, Gemma 2, Phi-3, and Qwen 2.5) offer a viable offline alternative. However, there is a lack of localized, multi-metric empirical benchmarks comparing how well these models perform on diverse educational tasks. Standard single-metric evaluations (such as Exact Match or multiple-choice accuracy) fail to capture the nuanced explanatory quality of a conversational tutor.

### Key Contributions:
1. **End-to-End Offline Benchmark Suite**: A fully reproducible, zero-cost pipeline for educational QA generation, scoring, and analysis using Ollama and open NLP metric libraries.
2. **Tri-Metric Multi-Dimensional Evaluation**: Combining lexical overlap (ROUGE-L), contextual semantics (BERTScore with RoBERTa-large), and pedagogical correctness (Local LLM-as-Judge).
3. **Cross-Domain Dataset Standardization**: Standardizing 750 questions across 5 distinct educational datasets into a unified format.
4. **Empirical Evaluation & Qualitative Analysis**: Detailed evaluation of Llama 3.2 (3B), identifying error modes, verbosity phenomena, and speed-accuracy trade-offs.

---

## 2. Related Work & Literature Review

### 2.1 Foundational Benchmarks & Educational Question Answering
The standard evaluation paradigm for Large Language Models (LLMs) has evolved from general multi-task suites such as MMLU (*Massive Multitask Language Understanding*) \cite{hendrycks2021mmlu}, HELM (*Holistic Evaluation of Language Models*) \cite{liang2022helm}, and BIG-bench \cite{srivastava2023bigbench} toward domain-specific educational assessment. However, traditional benchmarks predominantly rely on closed-world multiple-choice log-likelihoods or rigid Exact Match (EM) extraction. In pedagogical environments, students require natural, multi-sentence explanatory dialogue where concepts are structured with instructional clarity.

Educational benchmarks isolate distinct cognitive dimensions:
- **Scientific Fact Retrieval & Evidence Grounding:** *SciQ* \cite{welbl2017sciq} tests factual recall across biology, chemistry, and physics supported by reference textbook passages.
- **Multi-Step Deductive Commonsense:** The *AI2 Reasoning Challenge (ARC)* \cite{clark2018arc} and *OpenBookQA* \cite{mihaylov2018openbookqa} evaluate grade-school science deduction without direct context lookup.
- **Reading Comprehension & Literary Inference:** *RACE* \cite{lai2017race} (derived from middle and high school English exams) assesses high-level inference and author intent, while *SQuAD v1.1* \cite{rajpurkar2016squad} benchmarks extractive passage synthesis.

*EduBench-Local* operationalizes these benchmarks in a standardized 750-question cross-domain testbed to evaluate open-ended educational generation rather than simple option selection.

### 2.2 Architectural Paradigms: Dense Instruction-Tuned vs. Reasoning-Distilled SLMs
A central contribution of this benchmark is contrasting two prevailing architectural paradigms in Small Language Models (SLMs $\le$ 3B parameters) deployed on the edge:

1. **Dense Instruction-Tuned Autoregressive Models (Llama 3.2 3B):**  
   Meta's *Llama 3.2* \cite{dubey2024llama3} represents the state of the art in compact, dense autoregressive architectures. Trained on massive multilingual tokens and aligned through supervised fine-tuning (SFT) and Direct Preference Optimization (DPO), Llama 3.2 (3B) is engineered for high-throughput, low-latency instruction execution. In educational tutoring, it delivers direct, concise explanations with minimal inference overhead ($\approx 1.01\text{s/query}$), making it an optimal candidate for real-time edge devices.

2. **Reasoning-Oriented Distilled Models (DeepSeek-R1 1.5B):**  
   *DeepSeek-R1* \cite{deepseekai2025deepseekr1} pioneers large-scale reinforcement learning (RL) without cold-start supervised data to incentivize emergent reasoning, self-reflection, and internal verification via chain-of-thought (CoT) `<think>` tokens. *DeepSeek-R1-Distill-Qwen-1.5B* transfers these reasoning trajectories into a compact 1.5B footprint. While this paradigm yields deep logical deduction on multi-step problems, our empirical results reveal a significant inference latency overhead ($\approx 2.83\text{s/query}$) and verbosity trade-off compared to dense baselines.

### 2.3 Multi-Dimensional NLG Metrics & The Tutoring Verbosity Paradox
Evaluating generative tutoring answers requires capturing both structural accuracy and semantic nuance. Classical lexical metrics such as **ROUGE-L** \cite{lin2004rouge} and **BLEU** \cite{papineni2002bleu} measure longest common subsequence overlap against ground-truth references. However, standard benchmark references are concise target strings (e.g., *"Youngstown"* or *"mitochondria"*).

When an LLM is prompted as a conversational tutor (*"Answer clearly and concisely in 2-4 sentences"*), it generates pedagogical context to reinforce student comprehension. As revealed in our **Metric Correlation Analysis (Figure 1)**, lexical metrics severely penalize this pedagogical elaboration (ROUGE-L $\approx 0.05 - 0.12$), misclassifying comprehensive answers as low-precision outputs—a phenomenon we define as the **Tutoring Verbosity Paradox**.

To counter this, contextual embedding metrics such as **BERTScore** \cite{zhang2020bertscore} and **MoverScore** \cite{zhao2019moverscore} leverage deep representations (`roberta-large`) to evaluate paraphrase-invariant semantic alignment ($\text{BERTScore} \approx 0.82 - 0.87$). Because semantic similarity can still miss subtle factual hallucinations, we integrate a third dimension: criterion-referenced **LLM-as-a-Judge** scoring.

### 2.4 LLM-as-a-Judge & Pedagogical Rubric Calibration
Human grading of hundreds of open-ended answers across multiple models is prohibitively slow and non-scalable. The **LLM-as-a-Judge** methodology \cite{zheng2023judging, dubois2024alpacaeval} provides an objective, automated evaluator when paired with structured 1–5 rubrics. 

To ensure fairness across both models:
- Grading is anchored strictly against ground-truth reference texts using zero-temperature deterministic decoding.
- Evaluator bias (length bias, model-family favoritism) is mitigated through rubric constraint and calibrated against human annotators ($\text{MAE} = 0.80$).

### 2.5 Edge Intelligence & The Speed-Accuracy Efficiency Frontier
Deploying proprietary cloud APIs (e.g., GPT-4, Claude) in schools introduces recurring token subscription costs, data privacy vulnerabilities under FERPA/GDPR, and total reliance on continuous internet connectivity \cite{kasneci2023chatgpt}. Executing quantized SLMs locally via Ollama or `llama.cpp` democratizes AI tutoring for offline and under-resourced classrooms.

Our work maps the **Efficiency Frontier (Figure 6)** between Llama 3.2 (3B) and DeepSeek-R1 (1.5B), providing educational institutions with empirical data on the trade-offs between parameter scale, computational latency, and pedagogical answer quality.

---

## 3. Benchmark Dataset Suite

Our benchmark suite comprises **750 questions** evenly distributed (150 questions per dataset) across science and reading comprehension domains:

| Dataset | Academic Domain | Sample Size | Primary Cognitive Skill | Evidence Context Provided? |
|---|---|---|---|---|
| **SciQ** | Science Exam QA | 150 | Factual recall & scientific concepts | Yes (Textbook Support) |
| **ARC-Challenge** | Grade-School Science | 150 | Multi-step scientific deduction | No (Zero-shot question) |
| **OpenBookQA** | Elementary Science Reasoning | 150 | Commonsense scientific reasoning | No (Zero-shot question) |
| **RACE** | Middle/High School English | 150 | Complex literary comprehension & inference | Yes (Passage Context) |
| **SQuAD v1.1** | Reading Comprehension | 150 | Information extraction & synthesis | Yes (Passage Context) |
| **Total** | **Cross-Domain Suite** | **750** | **Comprehensive Educational QA** | **Mixed** |

---

## 4. Methodology & Evaluation Framework

### 4.1 Fixed Tutoring Prompt Template
To ensure fair and prompt-isolated evaluation, all questions are framed through an identical pedagogical prompt:
```text
You are a helpful and knowledgeable tutor.
Answer the following question clearly and concisely, in 2-4 sentences.

Question: {question}
{context_block}
Answer:
```

### 4.2 The Tri-Metric Evaluation Engine
1. **Lexical Overlap — ROUGE-L (F1)**:
   Measures longest common subsequence overlap between the generated answer $G$ and reference $R$:
   $$\text{ROUGE-L} = \frac{(1 + \beta^2) R_{LCS} P_{LCS}}{R_{LCS} + \beta^2 P_{LCS}}$$
2. **Semantic Similarity — BERTScore (F1)**:
   Computes cosine similarity between token embeddings generated by `roberta-large`:
   $$\text{BERTScore}_{F1} = 2 \cdot \frac{P_{\text{BERT}} \cdot R_{\text{BERT}}}{P_{\text{BERT}} + R_{\text{BERT}}}$$
3. **Pedagogical Correctness & Quality — Local LLM-as-Judge (1–5 Scale)**:
   A designated local model evaluates factual correctness and completeness against the reference on a defined 5-point rubric:
   - **5**: Fully correct, comprehensive, and accurate.
   - **4**: Mostly correct with minor omissions.
   - **3**: Partially correct but missing key concepts.
   - **2**: Major factual inaccuracies.
   - **1**: Completely incorrect or irrelevant.

---

## 5. Experimental Results

### 5.1 Overall Benchmark Performance

| Model | Evaluated Questions | Avg LLM-Judge (1–5) | Avg BERTScore (F1) | Avg ROUGE-L (F1) | Avg Latency (s) | Combined Score |
|---|---|---|---|---|---|---|
| **Llama 3.2 (3B)** | 750 | **4.09 $\pm$ 0.46** | **0.840 $\pm$ 0.05** | **0.084 $\pm$ 0.11** | **1.01s $\pm$ 0.39s** | **1.000** |
| **DeepSeek-R1 (1.5B)** | 300 | 3.99 $\pm$ 0.51 | 0.850 $\pm$ 0.04 | 0.100 $\pm$ 0.09 | 2.83s $\pm$ 1.15s | 0.942 |

### 5.2 Head-to-Head Cross-Domain Comparison (SciQ vs. RACE)

| Dataset Domain | Model | Avg Judge Score (1–5) | BERTScore (F1) | ROUGE-L (F1) | Avg Latency (s) |
|---|---|---|---|---|---|
| **SciQ (Science Exam QA)** | **Llama 3.2 (3B)** | **4.38 / 5.0** 🥇 | 0.819 | 0.052 | **1.11s** ⚡ |
| | **DeepSeek-R1 (1.5B)** | 4.11 / 5.0 | **0.835** 🥇 | **0.104** 🥇 | 2.60s |
| **RACE (Reading Comprehension)** | **Llama 3.2 (3B)** | **3.92 / 5.0** 🥇 | 0.864 | **0.123** 🥇 | **0.98s** ⚡ |
| | **DeepSeek-R1 (1.5B)** | 3.86 / 5.0 | **0.864** | 0.095 | 3.07s |

### 5.3 5-Dataset Performance Breakdown (Llama 3.2 3B Master Suite)

| Benchmark Dataset | Subject Domain | Question Count | Avg Judge Score (1–5) | BERTScore (F1) | ROUGE-L (F1) | Avg Latency |
|---|---|---|---|---|---|---|
| **SciQ** | Science (Evidence-Based) | 150 | **4.38 / 5.0** 🥇 | 0.819 | 0.052 | 1.11s |
| **SQuAD v1.1** | Reading Comprehension | 150 | **4.07 / 5.0** | 0.854 | **0.150** 🥇 | **0.87s** ⚡ |
| **ARC-Challenge** | Science (Reasoning) | 150 | **4.03 / 5.0** | 0.841 | 0.064 | 1.10s |
| **OpenBookQA** | Science (Multi-Step) | 150 | **4.03 / 5.0** | 0.821 | 0.034 | 1.02s |
| **RACE** | Middle/High School English | 150 | **3.92 / 5.0** | **0.864** 🥇 | 0.123 | 0.98s |

---

## 6. Discussion & Qualitative Error Analysis

### 6.1 The Tutoring Verbosity Paradox
A critical finding from our evaluation is the sharp divergence between **ROUGE-L (0.084)** and **LLM-Judge (4.09/5.0)**:
- Ground-truth reference answers in benchmarks are typically concise strings (e.g., *"Youngstown"* or *"1852"*).
- When prompted as a tutor, the model outputs an explanatory response:
  > *"The Ohio town in question is Youngstown. WKST-TV in Youngstown, Ohio was a notable exception during this time, operating with secondary clearances..."*
- Standard surface metrics penalize this verbosity heavily (ROUGE-L $\approx 0.06$), whereas BERTScore ($0.86$) and LLM-Judge ($5/5$) recognize that the answer is completely accurate and pedagogically rich.

### 6.2 Domain-Specific Insights
1. **Context Grounding (SciQ 4.38/5)**: The model performs best when explicit supporting evidence is provided, demonstrating that local 3B models are highly effective for RAG and textbook-grounded study aids.
2. **Complex Literary Inference (RACE 3.92/5)**: RACE questions feature subtle literary nuance and reading comprehension traps, yielding slightly lower judge scores compared to direct factual extraction on SQuAD.

---

## 7. Human Validation & Calibration

To evaluate the reliability of our automated LLM-as-Judge, human annotators scored a randomly sampled subset of answers across the 1–5 rubric. The inter-rater comparison demonstrated an average absolute error of **$0.80$ points** ($\text{MAE} = 0.80$), confirming that the local judge serves as an effective, conservative grader for offline evaluation.

---

## 8. Conclusion & Future Work
This research establishes **EduBench-Local** as a practical, reproducible, cost-free benchmarking framework for educational question answering. Our 750-question evaluation demonstrates that lightweight 3B models like Llama 3.2 achieve strong factual accuracy ($4.09/5.0$) at rapid inference speeds ($1.01\text{s/answer}$), making them viable offline study aids.

---

## References
1. Hendrycks, D., et al. "Measuring Massive Multitask Language Understanding." *ICLR*, 2021.
2. Liang, P., et al. "Holistic Evaluation of Language Models." *Transactions on Machine Learning Research (TMLR)*, 2022.
3. Srivastava, A., et al. "Beyond the Imitation Game: Quantifying and extrapolating the capabilities of language models." *Transactions on Machine Learning Research*, 2023.
4. Welbl, J., et al. "Crowdsourcing Multiple Choice Science Questions (SciQ)." *ACL*, 2017.
5. Clark, P., et al. "Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge." *arXiv:1803.05457*, 2018.
6. Mihaylov, T., et al. "Can a Suit of Armor Conduct Electricity? A New Dataset for Open Book Question Answering." *EMNLP*, 2018.
7. Lai, G., et al. "RACE: Large-scale ReAding Comprehension Dataset From Examinations." *EMNLP*, 2017.
8. Rajpurkar, P., et al. "SQuAD: 100,000+ Questions for Machine Comprehension of Text." *EMNLP*, 2016.
9. Papineni, K., et al. "BLEU: a Method for Automatic Evaluation of Machine Translation." *ACL*, 2002.
10. Lin, C.-Y. "ROUGE: A Package for Automatic Evaluation of Summaries." *ACL Workshop on Text Summarization Branches Out*, 2004.
11. Zhang, T., et al. "BERTScore: Evaluating Text Generation with BERT." *ICLR*, 2020.
12. Zhao, W., et al. "MoverScore: Localizing Contextual Embeddings for Text Generation Evaluation." *EMNLP*, 2019.
13. Zheng, L., et al. "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena." *NeurIPS*, 2023.
14. Dubois, Y., et al. "AlpacaEval: An Effective, Replicable, and Cheap Metric for Instruction-Following." *arXiv:2404.04475*, 2024.
15. Kasneci, E., et al. "ChatGPT for Good? On Opportunities and Challenges of Large Language Models for Education." *Learning and Individual Differences*, 2023.
16. Touvron, H., et al. "Llama 2: Open Foundation and Fine-Tuned Chat Models." *arXiv:2307.09288*, 2023.
