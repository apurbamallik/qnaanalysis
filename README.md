# EduBench-Local: Comparative Performance Analysis of Open-Source LLMs for Educational Question Answering

**EduBench-Local** is a fully offline, cost-free benchmark pipeline designed for student research to compare open-source Large Language Models (LLMs) on educational question answering across subjects using a tri-metric evaluation system.

---

## 🏗️ Architecture & Pipeline Overview

```
                          ┌──────────────────────────┐
                          │   Hugging Face Datasets  │
                          │     (SciQ & SQuAD/ARC)   │
                          └─────────────┬────────────┘
                                        │
                             [prepare_datasets.py]
                                        │
                                        ▼
                          ┌──────────────────────────┐
                          │   dataset_sample.json    │
                          └─────────────┬────────────┘
                                        │
                             [generate_answers.py]  ◄──  Ollama Local LLMs
                                        │               (Llama 3.2, Mistral, Gemma 2, etc.)
                                        ▼
                          ┌──────────────────────────┐
                          │     raw_answers.json     │
                          └─────────────┬────────────┘
                                        │
                             [evaluate_metrics.py]
                             ├── ROUGE-L (Lexical)
                             ├── BERTScore (Semantic)
                             └── LLM-as-Judge (Quality: 1-5)
                                        │
                                        ▼
                          ┌──────────────────────────┐
                          │    scored_results.csv    │
                          └─────────────┬────────────┘
                                        │
                    ┌───────────────────┴───────────────────┐
                    ▼                                       ▼
          [analyze_and_plot.py]                         [app.py]
   ├── Leaderboard & Combined Scores             Streamlit Interactive Dashboard
   ├── Subject Heatmaps & Radars                 ├── Leaderboard & KPIs
   ├── Efficiency Frontier                       ├── Qualitative Answer Inspector
   └── Wilcoxon Significance Tests               └── Side-by-Side Model Comparison
```

---

## 🚀 Quickstart Guide

### 1. Activate Environment
```powershell
.\edubench-env\Scripts\activate
```

### 2. Prepare Datasets (SciQ + Reading Comprehension)
```powershell
python prepare_datasets.py --sample-size 30 --seed 42
```
*Creates `sciq_sample.json`, `rc_sample.json`, and `dataset_sample.json`.*

### 3. Generate Answers Across Models
Pull any target models in Ollama:
```powershell
ollama pull llama3.2:3b
ollama pull mistral:7b
ollama pull gemma2:2b
ollama pull phi3:mini
ollama pull qwen2.5:3b
```

Run generation:
```powershell
python generate_answers.py --dataset dataset_sample.json --models llama3.2:3b mistral:7b gemma2:2b phi3:mini qwen2.5:3b
```
*(Supports checkpointing — resuming will not re-generate completed answers).*

### 4. Tri-Metric Evaluation
```powershell
python evaluate_metrics.py --input raw_answers.json --output-csv scored_results.csv --judge-model llama3.2:3b
```

### 5. Aggregation, Statistical Analysis & Visualizations
```powershell
python analyze_and_plot.py --input scored_results.csv --output-dir plots
```
Generates:
- `plots/leaderboard_bar.png`
- `plots/subject_heatmap.png`
- `plots/radar_metrics.png`
- `plots/efficiency_frontier.png`
- `plots/leaderboard_summary.csv`
- `plots/wilcoxon_p_values.csv`

### 6. Launch Interactive Dashboard
```powershell
streamlit run app.py
```

---

## 📐 The Three Evaluation Metrics

| Metric | Category | Dimension Captured | Library / Backend |
|---|---|---|---|
| **ROUGE-L** | Lexical | Surface word & phrase overlap with reference | `rouge-score` |
| **BERTScore (F1)** | Semantic | Contextual embedding similarity (paraphrase-robust) | `bert-score` (`roberta-large`) |
| **LLM-as-Judge (1–5)** | Correctness / Quality | Factual accuracy and conceptual completeness | Local Ollama Model |

---

## 📁 Repository Structure

```
d:/edubench-project/
├── prepare_datasets.py      # Dataset loading & schema standardization
├── generate_answers.py     # Multi-model inference & latency tracking
├── evaluate_metrics.py     # ROUGE-L, BERTScore, LLM-as-Judge scoring
├── analyze_and_plot.py     # Aggregation, Wilcoxon tests, high-res plots
├── app.py                  # Streamlit interactive UI dashboard
├── dataset_sample.json     # Standardized benchmark question set
├── raw_answers.json        # Raw model outputs with generation metadata
├── scored_results.csv      # Evaluated dataset with all metric scores
└── plots/                  # Publication-ready figures & statistical CSVs
    ├── leaderboard_bar.png
    ├── subject_heatmap.png
    ├── radar_metrics.png
    ├── efficiency_frontier.png
    └── leaderboard_summary.csv
```
