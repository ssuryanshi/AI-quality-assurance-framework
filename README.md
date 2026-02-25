# 🔍 AI Model QA & Hallucination Detection Framework

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Pytest](https://img.shields.io/badge/tests-pytest-green.svg)](https://pytest.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)

An automated framework to **evaluate AI/LLM responses** for accuracy, hallucination, bias, and consistency using structured datasets and regression pipelines. Built for QA engineers, ML teams, and anyone deploying LLMs in production.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    AI Model QA Framework                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │   Datasets    │───▶│  Model Query │───▶│  Evaluation Engine   │  │
│  │   Manager     │    │   Engine     │    │                      │  │
│  │              │    │              │    │  ┌────────────────┐  │  │
│  │ • factual_qa │    │ • OpenAI API │    │  │  Accuracy      │  │  │
│  │ • consistency│    │ • HuggingFace│    │  │  Hallucination │  │  │
│  │ • hallucinate│    │ • Retry/Back │    │  │  Consistency   │  │  │
│  └──────────────┘    └──────────────┘    │  │  Bias Detection│  │  │
│                                          │  └────────────────┘  │  │
│                                          └──────────┬───────────┘  │
│                                                     │              │
│                      ┌──────────────────────────────┤              │
│                      ▼                              ▼              │
│  ┌──────────────────────┐    ┌──────────────────────────┐          │
│  │  Regression Engine   │    │   Reporting System       │          │
│  │                      │    │                          │          │
│  │ • Baseline Manager   │    │  • CSV Reports           │          │
│  │ • Metric Comparison  │    │  • Matplotlib Charts     │          │
│  │ • Degradation Alert  │    │  • Dashboard PNG         │          │
│  └──────────────────────┘    └──────────────────────────┘          │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  CI/CD: GitHub Actions  │  Container: Docker & Compose     │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
AI-Quality-Assurance-Framework/
├── datasets/                    # Test datasets
│   ├── factual_qa.json          # 50 factual Q&A pairs
│   ├── consistency_prompts.json # 20 rephrased prompt sets
│   ├── hallucination_prompts.json # 20 adversarial prompts
│   └── dataset_loader.py       # Dataset loading & validation
│
├── models/                      # Model adapters
│   ├── base_model.py            # Abstract base with retry logic
│   ├── openai_model.py          # OpenAI Chat Completions wrapper
│   ├── huggingface_model.py     # HuggingFace Inference API wrapper
│   └── model_factory.py         # Factory pattern model creation
│
├── evaluation/                  # Evaluation engine
│   ├── accuracy.py              # Exact, fuzzy, semantic matching
│   ├── hallucination.py         # Multi-strategy hallucination detection
│   ├── consistency.py           # Pairwise response consistency
│   ├── bias_detector.py         # Sentiment, demographic, stereotype detection
│   └── metrics.py               # Aggregated metrics & composite scoring
│
├── regression/                  # Regression testing
│   ├── baseline_manager.py      # Save/load metric baselines
│   └── regression_runner.py     # Compare & detect regressions
│
├── reports/                     # Reporting system
│   ├── csv_reporter.py          # CSV output generation
│   ├── visual_reporter.py       # Matplotlib chart generation
│   └── report_generator.py      # Report orchestrator
│
├── tests/                       # Pytest test suite
│   ├── conftest.py              # Shared fixtures & mock data
│   ├── test_dataset_loader.py   # Dataset tests (16 cases)
│   ├── test_accuracy.py         # Accuracy tests (22 cases)
│   ├── test_hallucination.py    # Hallucination tests (12 cases)
│   ├── test_consistency.py      # Consistency tests (13 cases)
│   ├── test_bias_detector.py    # Bias detection tests (8 cases)
│   ├── test_regression.py       # Regression tests (10 cases)
│   └── test_reporters.py        # Reporter tests (13 cases)
│
├── scripts/                     # CLI entry points
│   ├── run_evaluation.py        # Full evaluation pipeline
│   └── run_regression.py        # Regression test runner
│
├── .github/workflows/           # CI/CD
│   ├── ci.yml                   # Test on push/PR
│   └── evaluation.yml           # Scheduled evaluation
│
├── config.yaml                  # Configuration file
├── .env.example                 # API key template
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Container build
├── docker-compose.yml           # Container orchestration
└── README.md                    # This file
```

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/your-username/ai-qa-framework.git
cd ai-qa-framework

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
# Copy the template
cp .env.example .env

# Edit .env with your keys
# OPENAI_API_KEY=sk-your-key-here
# HUGGINGFACE_API_TOKEN=hf_your-token-here
```

### 3. Run Tests (No API Keys Needed)

```bash
# Run the full test suite
python -m pytest tests/ -v

# With coverage
python -m pytest tests/ -v --cov=evaluation --cov=datasets --cov-report=term-missing
```

### 4. Run Evaluation (Requires API Key)

```bash
# Full evaluation pipeline
python scripts/run_evaluation.py

# Specific model provider
python scripts/run_evaluation.py --model openai
python scripts/run_evaluation.py --model huggingface

# Custom dataset
python scripts/run_evaluation.py --dataset datasets/factual_qa.json
```

### 5. Run Regression Test

```bash
python scripts/run_regression.py
```

---

## Docker

```bash
# Build the image
docker build -t ai-qa-framework .

# Run evaluation
docker run --env-file .env ai-qa-framework

# Run tests
docker run ai-qa-framework python -m pytest tests/ -v

# Docker Compose
docker compose up evaluator
docker compose --profile test up test
```

---

## Evaluation Metrics

| Dimension | Metrics | Weight |
|-----------|---------|--------|
| **Accuracy** | Exact Match Rate, Fuzzy Score, Semantic Similarity | 35% |
| **Hallucination** | Contradiction, Fabrication, Hallucination Rate | 30% |
| **Consistency** | Pairwise Similarity, Contradiction Detection | 20% |
| **Bias** | Sentiment Skew, Demographic Balance, Stereotypes | 15% |

### Overall Score Formula

```
Overall = (Accuracy × 0.35) + ((100 - HallucinationRate) × 0.30) +
          (Consistency × 0.20) + ((100 - BiasRate) × 0.15)
```

---

## Generated Reports

After running an evaluation, reports are saved to the `reports/` directory:

| Type | File | Description |
|------|------|-------------|
| CSV | `reports/csv/summary_*.csv` | All metrics in tabular format |
| CSV | `reports/csv/accuracy_detail_*.csv` | Per-question accuracy scores |
| CSV | `reports/csv/hallucination_detail_*.csv` | Per-prompt hallucination flags |
| PNG | `reports/charts/accuracy_metrics.png` | Accuracy bar & pie charts |
| PNG | `reports/charts/hallucination_metrics.png` | Hallucination rate visualization |
| PNG | `reports/charts/consistency_metrics.png` | Consistency score distribution |
| PNG | `reports/charts/regression_comparison.png` | Baseline vs current comparison |
| PNG | `reports/charts/overall_dashboard.png` | Combined metrics dashboard |

---

## Configuration

Edit `config.yaml` to customize:

```yaml
model:
  provider: "openai"        # or "huggingface"
  openai:
    model_name: "gpt-3.5-turbo"
    temperature: 0.0

evaluation:
  accuracy:
    fuzzy_match_threshold: 0.80
  hallucination:
    max_acceptable_rate: 0.10

regression:
  degradation_threshold: 0.05  # Flag if metric drops >5%
```

---

## CI/CD Pipeline

### Automatic Testing (`ci.yml`)
- **Trigger**: Push/PR to `main`
- **Matrix**: Python 3.10, 3.11, 3.12
- **Steps**: Install → Lint → Test → Upload Coverage

### Scheduled Evaluation (`evaluation.yml`)
- **Trigger**: Weekly + manual dispatch
- **Steps**: Install → Evaluate → Regression Check → Upload Reports

### Setting Up Secrets
Add your API keys to GitHub repository secrets:
- `OPENAI_API_KEY`
- `HUGGINGFACE_API_TOKEN`

---


## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Roadmap

- [ ] Multi-model comparison dashboards
- [ ] Fine-tuned dataset evaluation
- [ ] Human feedback loop integration
- [ ] Slack/email alerts for regressions
- [ ] Web-based dashboard UI
- [ ] Custom metric plugin system
