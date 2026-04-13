# 📧 SMS Spam Classifier

ML-powered API for detecting spam messages using FastAPI, scikit-learn, and MLflow.

## ✨ Features
- 🤖 Binary classification model (spam/ham)
- 🔬 MLflow experiment tracking & model registry
- 🐳 Docker containerization
- 🧪 pytest test suite with coverage
- ⚡ FastAPI with async support
- 🔐 Environment-based configuration

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- Docker (optional)

### Local Development
```bash
# Clone & setup
git clone https://github.com/YOUR_USERNAME/sms-spam-detector.git
cd sms-spam-detector

# Install dependencies
uv sync --all-extras

# Activate environment
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows

# Run tests
uv run pytest

# Start API server
uv run uvicorn src.app:app --reload

# Access docs: http://localhost:8000/docs