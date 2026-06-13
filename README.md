# 🚀 AI Price Predictor - ML, Neural Networks & LLMs

A comprehensive end-to-end Machine Learning and AI project that predicts product prices from textual descriptions using Traditional Machine Learning, Neural Networks, and Large Language Models (LLMs).

The project follows the complete AI development lifecycle from data curation and preprocessing to model evaluation, deep learning, frontier LLM experimentation, and fine-tuning workflow preparation.

---

## 📌 Features

* 📦 Product Price Prediction from Text Descriptions
* 🧹 Data Curation & Preprocessing Pipeline
* 📊 Traditional Machine Learning Models
* 🧠 Deep Neural Network using PyTorch
* 🤖 Frontier LLM Evaluation
* 🏷️ Fine-Tuning Dataset Preparation
* 📈 Model Evaluation & Benchmarking
* ⚖️ Human vs AI Performance Comparison
* 📁 JSONL Dataset Generation for Fine-Tuning

---

## 🏗️ Project Workflow

```text
Amazon Product Data
        ↓
Data Curation
        ↓
Data Preprocessing
        ↓
Traditional ML Models
        ↓
Neural Networks
        ↓
Frontier LLM Evaluation
        ↓
Fine-Tuning Preparation
        ↓
Performance Analysis
```

---

## 📅 Project Stages

### 📦 Day 1 – Data Curation

* Collected Amazon product data
* Removed noisy and invalid records
* Structured raw product information
* Created clean datasets for downstream tasks

### 🧹 Day 2 – Data Preprocessing

* Text cleaning and normalization
* Feature extraction and preparation
* Dataset transformation into model-ready format
* Structured product summaries generation

### 📊 Day 3 – Traditional Machine Learning

Baseline Models

Established benchmark performance using:

Random Pricer Baseline
Constant Average Price Baseline

These baselines provided reference points for comparing more advanced machine learning models.

Traditional Machine Learning

Implemented and evaluated:

Linear Regression
Bag of Words + Linear Regression
Random Forest Regressor
XGBoost Regressor

Compared models using prediction error metrics and benchmarking techniques.

### 🧠 Day 4 – Neural Networks & LLMs

#### Human Evaluation

Created a human benchmark using:

human_in.csv containing product descriptions
human_out.csv containing human-estimated prices

This enabled comparison between human performance and AI-based price prediction systems.

#### Neural Network

* Built a Deep Neural Network using PyTorch
* Applied HashingVectorizer for text representation
* Trained the network to predict product prices

#### Frontier LLM Evaluation

Evaluated frontier models for zero-shot price estimation:

* GPT-4.1 Nano
* GPT-5.1
* Gemini 2.5 Flash Lite

Compared:

* Human Predictions
* Baseline Models
* Traditional ML Models
* Neural Network Predictions
* LLM Predictions

### 🏷️ Day 5 – Fine-Tuning Preparation

Implemented the complete fine-tuning workflow:

* Generated training datasets in JSONL format
* Created validation datasets
* Prepared OpenAI fine-tuning pipeline
* Configured training and evaluation workflow

> **Note:** Fine-tuning datasets and pipeline were successfully prepared. Full fine-tuning execution was limited by API quota and billing constraints.

---

## 🤖 Models Explored

### Baselines
* Random Pricer
* Constant Average Price Pricer

### Traditional Machine Learning

* Linear Regression
* Random Forest
* XGBoost

### Deep Learning

* Feed Forward Neural Network (PyTorch)

### Frontier LLMs

* GPT-4.1 Nano
* GPT-5.1
* Gemini 2.5 Flash Lite
---

## 📊 Evaluation Metrics

| Metric                   | Description                                   |
| ------------------------ | --------------------------------------------- |
| Error                    | Difference between actual and predicted price |
| Mean Squared Error (MSE) | Average squared prediction error              |
| R² Score                 | Model explanatory power                       |
| Human Comparison         | Human vs AI prediction performance            |

---

## ⚙️ Tech Stack

### Programming Language

* Python

### Machine Learning

* Scikit-learn
* XGBoost

### Deep Learning

* PyTorch

### Large Language Models

* OpenAI
* Gemini
* LiteLLM

### Data Processing

* NumPy
* Pandas

### Utilities

* Hugging Face Hub
* tqdm
* python-dotenv

---

## 🎯 Future Improvements

* Complete LLM Fine-Tuning Experiments
* Deploy as a Web Application
* Evaluate Additional Frontier Models
* Advanced Feature Engineering
* Larger Dataset Experiments
* Automated Benchmarking Dashboard

---

## 📜 License

This project is intended for educational, research, and learning purposes.

---

## 👨‍💻 Author

**N. Raja Sujith Varma**

AI | Machine Learning | Deep Learning | LLM Engineering | Generative AI
