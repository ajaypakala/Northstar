# 📊 Northstar AI-Powered Business Analytics Platform

An AI-powered Data Analysis application built using **Python**, **Streamlit**, **Pandas**, **Plotly**, and **Google Gemini AI**.

---

# Features

- Upload CSV files
- Automatic Data Cleaning
- Dataset Preview
- Summary Statistics
- KPI Cards
- Revenue Analysis
- Top Customers
- Top Products
- Category-wise Analysis
- Interactive Plotly Charts
- AI Business Insights
- Chat with Your Data
- Executive Summary
- Marketing Recommendations

---

# Tech Stack

## Frontend

- Streamlit

## Backend

- Python

## Data Analysis

- Pandas

## Visualization

- Plotly

## AI

- Google Gemini API

## Environment Variables

- python-dotenv

---

# Project Structure

```text
AI_CSV_Analyzer/
│
├── app.py
├── requirements.txt
├── README.md
├── .env
├── test_ai.py
│
├── analysis/
│     analyzer.py
│
├── charts/
│     charts.py
│
├── ai/
│     ai_service.py
│
├── data/
│     sales.csv
│
├── reports/
│
└── utils/
```

---

# Installation

Clone the repository

```bash
git clone <repository-url>
```

Move into the project

```bash
cd Northstar 
```

Create a virtual environment

```bash
python -m venv venv
```

Activate the environment

Windows

```bash
venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Configure Gemini API

Create a `.env` file.

```env
GEMINI_API_KEY=YOUR_API_KEY
```

---

# Run the Application

```bash
streamlit run app.py
```

The application will open in your browser at:

```
http://localhost:8501
```

---

# Project Workflow

```text
Upload CSV
      │
      ▼
Load Dataset
      │
      ▼
Data Cleaning
      │
      ▼
Feature Engineering
      │
      ▼
Summary Statistics
      │
      ▼
Interactive Charts
      │
      ▼
AI Business Insights
      │
      ▼
Chat with Dataset
```

---

# Sample Questions

- Who is the highest spending customer?
- Which product generated maximum revenue?
- Give marketing suggestions.
- Summarize the business performance.
- What are the business risks?
- Which category contributes the most revenue?

---

# Future Improvements

- PDF Report Generation
- Excel Report Export
- AI Dashboard
- Forecasting
- SQL Database Support
- Multi-file Upload
- Authentication
- Docker Deployment
- Cloud Deployment
- RAG-based Data Chat

---

# Author

Ajaybabu Pakala

GitHub

https://github.com/ajaypakala

LinkedIn

www.linkedin.com/in/pakala-ajaybabu

---

# App Link

https://northstar-ajaypakala.streamlit.app/
