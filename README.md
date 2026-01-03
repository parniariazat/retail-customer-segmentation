# Retail Customer Segmentation (RFM + K-Means)

## Overview
This project performs customer segmentation and value analysis using RFM metrics and K-Means clustering on real retail transaction data.

## Methodology
- Data cleaning and preparation
- RFM feature engineering
- Feature scaling and transformation
- K-Means clustering
- Segment profiling and visualization

## Segments (k=4)
- High-Value Loyal Customers
- Growth Potential Customers
- At-Risk Customers
- Low-Value / Occasional Customers

## Project Structure
- `src/analysis.py` — end-to-end analysis
- `figures/` — charts (Elbow, Silhouette, segment plots)
- `reports/` — executive summary
- `data/` — dataset placeholder (not included)

## Tools
Python (pandas, scikit-learn), Excel, matplotlib, seaborn

## Author
Parnia Riazat

## How to Run
```bash
pip install -r requirements.txt
python src/analysis.py

