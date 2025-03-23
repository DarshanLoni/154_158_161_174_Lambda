InfluencerIQ

Smart Insights for Influencer Marketing
Overview

InfluencerIQ is an AI-powered influencer ranking and analytics platform that helps brands identify top-performing influencers based on real data insights. The system processes influencer data, cleans it, and ranks influencers based on key performance metrics, making it easier for brands to make informed decisions.
Features

* AI-powered Influencer Ranking – Uses performance metrics to rank influencers.
* Data Cleaning & Preprocessing – Handles missing values and standardizes data.
* Dynamic Filtering – Enables brands to find influencers based on specific criteria.
* Interactive Dashboard – Provides a user-friendly UI for insights.
* Downloadable Reports – Users can export ranked influencer data.

Tech Stack

Component -	Technology
Frontend  -	Streamlit
Backend	  -  Python (Pandas, NumPy)
Database  -	CSV-based data processing
Deployment - Streamlit Cloud / Local Execution

Project Structure

influencerIQ/
│── data/  
│   ├── influencers.csv  # Raw dataset  
│   ├── ranked_influencers.csv  # Processed & ranked dataset  
│── src/  
│   ├── data_processing.py  # Handles data cleaning & ranking  
│   ├── app.py  # Streamlit dashboard for visualization  
│── README.md  # Project documentation  
│── requirements.txt  # Dependencies  


Installation & Setup

Clone the Repository:
git clone https://github.com/DarshanLoni/InfluenceIQ.git
cd influencerIQ
Install Dependencies:
pip install -r requirements.txt
Run the Streamlit App:
streamlit run src/app.py


Usage

Upload your dataset (influencers.csv).
View ranked influencers based on performance metrics.
Apply filters (e.g., country, engagement rate).
Download the processed dataset for further analysis.


Resources & References

Research Papers on influencer marketing analytics.
Case Studies from brands using AI for influencer selection.
Documentation for Streamlit, Pandas, and NumPy.
Open-source influencer datasets and benchmarking studies.