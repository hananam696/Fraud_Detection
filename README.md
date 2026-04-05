# Fraud_Detection
This project involves conducting comprehensive data analysis, developing predictive models, and implementing an end-to-end machine learning pipeline. 

# Bornze Layer: Understanding and loading the raw data
 ## Data Source

Dataset downloaded from Kaggle using Kaggle API.  
The dataset contains 5 files:
- transactions  
- cards  
- users  
- fraud labels  
- merchant data  

All 5 files were used in this project.

## Data Understanding & Initial EDA

Initial exploration was done to:
- understand data structure  
- identify missing values  
- detect inconsistent values  
- analyze distributions and patterns  

Visualizations were created to understand fraud behavior and feature relationships.

The processed data was saved as parquet files for faster loading and efficient storage.

Refer: `notebooks/01_EDA_and_Data_understanding.ipynb`
