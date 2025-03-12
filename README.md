# House Price Prediction - Machine Learning Project with Local ETL Pipeline

## Overview
This project aims to predict house prices for properties in Gurgaon based on data scraped from [99acres.com](https://www.99acres.com/). It involves data processing, feature engineering, and machine learning model training to develop an efficient house recommendation system.

## Dataset
The dataset was collected through web scraping from 99acres.com, containing details about various properties in Gurgaon, such as:
- Number of rooms
- Locality
- Amenities (lift, parking, gym, etc.)
- Availability status
- Price
- Square footage

## Data Ingestion
The data ingestion process consists of:
1. **Web Scraping:** Using Python libraries like `BeautifulSoup` and `Scrapy` to extract structured data from 99acres.com.
2. **Data Storage:** Saving the scraped data in a CSV file for further processing.

## Data Processing & Feature Engineering
To make the dataset suitable for training, various preprocessing steps are applied:
1. **Handling Missing Values:** Using `SimpleImputer` to fill missing values with appropriate strategies (mean, median, or most frequent values).
2. **Feature Encoding:**
   - Categorical features are transformed using `OrdinalEncoder`.
   - One-hot encoding is applied where necessary.
3. **Scaling:**
   - `StandardScaler` is used to normalize numerical features.
   - `ColumnTransformer` integrates encoding and scaling in a single pipeline.

## Model Training
Various machine learning algorithms were trained and evaluated, including:
- **XGBoost:** Gradient boosting algorithm providing high performance.
- **Random Forest:** Ensemble learning method improving generalization.

## Training Pipeline
A structured training pipeline was implemented using object-oriented principles. The pipeline includes:
1. **Data Loading:** Fetching preprocessed data.
2. **Feature Transformation:** Applying transformations using `ColumnTransformer`.
3. **Model Training:** Training multiple models and selecting the best-performing one.
4. **Model Evaluation:** Measuring performance using RMSE, R² score, and MAE.

## Best Performing Model
After evaluating different models, the best-performing model was integrated into a house recommendation system to provide accurate price predictions.

## How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/house-price-prediction.git
   cd house-price-prediction
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the training pipeline:
   ```bash
   python train_pipeline.py
   ```

steps
1. conda create -n venv python=3.8 -y
2. conda activate venv
3. pip install -r requirements.txt


## Conclusion
This project effectively integrates web scraping, data preprocessing, and machine learning to create a robust house price prediction system. The structured training pipeline ensures high performance and seamless integration into recommendation systems.
