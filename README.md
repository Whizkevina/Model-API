# Digital Health Demand Forecasting

## Overview
This project focuses on analyzing and predicting the demand for digital health solutions across various regions. By leveraging data on digital readiness metrics, such as the Digital Adoption Index (DAI) and its sub-indices, the project aims to identify regions with high demand for digital health solutions. The analysis includes data preprocessing, feature engineering, model training, and evaluation, culminating in a predictive model that forecasts demand levels.

## Objectives
1. **Analyze Digital Readiness Metrics**: Understand the factors influencing digital health adoption, including business, people, and government sub-indices.
2. **Predict Demand Levels**: Build a machine learning model to classify regions into high or low demand categories.
3. **Feature Importance Analysis**: Identify the most influential factors driving demand.
4. **Provide Scalable Predictions**: Enable predictions for new regions or future scenarios using the trained model.

## Data Sources
The project uses the following datasets:
- `hatch_data.csv`: Contains regional data on digital health adoption.
- `dai_data.csv`: Includes Digital Adoption Index (DAI) metrics and sub-indices for various regions and years.
- `stock_data.csv`: Provides additional contextual data for analysis.

## Methodology
1. **Data Preprocessing**:
   - Ensured consistent data types for merging (e.g., `Region` column).
   - Handled missing values by filling or dropping rows as needed.
   - Calculated adoption growth rates (`Adoption_Growth`) for each region over time.

2. **Feature Engineering**:
   - Created a `High_Demand` label based on the median adoption growth rate.
   - Normalized features using `StandardScaler` for model training.

3. **Model Training**:
   - Used a `RandomForestClassifier` to predict demand levels.
   - Split the data into training and testing sets (80/20 split).
   - Evaluated the model using metrics such as accuracy, precision, recall, and confusion matrix.

4. **Visualization**:
   - Generated feature importance plots to highlight key drivers of demand.
   - Visualized adoption growth trends across regions.

## Results
1. **Model Performance**:
   - Achieved high accuracy and precision in predicting demand levels.
   - Confusion matrix and classification report indicate strong performance on both high and low demand categories.

2. **Feature Importance**:
   - The `Digital Adoption Index` and `Adoption_Growth` were identified as the most influential features.

3. **Predictions**:
   - The model successfully predicts demand levels for new regions or future scenarios.

## Outputs
1. **Cleaned and Processed Data**:
   - `clean_df` contains the final dataset with calculated features and labels.
   - Exported to `forecasted_demand_output.xlsx`.

2. **Feature Importance Plot**:
   - Saved as `feature_importance.png`.

3. **Trained Model and Scaler**:
   - Model: `models/demand_forecasting_model.pkl`.
   - Scaler: `models/dai_scaler.pkl`.

4. **Streamlit Application**:
   - A user-friendly interface for predicting demand levels based on input metrics.

## How to Use
1. **Run the Analysis**:
   - Execute the Jupyter Notebook `healthPact.ipynb` to preprocess data, train the model, and generate outputs.

2. **Streamlit App**:
   - Launch the app using `streamlit run streamlit_app.py`.
   - Input digital readiness metrics to predict demand levels.

3. **Exported Files**:
   - Use `forecasted_demand_output.xlsx` for further analysis or reporting.

## Key Files
- `healthPact.ipynb`: Main analysis and model training notebook.
- `streamlit_app.py`: Streamlit application for demand prediction.
- `models/demand_forecasting_model.pkl`: Trained Random Forest model.
- `models/dai_scaler.pkl`: Scaler for feature normalization.
- `data/hatch_data.csv`, `data/dai_data.csv`, `data/stock_data.csv`: Input datasets.

## Conclusion
This project provides a comprehensive framework for analyzing and predicting digital health demand. By leveraging machine learning and feature engineering, it identifies key drivers of demand and enables scalable predictions for future scenarios. The results can guide stakeholders in prioritizing regions for digital health investments.

## Dependencies
- Python 3.x
- Libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `streamlit`, `joblib`

## License
This project is licensed under the MIT License.