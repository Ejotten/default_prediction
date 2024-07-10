# ML Model with PyCaret
[![Python](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/)

This Streamlit application uses a pre-trained Light Gradient Boosting Machine (LGBM) model to predict the default probability of customers. Users can either upload a dataset or manually input customer information to receive predictions.

# Dependencies
The following dependencies are required to run this application:

pandas

streamlit

pycaret

xlsxwriter

# Usage
## Uploading a Dataset
Choose the "Upload file" option from the sidebar.
Upload your dataset in CSV or Feather format.
The application will display a preview of the uploaded data.
Predictions will be made on the uploaded data, you will be able to dowload the results as an Excel file.
## Manual Input
Choose the "Manual input" option from the sidebar.
Fill in the customer information in the provided form.
Click the "Predict" button to receive the prediction.
The application will display the prediction results, 0 being no default and 1 as default probability.

Streamlit Video:

For a detailed walkthrough of the application, please refer to the following video:


https://github.com/Ejotten/default_prediction/assets/114586943/addd02b1-9fd0-4e97-85e8-58cf710bffcf


