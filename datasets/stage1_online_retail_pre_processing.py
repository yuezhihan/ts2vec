import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Step 1: Load and Combine Data from Excel Sheets
def load_and_combine_sheets(file_path, sheets):
    """
    Load data from multiple Excel sheets and combine into a single DataFrame.
    """
    combined_data = pd.DataFrame()
    for sheet in sheets:
        logging.info(f"Loading data from sheet: {sheet}")
        sheet_data = pd.read_excel(file_path, sheet_name=sheet)
        combined_data = pd.concat([combined_data, sheet_data], ignore_index=True)
    logging.info("Data successfully loaded and combined.")
    return combined_data

# Step 2: Preprocess Data
def preprocess_data(data):
    """
    Preprocess data by converting dates, normalizing numeric columns, and handling missing values.
    """
    # Convert 'InvoiceDate' to datetime and ensure numerical consistency in key columns
    logging.info("Preprocessing data: converting dates and normalizing numeric columns.")
    data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'], errors='coerce')
    data['Quantity'] = pd.to_numeric(data['Quantity'], errors='coerce')
    data['Price'] = pd.to_numeric(data['Price'], errors='coerce')
    data['Customer ID'] = pd.to_numeric(data['Customer ID'], errors='coerce')

    # Remove rows where the Invoice starts with 'C' (canceled orders)
    data = data[~data['Invoice'].astype(str).str.startswith('C')]

    # Drop rows with missing critical data
    data = data.dropna(subset=['InvoiceDate', 'Customer ID', 'Quantity', 'Price'])

    # Normalize 'Quantity' and 'Price' using Min-Max scaling to insure values are positive
    scaler = MinMaxScaler()
    data[['Quantity', 'Price']] = scaler.fit_transform(data[['Quantity', 'Price']])
    logging.info("Data normalized and missing values handled.")

    return data

# Step 3: Aggregate Data
def aggregate_data(data):
    """
    Aggregate data by summing 'Quantity' and averaging 'Price' daily.
    """
    logging.info("Aggregating data by Date.")
    # Group by Date, aggregating Quantity and Price
    data_agg = data.groupby(pd.Grouper(key='InvoiceDate', freq='D')).agg({
        'Quantity': 'sum',
        'Price': 'mean'
    }).reset_index()

    logging.info("Data aggregation complete.")
    return data_agg

# Main Function to Run All Steps
def main():
    # File path and sheets to load
    file_path = 'online_retail_II.xlsx'
    sheets = ['Year 2009-2010', 'Year 2010-2011']
    
    # Load and preprocess the data
    combined_data = load_and_combine_sheets(file_path, sheets)
    cleaned_data = preprocess_data(combined_data)
    
    # Aggregate the data
    aggregated_data = aggregate_data(cleaned_data)

    # Save the final reshaped and adjusted data to CSV
    aggregated_data.to_csv('ts2vec_online_retail_II_data.csv', index=False)
    logging.info("Final data saved successfully.")

if __name__ == "__main__":
    main()
