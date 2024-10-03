import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Step 1: Load the original Online Retail II dataset
def load_data(file_path):
    logging.info(f"Loading data from: {file_path}")
    data = pd.read_excel(file_path)
    logging.info(f"Data successfully loaded with {len(data)} records.")
    return data

# Step 2: Clean and preprocess the dataset
def preprocess_data(data):
    logging.info("Preprocessing data: cleaning and handling missing values.")
    # Convert 'InvoiceDate' to datetime and ensure numerical consistency
    data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'], errors='coerce')
    data['Quantity'] = pd.to_numeric(data['Quantity'], errors='coerce')
    data['Price'] = pd.to_numeric(data['Price'], errors='coerce')
    data['Customer ID'] = pd.to_numeric(data['Customer ID'], errors='coerce')

    # Remove cancelled orders (invoices starting with 'C')
    data = data[~data['Invoice'].str.startswith('C', na=False)]

    # Drop rows with missing values in key columns
    data = data.dropna(subset=['InvoiceDate', 'Customer ID', 'Quantity', 'Price'])

    logging.info(f"Data cleaned. Remaining records: {len(data)}.")
    return data

# Step 3: Group by CustomerID and InvoiceNo
def group_by_customer_invoice(data):
    logging.info("Grouping by Customer ID and Invoice Number.")
    # Group by CustomerID and InvoiceNo to represent each invoice as a time series record
    grouped = data.groupby(['Customer ID', 'Invoice']).agg({
        'InvoiceDate': 'first',  # First date of the invoice
        'Quantity': 'sum',       # Sum of quantities in the invoice
        'Price': 'mean'          # Average price in the invoice
    }).reset_index()

    logging.info(f"Grouped data created with {len(grouped)} records.")
    return grouped

# Step 4: Save the restructured dataset
def save_data(grouped_data, output_file):
    logging.info(f"Saving restructured data to {output_file}.")
    grouped_data.to_csv(output_file, index=False)
    logging.info("Data successfully saved.")

# Main function to run the entire preprocessing pipeline
def main():
    file_path = 'online_retail_II.xlsx'
    output_file = ('restructured_ts2vec_online_retail.csv')

    # Load and preprocess the data
    data = load_data(file_path)
    cleaned_data = preprocess_data(data)

    # Group data by CustomerID and InvoiceNo
    grouped_data = group_by_customer_invoice(cleaned_data)

    # Save the restructured dataset
    save_data(grouped_data, output_file)

if __name__ == "__main__":
    main()