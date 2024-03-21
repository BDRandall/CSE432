import pandas as pd
import os
from constants import DATA_DIR


# def convert_xls_to_csv(file_path):
#     # Read the HTML tables
#     tables = pd.read_html(file_path)
    
#     # This code assumes that your data is in the first table
#     df = tables[0]
    
#     # Convert to CSV
#     csv_file = file_path.replace('.xls', '.csv')
#     df.to_csv(csv_file, index=False, header=True)
    
#     # Delete the original .xls file
#     os.remove(file_path)
#     print(f"Deleted original file: {file_path}")


def remove_asterisks_from_column(df, column_name):
    if df[column_name].dtype == object:  # Check if the column is of object type (string)
        df[column_name] = df[column_name].str.replace('*', '', regex=False)


# def clean_headers(file_path):
#     # Read the CSV file
#     df = pd.read_csv(file_path)
    
#     # Rename columns that contain 'Unnamed'
#     df.columns = [col if 'Unnamed' not in col else '' for col in df.columns]

#     # for col in df.columns:
#     #     for row in df.index:
#     #         cell_value = df.at[row, col]
#     #         if isinstance(cell_value, object):
#     #             # Remove asterisks from string
#     #             df.at[row, col] = cell_value.replace('*', '')
                
#     #             if cell_value.__contains__('Unnamed'):
#     #                 df.at[row, col] = ''
    
#     # Save the cleaned data to CSV
#     df.to_csv(file_path, index=False, header=True)
    
#     print(f"Cleaned file: {file_path}")


# def convert_xls():
#     for filename in os.listdir(DATA_DIR):
#         if filename.endswith('.xls'):
#             file_path = os.path.join(DATA_DIR, filename)
#             convert_xls_to_csv(file_path)
            
#     for filename in os.listdir(DATA_DIR):
#         if filename.endswith('.csv'):
#             file_path = os.path.join(DATA_DIR, filename)
#             clean_headers(file_path)

# convert_xls()
