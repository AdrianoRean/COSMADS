import pandas as pd

file_path = "data.csv"

def average_results(df = None):

    # File path to the .txt file
     # Replace with your actual file name

    # Reading the .txt file into a DataFrame
    if df == None:
        df = pd.read_csv(file_path)

    # Calculating the average for the last three columns
    averages = df[['acc_cell', 'acc_row', 'recall']].mean()
    
    return averages
    
##average_results(file_path1)
##print("\n ----------- \n")
if __name__ == "__main__":
    averages = average_results()
    print("Average values:")
    print(f"Average acc_cell: {averages['acc_cell']:.6f}")
    print(f"Average acc_row: {averages['acc_row']:.6f}")
    print(f"Average recall: {averages['recall']:.6f}")