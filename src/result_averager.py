import pandas as pd

file_path1 = "data.txt" 
file_path2 = "data copy.txt" 
file_path3 = "data3.txt" 

def average_results(file_path):

    # File path to the .txt file
     # Replace with your actual file name

    # Reading the .txt file into a DataFrame
    df = pd.read_csv(file_path, delim_whitespace=True)

    # Calculating the average for the last three columns
    averages = df[['acc_cell', 'acc_row', 'recall']].mean()

    # Printing the results
    print("Average values:")
    print(f"Average acc_cell: {averages['acc_cell']:.6f}")
    print(f"Average acc_row: {averages['acc_row']:.6f}")
    print(f"Average recall: {averages['recall']:.6f}")
    
##average_results(file_path1)
##print("\n ----------- \n")
average_results(file_path3)