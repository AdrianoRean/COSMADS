import pandas as pd

file_path = "data.csv"

def average_results(df = pd.DataFrame(), mode = "valentine"):
    if mode == "unified":
        mode = "valentine"

    # File path to the .txt file
     # Replace with your actual file name

    # Reading the .txt file into a DataFrame
    if df.empty:
        if mode == "valentine":
            df = pd.read_csv(file_path)
        elif mode == "llm":
            df = pd.DataFrame(data=[[],[]], columns=["index", "verdict"])
        else:
            df = pd.DataFrame(data=[[],[]], columns=["accuracy", "recall"])
    

    # Calculating the average for the last three columns
    if mode == "valentine":
        averages = df[['precision', 'acc_cell', 'acc_row', 'recall']].mean()
    elif mode == "llm":
        num_of_verdict = df.shape[0]
        averages = df.groupby(["verdict"]).count()
        averages.apply({lambda x: x/num_of_verdict})
    else:
        averages = df[["accuracy", "recall"]].mean()
    
    return averages
    
##average_results(file_path1)
##print("\n ----------- \n")
if __name__ == "__main__":
    averages = average_results()
    print("Average values:")
    print(f"Average acc_cell: {averages['acc_cell']:.6f}")
    print(f"Average acc_row: {averages['acc_row']:.6f}")
    print(f"Average recall: {averages['recall']:.6f}")