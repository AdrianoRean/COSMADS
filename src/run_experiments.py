import os

if __name__ == '__main__':
    databases = [
        "craftbeer",
        #"citeseer"
    ]

    models = [
        "openai",
        #"mistral",
        #"anthropic",
        #"deepseek"
    ]

    for database in databases:
        for model in models:
            print(f"Running experiments for database {database} and model {model}")
            result = os.system(f"python -u run_evaluation.py --database {database} --model {model}")
            if result != 0:
                print(f"Error running experiments for database {database} and model {model}")
            else:
                print(f"Experiments for database {database} and model {model} completed successfully")
            

