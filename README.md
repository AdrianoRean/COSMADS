# COSMADS: Composing SMArt Data Services through Large Language Models
This repository contains code for replicating the experiments in *"Composing SMArt Data Services through Large Language Models"* paper.


## Prerequisites
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html)


## Setup
- Create a virtual environment and install the dependencies
```bash
conda create -n pyllm python=3.9
conda activate pyllm
pip install -r requirements.txt
ollama pull llama3.2
```



## Usage

- Define the query in the [`json`](src/queries_pipelines.json) file.\
    As an example:
    ```bash
    "q5": {
        "query": "Please provide a table for the upcoming 30 cardboard pieces processed by the diecutter with ID 7, detailing (i) how many cardboard pieces are defect-free and (ii) how many contain defects.",
    }
    ```

- In the [main](src/main.py?plain=1#L293) file, specify the `<query_number>` to be executed.\
    As an **example**:
    ```bash
    ...
    if __name__ == "__main__":
        q = "q5"
        ...
    ```

- Run the LLM:
    ```bash
    cd src
    python main.py
    ```

- The LLM will generate a `temp_pipeline.py` file with the Python pipeline leveraging the proper `data services` to generate the requested information. \
Given the **Example**, the LLM will generate a schema as follows.
    ```bash
    +----+--------------------+---------------------+
    |    |   no_defects_count |   with_errors_count |
    |----+--------------------+---------------------|
    |  0 |                 17 |                  13 |
    +----+--------------------+---------------------+
    ```


## How to replicate the experiments

To run the experiments, execute the following command:
```bash
cd src
python run_evaluation.py
```

The script will create different `.csv` in [evaluation](src/evaluation/) folder containing the results of the run and computed metrics.


### Experiments results

[evaluation](src/evaluation/) folder contains the results of the experiments:
- COSMADS:
    - [evaluation results](src/evaluation/evaluation_results_standard.csv)
    - [metrics](src/evaluation/metrics_results_standard.csv)
- COSMADS (w/o similar pipelines):
    - [evaluation results](src/evaluation/evaluation_results_wrong.csv)
    - [metrics](src/evaluation/metrics_results_wrong.csv)
- COSMADS (w/o pipelines):
    - [evaluation results](src/evaluation/evaluation_results_wo_pipeline.csv)
    - [metrics](src/evaluation/metrics_results_wo_pipeline.csv)
- GitHub Copilot:
    - [evaluation results](src/evaluation/evaluation_results_copilot.csv)
    - [metrics](src/evaluation/metrics_results_copilot.csv)
