# Data Service Composition in Cyber-Physical Systems Adopting LLMs
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14999773.svg)](https://doi.org/10.5281/zenodo.14999773)

This repository contains code for replicating the experiments in *"Data Service Composition in Cyber-Physical Systems Adopting LLMs"*.


## Prerequisites
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- [OpenAI API Key](https://platform.openai.com)
- [Mistral API Key](https://console.mistral.ai)
- [Anthropic API Key](https://console.anthropic.com)
- [Deepseek API Key](https://platform.deepseek.com)


## Getting Started
Please download the *complete* repository, which includes the data and the experiments results from [Zenodo](https://doi.org/10.5281/zenodo.14999773) and extract the files:

```bash
unzip icws-248.zip
cd icws-248
```

- Create a virtual environment and install the dependencies
```bash
conda create -n pyllm python=3.9
conda activate pyllm
pip install -r requirements.txt
```
- Create a `.env` file in the `src` directory and add the following lines
```
OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>
MISTRAL_API_KEY=<YOUR_MISTRAL_API_KEY>
ANTHROPIC_API_KEY=<YOUR_ANTHROPIC_API_KEY>
DEEPSEEK_API_KEY=<YOUR_DEEPSEEK_API_KEY>
```

### Generating the plots
The code for generating the plots is located in the `experiments` folder. The folder contains the following 3 files:
- `run_experiments_selector.py`: This script reproduces the plots related to the *DS Selector* in the quantitative setting.
- `pipeline_gen_quantitative.py`: This script reproduces the plots related to the *LLM Agent* in the quantitative setting.
- `pipeline_gen_in_action.py`: This script reproduces the plots related to the *LLM Agent* in the in action setting.

The plots are saved in the `plots` folder in the `experiments` directory and are named after the corresponding figure number in the paper.


## Experimental results
The experimental results are stored in the `evaluation` folder. The folder contains a subfolder named after each database in the BIRD benchmark for the quantitative setting. The database name corresponding to the in action setting is `cardboard_production`. Each subfolder contains another subfolder named after each model used in the experiments. 

Each folder contains the following files:
- For the *DS Selector*
  - `evaluation_results_check_ground_truth__no_view__<database>__<enterprise>__<model_name>.csv`: This file contains the results for the *DS Selector* using a specific enterprise (i.e. OpenAI), a specific model (i.e. GPT-4o) on a specific database (e.g. `soccer_2016`).
  - `detailed_results_check_ground_truth__no_view__<database>__<enterprise>__<model_name>.csv`: This file contains the evaluation metric results for the *DS Selector* using a specific enterprise, a specific model on a specific database.
  - `summarized_results_check_ground_truth__no_view__<database>__<enterprise>__<model_name>.csv`: This file contains the summarized evaluation metric results for the *DS Selector* using a specific enterprise, a specific model, on a specific database.
- For the *LLM Agent*
  - `evaluation_results__<database>__<enterprise>__<model_name>__wo_pipeline_view__standard_evidence__tutti.csv`: This file contains the results for the *LLM Agent* using a specific enterprise, a specific model on a specific database, including the generated pipeline and the resulting output table.
  - `metrics_results__valentine__<database>__<enterprise>__<model_name>__wo_pipeline_view__standard_evidence__tutti.csv`: This file contains the evaluation metric results for the *LLM Agent* using a specific enterprise, a specific model on a specific database.
  - `summarized_results__valentine__<database>__<enterprise>__<model_name>__wo_pipeline_view__standard_evidence__tutti.csv`: This file contains the summarized evaluation metric results for the *LLM Agent* using a specific enterprise, a specific model, on a specific database.


The scripts for reproducing the experiments are located in the `src` folder. The folder contains the following 3 files:
- `run_experiments_selector.py`: This script runs the *DS Selector* on the subset of the BIRD benchmark with all the models and stores the results in the `evaluation` folder.
- `pipeline_gen_quantitative.py`: This script runs the *LLM Agent* on the subset of the BIRD benchmark with all the models and stores the results in the `evaluation` folder.
- `pipeline_gen_in_action.py`: This script runs the *LLM Agent* on the `cardboard_production` database (i.e. the placeholder database name for the in action setting) with all the models and stores the results in the `evaluation` folder.

To reproduce the experiments, run the following commands:
```bash
cd src
python run_experiments_selector.py
python pipeline_gen_quantitative.py
python pipeline_gen_in_action.py
```
