from pathlib import Path
import pandas as pd

root_dir = Path.cwd().parent
src_dir = root_dir / 'src'
evaluation_dir = src_dir / 'evaluation'

# define the list of database to consider
database_to_num_tables = [
    ("craftbeer", 2),
    ("trains", 2),
    ("music_tracker", 2),
    ("citeseer", 3),
    ("genes", 3),
    ("human_resources", 3),
    ("cookbook", 4),
    ("computer_student", 4),
    ("software_company", 5),
    ("shipping", 5),
    ("cs_semester", 5),
    ("law_episode", 6),
    ("language_corpus", 6),
    ("movielens", 7),
    ("chicago_crime", 7),
    ("car_retails", 8),
    ("video_games", 8),
    ("retails", 8),
    ("professional_basketball", 9),
    ("student_loan", 10),
    ("book_publishing_company", 11),
    ("olympics", 11),
    ("books", 15),
    ("public_review_platform", 15),
    ("movies_4", 17),
    ("soccer_2016", 21),
    ("mondial_geo", 34),
]

# define the list of models to consider
enterprise_list = ["Openai", "Mistral", "Anthropic", "Deepseek"]
enterprise_to_model_dict = {
    "Openai": "gpt_4o",
    "Mistral": "mistral_large_latest",
    "Anthropic": "claude_3_5_sonnet_latest",
    "Deepseek": "deepseek_chat",
}


# parse the results
parsed_result_list = []
filter_null = False

for database_tuple in database_to_num_tables:
    database_name, num_tables = database_tuple
    for enterprise in enterprise_list:

        result_entry = {}
        
        result_dirpath = evaluation_dir / database_name / enterprise
        if not result_dirpath.exists():
            result_entry = {
                "database_name": database_name,
                "num_tables": num_tables,
                "enterprise": enterprise,
                "model_name": model_name,
                "precision": None,
                "recall": None,
                "acc_cell": None,
                "acc_row": None,
                "equivalent_ratio": None,
                "pipeline_gen_error": None,
                "pipeline_exec_error": None,
                "mean_execution_accuracy": None,
            }
            parsed_result_list.append(result_entry)
            continue

        model_name = enterprise_to_model_dict[enterprise]

        # get the pipeline run log file
        pipeline_run_log_filepath = result_dirpath / f"evaluation_results__{database_name}__{enterprise}__{model_name}__wo_pipeline_view__standard_evidence__tutti.csv"
        if not pipeline_run_log_filepath.exists():
            result_entry = {
                "database_name": database_name,
                "num_tables": num_tables,
                "enterprise": enterprise,
                "model_name": model_name,
                "precision": None,
                "recall": None,
                "acc_cell": None,
                "acc_row": None,
                "equivalent_ratio": None,
                "pipeline_gen_error": 1,
                "pipeline_exec_error": None,
                "mean_execution_accuracy": None,
            }
            parsed_result_list.append(result_entry)
            continue

        pipeline_run_log_df = pd.read_csv(pipeline_run_log_filepath)
        # compute the number of empty values for the pipeline column
        pipeline_gen_error = pipeline_run_log_df["pipeline"].isna().sum()
        # get the indices of the pipeline_gen_error column 
        pipelin_gen_error_index_list = pipeline_run_log_df[pipeline_run_log_df["pipeline"].isna()].index.tolist()
        result_entry["pipeline_gen_error"] = pipeline_gen_error / pipeline_run_log_df.shape[0]
        # compute the number of empty values for the output_json column
        pipeline_exec_error = pipeline_run_log_df["output_json"].isna().sum()
        # compute the indices of the pipeline_gen_error column 
        pipeline_exec_error_index_list = pipeline_run_log_df[pipeline_run_log_df["output_json"].isna()].index.tolist()
        # combine the two indices
        indices_with_errors = set(pipelin_gen_error_index_list + pipeline_exec_error_index_list)
        result_entry["pipeline_exec_error"] = pipeline_exec_error / pipeline_run_log_df.shape[0]

        # get the valentine result
        valentine_result_filepath = result_dirpath / f"metrics_results__valentine__{database_name}__{enterprise}__{model_name}__wo_pipeline_view__standard_evidence__tutti.csv"  
        if not valentine_result_filepath.exists():
            result_entry = {
                "database_name": database_name,
                "num_tables": num_tables,
                "enterprise": enterprise,
                "model_name": model_name,
                "precision": None,
                "recall": None,
                "acc_cell": None,
                "acc_row": None,
                "equivalent_ratio": None,
                "pipeline_gen_error": 1,
                "pipeline_exec_error": None,
                "mean_execution_accuracy": None,
            }
            parsed_result_list.append(result_entry)
            continue
        
        valentine_result_df = pd.read_csv(valentine_result_filepath)
        # filter rows that have zero for all metrics
        if filter_null:
            # filter out the indices with errors usign the indices_with_errors
            valentine_result_df = valentine_result_df[valentine_result_df.index.isin(indices_with_errors) == False]
        # compute the mean of the metrics
        valentine_result_mean = valentine_result_df.mean()
        # convert the mean to a dictionary
        valentine_result_mean_dict = valentine_result_mean.to_dict()
        # add to the result entry
        result_entry.update(valentine_result_mean_dict)
        
        # get the judge result
        judge_result_filepath = result_dirpath / f"metrics_results__llm__{database_name}__{enterprise}__{model_name}__wo_pipeline_view__standard_evidence__tutti.csv"
        judge_result_df = pd.read_csv(judge_result_filepath)
        if filter_null:
            # filter out the indices with errors usign the indices_with_errors
            judge_result_df = judge_result_df[judge_result_df.index.isin(indices_with_errors) == False]
        # count the occurence of EQUIVALENT in the verdict column
        equivalent_count = judge_result_df["verdict"].value_counts().get("EQUIVALENT", 0)
        # divide by the total number of rows
        if judge_result_df.shape[0] == 0:
            equivalent_ratio = 0
        else:
            equivalent_ratio = equivalent_count / judge_result_df.shape[0]
        # add to the result entry
        result_entry["equivalent_ratio"] = equivalent_ratio
        
        # get the execution accuracy results
        execution_accuracy_filepath = result_dirpath / f"metrics_results__execution_accuracy__{database_name}__{enterprise}__{model_name}__wo_pipeline_view__standard_evidence__tutti.csv"
        # read the execution accuracy results
        execution_accuracy_df = pd.read_csv(execution_accuracy_filepath)
        # filter rows based on the indices_with_errors
        if filter_null:
            execution_accuracy_df = execution_accuracy_df[execution_accuracy_df.index.isin(indices_with_errors) == False]
        # compute the mean of the execution_accuracy column
        mean_execution_accuracy = execution_accuracy_df["execution_accuracy"].mean()
        # add to the result entry
        result_entry["mean_execution_accuracy"] = mean_execution_accuracy
        
        # add the database name
        result_entry["database_name"] = database_name
        # add the number of tables
        result_entry["num_tables"] = num_tables
        # add the enterprise name
        result_entry["enterprise"] = enterprise
        # add the model name
        result_entry["model_name"] = model_name
        # add to the parsed result list
        parsed_result_list.append(result_entry)

# create a dataframe from the parsed result list
parsed_result_df = pd.DataFrame(parsed_result_list)

# create a vertical subplot with precision, recall, acc_row
from plotly.subplots import make_subplots
from plotly import graph_objects as go

# define the list of metrics to consider
metrics = [
    "precision",
    "recall",
    "acc_row",
    "equivalent_ratio",
    "pipeline_gen_error",
    "pipeline_exec_error"
]

fig = make_subplots(
    rows=3,
    cols=1
)

enterprise_to_color = {
    "Openai": "#636efa",
    "Mistral": "#ef553b",
    "Anthropic": "#01cc96",
    "Deepseek": "#ab63fa"
}

for i, metric in enumerate(["precision", "recall", "acc_row"]):
    for enterprise in enterprise_list:
        x = []
        y = []
        for database_tuple in database_to_num_tables:
            database_name, _ = database_tuple
            
            x.append(database_name)
            
            # filter the dataframe
            filtered_df = parsed_result_df[
                (parsed_result_df["database_name"] == database_name) &
                (parsed_result_df["enterprise"] == enterprise)
            ]
            # get the metric value
            metric_value = filtered_df[metric].values[0]
            # add to the y list
            y.append(metric_value)
        
        # add the bar for the enterprise
        fig.add_trace(
            go.Bar(
                x=x,
                y=y,
                name=enterprise,
                marker_color=enterprise_to_color[enterprise]
            ),
            row=i+1,
            col=1
        )

    # update the layout
    #fig.update_xaxes(title_text="Database", row=i+1, col=1)
    fig.update_yaxes(title_text=metric, row=i+1, col=1)
    # increase yaxes title font size
    fig.update_yaxes(title_font=dict(size=18), row=i+1, col=1)

# increase gap between bar groups
fig.update_layout(bargap=0.3)

# tilt the x-axis labels
fig.update_xaxes(tickangle=-22)
# increase x-axis font size
fig.update_xaxes(tickfont=dict(size=15))
# increase y-axis font size
fig.update_yaxes(tickfont=dict(size=15))

# set width and height
fig.update_layout(width=1600, height=1000)

# remove margin
fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))

names = set()
fig.for_each_trace(
    lambda trace:
        trace.update(showlegend=False)
        if (trace.name in names) else names.add(trace.name))

# increase legend font size
fig.update_layout(legend_font=dict(size=22))

# put the legend at the top and horizontal
fig.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1
))

# save as pdf
if not Path("plots").exists():
    Path("plots").mkdir()
fig.write_image(f"plots/Figure5.pdf")
#fig.show()


# for each metric, group by database and rank the enterprise based on the metric (higher metric value is better, i.e. rank is 1 for the highest value)

precision_result_list = []
recall_result_list = []
acc_row_result_list = []


for database_name, _ in database_to_num_tables:
    # filter the dataframe
    filtered_df = parsed_result_df[parsed_result_df["database_name"] == database_name]
    sorted_df = filtered_df.sort_values(by="precision", ascending=False)
    sorted_df["precision_rank"] = range(1, sorted_df.shape[0]+1)
    sorted_df["precision_rank"] = sorted_df["precision_rank"].astype(int)
    precision_result_list.append(sorted_df[["database_name", "enterprise", "precision_rank", "num_tables"]])

    sorted_df = filtered_df.sort_values(by="recall", ascending=False)
    sorted_df["recall_rank"] = range(1, sorted_df.shape[0]+1)
    sorted_df["recall_rank"] = sorted_df["recall_rank"].astype(int)
    recall_result_list.append(sorted_df[["database_name", "enterprise", "recall_rank", "num_tables"]])

    sorted_df = filtered_df.sort_values(by="acc_row", ascending=False)
    sorted_df["acc_row_rank"] = range(1, sorted_df.shape[0]+1)
    sorted_df["acc_row_rank"] = sorted_df["acc_row_rank"].astype(int)
    acc_row_result_list.append(sorted_df[["database_name", "enterprise", "acc_row_rank", "num_tables"]])


# create a dataframe from the result lists
precision_rank_df = pd.concat(precision_result_list)
recall_rank_df = pd.concat(recall_result_list)
acc_row_rank_df = pd.concat(acc_row_result_list)

# create a vertical subplot with precision, recall, acc_row ranks
# x axis number of tables, y axis rank average across databases with the same number of tables
# use line plot
from plotly.subplots import make_subplots

enterprise_to_color = {
    "Openai": "#636efa",
    "Mistral": "#ef553b",
    "Anthropic": "#01cc96",
    "Deepseek": "#ab63fa"
}

fig = make_subplots(
    rows=3,
    cols=1,
    vertical_spacing=0.1
)

for i, (rank_df, metric_name) in enumerate(zip([precision_rank_df, recall_rank_df, acc_row_rank_df], ["precision", "recall", "acc_row"])):
    for enterprise in enterprise_list:
        x = []
        y = []
        for num_tables in range(2, 35):
            filtered_df = rank_df[
                (rank_df["enterprise"] == enterprise) &
                (rank_df["num_tables"] == num_tables)
            ]
            if filtered_df.shape[0] == 0:
                continue
            x.append(num_tables)
            y.append(filtered_df[f"{metric_name}_rank"].mean())
        
        # add the line for the enterprise
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines+markers",
                name=enterprise,
                marker=dict(
                    color=enterprise_to_color[enterprise]
                )
            ),
            row=i+1,
            col=1
        )

    # update the layout
    fig.update_xaxes(title_text="Number of tables", row=i+1, col=1)
    fig.update_yaxes(title_text=f"{metric_name} rank", row=i+1, col=1)

# remove margin
fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))

# drop duplicate legend entries
names = set()
fig.for_each_trace(
    lambda trace:
        trace.update(showlegend=False)
        if (trace.name in names) else names.add(trace.name))

# set width and height
fig.update_layout(width=1200, height=1000)

# set x-axis and y-axis title font size
fig.update_xaxes(title_font=dict(size=25))
fig.update_yaxes(title_font=dict(size=25))

# set x-axis and y-axis tick font size
fig.update_xaxes(tickfont=dict(size=20))
fig.update_yaxes(tickfont=dict(size=20))

# increase legend font size
fig.update_layout(legend_font=dict(size=25))

# put the legend at the top and horizontal
fig.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1
))

fig.write_image("plots/Figure6.pdf")
#fig.show()


