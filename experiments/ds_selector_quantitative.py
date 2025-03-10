# %%
from pathlib import Path
import pandas as pd

root_dir = Path.cwd().parent
src_dir = root_dir / 'src'
evaluation_dir = src_dir / 'evaluation'

# %%
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
    ("college_completion", 4),
    ("software_company", 5),
    ("shipping", 5),
    ("cs_semester", 5),
    ("law_episode", 6),
    ("language_corpus", 6),
    ("image_and_language", 6),
    ("movielens", 7),
    ("chicago_crime", 7),
    ("simpson_episodes", 7),
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

# %%
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
                "accuracy": None,
                "precision": None,
                "recall": None
            }
            parsed_result_list.append(result_entry)
            continue

        model_name = enterprise_to_model_dict[enterprise]

        # get the pipeline run log file
        pipeline_run_log_filepath = result_dirpath / f"summarized_results_check_ground_truth__no_view__{database_name}__{enterprise}__{model_name}.csv"
        if not pipeline_run_log_filepath.exists():
            result_entry = {
                "database_name": database_name,
                "num_tables": num_tables,
                "enterprise": enterprise,
                "model_name": model_name,
                "accuracy": None,
                "precision": None,
                "recall": None
            }
            parsed_result_list.append(result_entry)
            continue

        res_df = pd.read_csv(pipeline_run_log_filepath)
        acc_res = float(res_df.iloc[0,0])
        prec_res = float(res_df.iloc[1,0])
        rec_res = float(res_df.iloc[2,0])
        
        result_entry = {
            "database_name": database_name,
            "num_tables": num_tables,
            "enterprise": enterprise,
            "model_name": model_name,
            "accuracy": acc_res,
            "precision": prec_res,
            "recall": rec_res
        }
        parsed_result_list.append(result_entry)
        
# create a dataframe from the parsed result list
parsed_result_df = pd.DataFrame(parsed_result_list)

# %%
# # using plotly go make a barplot of the results
# # x-axis: database_name and y-axis: mean of the metrics (for each metric)

# from plotly import graph_objects as go

# # define the list of metrics to consider
# metrics = [
#     "precision",
#     "recall"
# ]

# for metric in metrics:
#     fig = go.Figure()
#     for enterprise in enterprise_list:
#         x = []
#         y = []
#         for database_tuple in database_to_num_tables:
#             database_name, _ = database_tuple
            
#             x.append(database_name)
            
#             # filter the dataframe
#             filtered_df = parsed_result_df[
#                 (parsed_result_df["database_name"] == database_name) &
#                 (parsed_result_df["enterprise"] == enterprise)
#             ]
#             # get the metric value
#             metric_value = filtered_df[metric].values[0]
#             # add to the y list
#             y.append(metric_value)
        
#         # add the bar for the enterprise
#         fig.add_trace(
#             go.Bar(
#                 x=x,
#                 y=y,
#                 name=enterprise
#             )
#         )
        
#     # update the layout
#     fig.update_layout(
#         #title=f"{metric} DS selector over BIRD databases",
#         xaxis_title="Database",
#         yaxis_title=metric,
#         barmode='group'
#     )

#     # increase gap between bar groups
#     fig.update_layout(bargap=0.3)

#     # tilt the x-axis labels
#     fig.update_layout(xaxis_tickangle=-45)

#     # set width and height
#     fig.update_layout(width=1600, height=600)

#     fig.update_layout(
#         legend=dict(
#             yanchor="bottom",
#             y=0.05,
#             xanchor="right",
#             x=0.11
#         ),
#         font=dict(
#             size=18
#         )
#     )

#     # save as pdf
#     if not Path("plots").exists():
#         Path("plots").mkdir()
#     fig.write_image(f"plots/{metric}_barplot_selector.pdf")
    
#     fig.show()


# %%
# # now for each metric let's do some line plots
# # each line represent a model, x-axis: number of tables, y-axis: mean of the metric with error bars
# # error bars: standard deviation
# from plotly import graph_objects as go


# # define the list of metrics to consider


# for metric in metrics:
#     fig = go.Figure()
#     for enterprise in enterprise_list:
#         model_name = enterprise_to_model_dict[enterprise]
#         x = []
#         y = []
#         error_y = []
#         # filter the dataframe
#         filtered_df = parsed_result_df[
#             (parsed_result_df["enterprise"] == enterprise) &
#             (parsed_result_df["model_name"] == model_name)
#         ]
#         # group by the number of tables
#         grouped_df = filtered_df.groupby("num_tables")
#         for num_tables, group_df in grouped_df:
#             x.append(num_tables)
#             y.append(group_df[metric].mean())
#             error_y.append(group_df[metric].std())

#         # add the line for the model
#         fig.add_trace(
#             go.Scatter(
#                 x=x,
#                 y=y,
#                 name=model_name,
#                 line = dict(width=4)
#             )
#         )

#     # set dtick for x-axis to 2
#     fig.update_layout(
#         xaxis_dtick=2,
#         yaxis=dict(range=[0, 1.03])
#     )

#     # update the layout
#     fig.update_layout(
#         #title=f"{metric} vs num ",
#         xaxis_title="Number of tables",
#         yaxis_title=metric,
#         legend=dict(
#             yanchor="bottom",
#             y=0.05,
#             xanchor="right",
#             x=0.23
#         ),
#         font=dict(
#             size=20
#         )
#     )

#     # set width and height
#     fig.update_layout(width=1600, height=600)
#     # save as pdf
#     fig.write_image(f"results/{metric}_lineplot_selector.pdf")


#     fig.show()

# %%
enterprise_to_color = {
    "Openai": "#636efa",
    "Mistral": "#ef553b",
    "Anthropic": "#01cc96",
    "Deepseek": "#ab63fa"
}

# organize the results in a vertical subplot
from plotly.subplots import make_subplots 
from plotly import graph_objects as go


import plotly.io as pio   
pio.kaleido.scope.mathjax = None

fig = make_subplots(
    rows=2,
    cols=1,
    vertical_spacing=0.1
)

metrics = [
    "precision",
    "recall"
]

for idx, metric in enumerate(metrics):
    for enterprise in enterprise_list:
        model_name = enterprise_to_model_dict[enterprise]
        x = []
        y = []
        error_y = []
        # filter the dataframe
        filtered_df = parsed_result_df[
            (parsed_result_df["enterprise"] == enterprise) &
            (parsed_result_df["model_name"] == model_name)
        ]
        # group by the number of tables
        grouped_df = filtered_df.groupby("num_tables")
        for num_tables, group_df in grouped_df:
            x.append(num_tables)
            y.append(group_df[metric].mean())
            error_y.append(group_df[metric].std())

        # add the line for the model
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                name=enterprise,
                line = dict(width=4, color=enterprise_to_color[enterprise])
            ),
            row=idx+1,
            col=1
        )

    # set dtick for x-axis to 2
    #fig.update_xaxes(dtick=2, row=idx+1, col=1)
    fig.update_yaxes(range=[0, 1.03], row=idx+1, col=1)

    # update the layout
    fig.update_xaxes(title_text="Number of tables", row=idx+1, col=1, title_font=dict(size=35))
    fig.update_yaxes(title_text=metric, row=idx+1, col=1, title_font=dict(size=35))

# set width and height
fig.update_layout(width=1600, height=1200)

# remove duplicated legend entries
names = set()
fig.for_each_trace(
    lambda trace:
        trace.update(showlegend=False)
        if (trace.name in names) else names.add(trace.name))

# remove margin
fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
# increase x-axis font size
fig.update_xaxes(tickfont=dict(size=28))
# increase y-axis font size
fig.update_yaxes(tickfont=dict(size=28))

# increase legend font size
fig.update_layout(legend_font=dict(size=35))

# put legend on the bottom and horizontal
fig.update_layout(
    legend=dict(
        yanchor="bottom",
        y=-0.15,
        xanchor="right",
        x=1,
        orientation="h"
    )
)

if not Path("plots").exists():
    Path("plots").mkdir()
fig.write_image("plots/Figure4.pdf")
#fig.show()


