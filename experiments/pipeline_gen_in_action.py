from pathlib import Path
import pandas as pd

root_dir = Path.cwd().parent
src_dir = root_dir / 'src'
evaluation_dir = src_dir / 'evaluation'

# parse the results
df_list = []

database_name = "cardboard_production"
num_tables = 12

enterprise_list = ["Openai", "Mistral", "Anthropic", "Deepseek", "Github"]
enterprise_to_model_dict = {
    "Openai": "gpt_4o",
    "Mistral": "mistral_large_latest",
    "Anthropic": "claude_3_5_sonnet_latest",
    "Deepseek": "deepseek_chat",
    "Github": "github_copilot"
}


for enterprise in enterprise_list:
    # get the valentine result
    result_dirpath = evaluation_dir / database_name / enterprise
    model_name = enterprise_to_model_dict[enterprise]
    valentine_result_filepath = result_dirpath / f"metrics_results__valentine__{database_name}__{enterprise}__{model_name}__in_action__standard_evidence__tutti.csv"      
    valentine_result_df = pd.read_csv(valentine_result_filepath)
    # map index 0-9 to "q0", 10-19 to "q1", 20-29 to "q2", 30-39 to "q3", 40-49 to "q4"
    valentine_result_df["query_id"] = valentine_result_df["index"].apply(lambda x: f"q{x//10}")
    # average the results for each query (only for recall, precision, acc_row)
    valentine_result_df = valentine_result_df.groupby("query_id").mean()
    valentine_result_df["enterprise"] = enterprise
    valentine_result_df["model"] = model_name
    # restore the query_id as a column instead of index
    valentine_result_df.reset_index(inplace=True)
    # drop index column
    valentine_result_df.drop(columns=["index"], inplace=True)
    df_list.append(valentine_result_df)

# combine the results
result_df = pd.concat(df_list)

# make the same plot using subplots
from plotly.subplots import make_subplots
from plotly import graph_objects as go

import plotly.io as pio   
pio.kaleido.scope.mathjax = None

metric_list = ["precision", "recall", "acc_row"]

enterprise_to_color = {
    "Openai": "#636efa",
    "Mistral": "#ef553b",
    "Anthropic": "#01cc96",
    "Deepseek": "#ab63fa",
    "Github": "goldenrod"
}


fig = make_subplots(rows=3, 
                    cols=1,
                    vertical_spacing=0.1)

for i, metric in enumerate(metric_list):
    x = result_df["query_id"].unique()
    for enterprise in enterprise_list:
        y = []
        model = enterprise_to_model_dict[enterprise]
        for query_id in result_df["query_id"].unique():
            y_enterprise = result_df[(result_df["enterprise"] == enterprise) & (result_df["model"] == model) & (result_df["query_id"] == query_id)][metric]
            y.append(y_enterprise.values[0])
        fig.add_trace(go.Bar(x=x, y=y, name=enterprise, marker_color=enterprise_to_color[enterprise]), row=i+1, col=1)

    # set y-axis title
    fig.update_yaxes(title_text=metric, row=i+1, col=1, title_font=dict(size=32))
    # set x-axis title
    fig.update_xaxes(title_text="query_id", row=i+1, col=1, title_font=dict(size=32))

# increase x-axis font size
fig.update_xaxes(tickfont=dict(size=28))
# increase y-axis font size
fig.update_yaxes(tickfont=dict(size=28))
# increase legend font size
fig.update_layout(legend=dict(
    font=dict(
        size=35
    )
))

# drop duplicate legend
names = set()
fig.for_each_trace(
    lambda trace:
        trace.update(showlegend=False)
        if (trace.name in names) else names.add(trace.name))

# drop margin
fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))

# set width and height
fig.update_layout(width=1600, height=1200)

# put legend top horizontal
fig.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1
))

fig.update_layout(barmode='group')

plots_dir = "plots"
plots_dir.mkdir(exist_ok=True)
fig.write_image(plots_dir / "Figure9.pdf")
#fig.show()


