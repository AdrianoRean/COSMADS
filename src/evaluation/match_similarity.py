from valentine import valentine_match
from valentine.algorithms import Coma
import numpy as np


def match_similarity(df1, df2):

    matcher = Coma(use_instances=True, java_xmx='1024m')
    matches = valentine_match(df1, df2, matcher)
    #print("matches\n",matches)
    #print()

    # construct the ground truth based on the columns of df1
    col_df1 = df1.columns.to_list()
    col_df2 = df2.columns.to_list()
    ground_truth = []
    for i in range(len(col_df1)):
        found = False
        for match in matches:
            if col_df1[i] == match[0][1]:
                ground_truth.append((col_df1[i], match[1][1]))
                found = True
                break
        if not found:
            ground_truth.append((col_df1[i]))
    #print("ground_truth\n",ground_truth)
    #print()


    # accoppiamenti ground truth delle colonne
    metrics = matches.get_metrics(ground_truth)
    '''print("metrics\n",metrics)
    print("precision",metrics["Precision"])
    print("recall",metrics["RecallAtSizeofGroundTruth"])
    print()'''


    # order the two schemas based on the matches
    table2_schema = []
    table1_schema = []
    for match in matches:
        # check time
        if "time" in match[0][1].lower() or "time" in match[1][1].lower():
            continue
        table1_schema.append(match[0][1])
        table2_schema.append(match[1][1])


    # select only the columns that are in the schema
    df1 = df1[table1_schema]
    df2 = df2[table2_schema]

    '''print("df1\n",df1)
    print()
    print("df2\n",df2)
    print()'''


    # devo ordinare le colonne allo stesso modo


    def compute_matrix_similarity(table1, table2):
        matrix1 = table1.to_numpy()
        matrix2 = table2.to_numpy()
        try:
            exact_similarity = np.sum(matrix1 == matrix2) / matrix1.size
        except:
            exact_similarity = 0
        return exact_similarity

    table_similarity = compute_matrix_similarity(df1, df2)
    #print("table_similarity (cell)",table_similarity)


    def check_tables_equal(table1, table2):
        set1 = set([tuple(row) for row in table1.to_numpy()])
        set2 = set([tuple(row) for row in table2.to_numpy()])
        
        intersection = set1.intersection(set2)
        union = set1.union(set2)

        score = len(intersection) / len(union)
        
        return score

    other_precision = check_tables_equal(df1, df2)
    #print("table_similarity (row)",other_precision)

    return metrics["Precision"], metrics["RecallAtSizeofGroundTruth"], table_similarity, other_precision