table = """\multicolumn{1}{r}{No.} & GNN Core & Gating & TUF & MLP \newline Asym & SAGE \newline Asym & Dev \newline Accuracy & Name\\\hline\hline
        & Edge-Core & \yes & \no & \no & \no & 60.5 & base0 new\\\hline
        & Edge-Core & \no & \no & \no & \no & 52.7 & new no gate\\\hline
        & Edge-Core & \yes & \yes & \no & \no & 61.5 & new tuf\\\hline
        & Edge-Core & \no & \yes & \no & \no & 64.1 & new tuf no gate\\\hline

        & Edge-Core & \no & \no & \yes & \no & 58.2 & newnew no gate\\\hline
        & Edge-Core & \yes & \no & \yes & \no & 60.3 & newnew\\\hline
        & Edge-Core & \yes & \no & \no & \yes & - & new sage\\\hline
        & Edge-Core & \no & \no & \no & \yes & - & new sage no gate\\\hline\hline

        
        & Switch-Core & \yes & \no & \no & \no & 58.9 & base0 linear2\\\hline
        & Switch-Core & \yes & \no & \yes & \no & 58 & newnew switch\\\hline
        & Switch-Core & \no & \no & \yes & \no & todo & -\\\hline\hline

        & SAGE-Core & \yes & \no & \no & \yes & - & realsage\\\hline
        & SAGE-Core & \yes & \no & \yes & \no & 59.6 & base0 edge\\\hline
        & SAGE-Core & \no & \no & \no & \yes & - & realsage nogate\\\hline
        & SAGE-Core & \no & \no & \no & \no & todo & -\\\hline"""

table = table.replace("\\\hline\hline", "\\\hline")
table = table.replace("\n        ", "")
table = table.replace("\multicolumn{1}{r}{No.} & ", "")
table = table.replace(" \newline", "")
table = table.replace("\yes", "yes")
table = table.replace("\no", "no")
table = table.replace("todo", "-")

rows = table.split("\\\hline")[:-1]

for i in range(len(rows)):
    if rows[i][0] == "\n":
        rows[i] = rows[i][1:].strip()
    if rows[i][0] == "&":
        rows[i] = rows[i][1:].strip()

cells = [row.split("&") for row in rows]
cells = [[cell.strip() for cell in row] for row in cells]

import pandas as pd

df = pd.DataFrame(cells)
# new_header = df.iloc[0] #grab the first row for the header
# df = df[1:] #take the data less the header row
# df.columns = new_header #set the header row as the df header
# df.reset_index(drop=True)
df.rename(columns=df.iloc[0], inplace = True)
df.drop([0], inplace=True)
df.reset_index(drop=True, inplace=True)

print(df)
print("\n-------------------------------\n+++++++++++++++++++++++++++++++\n-------------------------------\n")


def get_dependent_column_names(independent):
    dependents = list(df.columns)
    removes = [independent, "Dev Accuracy", "Name"]
    [dependents.remove(rem) for rem in removes]
    return dependents


def get_dependents(independent):
    dependents = df[get_dependent_column_names(independent)]
    return dependents


def get_fair_tests(column: str, print_tests=True):
    dependents = get_dependents(column)

    all_duplicate_groups = []
    duplicate_group = set()
    all_duplicates = set()
    for i1, row1 in dependents.iterrows():
        for i2, row2 in dependents.iterrows():
            if i2 <= i1 or i1 in all_duplicates or i2 in all_duplicates:
                continue
            if tuple(row1) == tuple(row2):
                duplicate_group.add(i1)
                duplicate_group.add(i2)
                # print("found dup rows:", row1, row2, "i=", (i1, i2))

        if len(duplicate_group) > 0:
            all_duplicate_groups.append(duplicate_group)
            all_duplicates.update(duplicate_group)
            duplicate_group = set()

    all_duplicate_groups = [sorted(list(group)) for group in all_duplicate_groups]

    if print_tests:
        print("\n------------------------------")
    for group in all_duplicate_groups:
        independents = df[column].iloc[group]
        accs = df["Dev Accuracy"].iloc[group]
        dependents_group = dependents.iloc[group]
        fair_test = pd.concat([independents, accs, dependents_group], axis=1)
        fair_test.sort_values("Dev Accuracy", inplace=True, ascending=False)
        if print_tests:
            print(fair_test.to_markdown(), "\n------------------------------\n")

    leftover_indices = set(df.index) - set(all_duplicates)
    leftovers = df.iloc[sorted(list(leftover_indices))]
    leftovers = leftovers.sort_values(column, ascending=False)

    if print_tests:
        print("leftovers:\n", leftovers.to_markdown())
    return all_duplicate_groups


def plot_deltas(column, primary_value):
    all_duplicate_groups = get_fair_tests(column, print_tests=False)
    dependents = get_dependents(column)
    all_independent_values = set()
    for group in all_duplicate_groups:
        for row_num in group:
            ind_val = df.iloc[group].at[row_num, column]
            all_independent_values.add(ind_val)
    all_independent_values = [primary_value] + list(all_independent_values - {primary_value})
    deltas_table = pd.DataFrame(columns=all_independent_values + get_dependent_column_names(column))

    for i, group in enumerate(all_duplicate_groups):
        # new row. start with empty deltas
        deltas_table.loc[i] = ["-"] * len(all_independent_values) + list(dependents.iloc[group[0]])
        # print("delts:", deltas_table)
        primary_index = df.iloc[group].index[df.iloc[group][column] == primary_value].tolist()[0]
        primary_acc = df.at[primary_index, "Dev Accuracy"]
        for row_num in group:
            dup_acc = df.at[row_num, "Dev Accuracy"]
            if row_num == primary_index:
                delta = primary_acc
            else:
                if primary_acc.replace('.','1').isdigit() and dup_acc.replace('.','1').isdigit():
                    delta = round(float(dup_acc) - float(primary_acc), 2)
                else:
                    delta = dup_acc
            row_independent_value = df.at[row_num, column]
            # print("row ind val:", row_independent_value)
            deltas_table[row_independent_value][i] = delta

    # sort by number of non-empty cells
    num_empties = [len(deltas_table[deltas_table[iv] == "-"]) for iv in deltas_table.columns]
    non_primary_values = list({*all_independent_values} - {primary_value})
    si = sorted(list(range(0, len(non_primary_values))), key=lambda i: num_empties[i+1])
    sorted_columns = [primary_value] + [non_primary_values[so] for so in si] + get_dependent_column_names(column)
    deltas_table = deltas_table.reindex(sorted_columns, axis=1)
    renames = {npv: "Δ " + npv for npv in non_primary_values}
    renames.update({"yes": column, "no": "no " + column})
    deltas_table = deltas_table.rename(columns=renames)  # old method

    print(column, "delts tab:\n", deltas_table.to_markdown())


# get_fair_tests("GNN Core")
# get_fair_tests("Gating")
# get_fair_tests("TUF")
# get_fair_tests("SAGE Asym")
# get_fair_tests("MLP Asym")


plot_deltas("GNN Core", "Edge-Core")
plot_deltas("TUF", "no")
# plot_deltas("Gating", "yes")
plot_deltas("MLP Asym", "no")



