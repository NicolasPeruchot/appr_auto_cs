from collections import OrderedDict

import numpy as np


def drop_useless(data):
    """Drop useless features."""
    data = data.drop(
        columns=[
            "Listing ID",
            "Listing Name",
            "Host ID",
            "Host Name",
            "Host Since",
            "Is Exact Location",
            "Country",
            "First Review",
            "Last Review",
            "City",
            "Postal Code",
            "Country Code",
        ],
    )
    return data


def incomplete_columns(data):
    total = 0
    list_features = {}
    for col in data.columns:
        miss = data[col].isnull()
        pct = miss.mean() * 100
        total += miss.sum()
        if pct != 0:
            list_features[col] = {"sum": miss.sum(), "pct": round(pct, 2)}

    for x in OrderedDict(sorted(list_features.items(), key=lambda i: i[1]["pct"], reverse=True)):
        print(f"{x} => {list_features[x]['sum']} [{list_features[x]['pct']}%]")

    return list_features


def min_miss_value_corr(data, listfeatures):
    tablefeatures = np.ones((len(listfeatures), len(listfeatures)))

    # calcul des corrélations de missing entre deux features
    index1 = 0
    for q in listfeatures:
        q_nan = data[q].isna()
        nanrows = sum(q_nan)
        index2 = 0
        for p in listfeatures:
            p_nan = data[p].isna()
            if q != p:
                bothmiss = sum(q_nan & p_nan)
                # print ("P("+q+" et "+p+" vaut NaN) =" + str(bothmiss/nanrows))
                tablefeatures[index1, index2] = bothmiss / nanrows
            index2 += 1
        index1 += 1
        min_corr_pct = round(tablefeatures.min() * 100, 2)

    return min_corr_pct
