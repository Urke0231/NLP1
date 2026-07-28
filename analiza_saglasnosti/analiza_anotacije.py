import json
from itertools import combinations

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import cohen_kappa_score, confusion_matrix
from statsmodels.stats.inter_rater import fleiss_kappa
import krippendorff

# Heatmap

def save_heatmap(matrix, title, filename, vmin, vmax):

    plt.figure(figsize=(7, 5))

    sns.heatmap(
        matrix,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        square=True,
        linewidths=0.5
    )

    plt.title(title)
    plt.xlabel("Anotator")
    plt.ylabel("Anotator")

    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=300
    )

    plt.close()

def save_table_as_png(df, title, filename, figsize=(8, 3)):
    fig, ax = plt.subplots(figsize=figsize)

    ax.axis("off")

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)

    plt.title(
        title,
        fontsize=14,
        pad=20
    )

    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# Učitavanje podataka
INPUT_FILE = "poredjenje_anotacija.json"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    dataset = json.load(f)


annotators = sorted(dataset[0]["sentimenti"].keys())


# Procentualna saglasnost
# Cohen kappa
# Scott pi matrice

agreement_matrix = pd.DataFrame(
    np.eye(len(annotators)) * 100,
    index=annotators,
    columns=annotators
)

cohen_matrix = pd.DataFrame(
    np.eye(len(annotators)),
    index=annotators,
    columns=annotators
)

scott_matrix = pd.DataFrame(
    np.eye(len(annotators)),
    index=annotators,
    columns=annotators
)

sum_agreement = 0

for a1, a2 in combinations(annotators, 2):

    labels1 = []
    labels2 = []

    for item in dataset:

        sentiments = item["sentimenti"]

        if a1 in sentiments and a2 in sentiments:
            labels1.append(sentiments[a1])
            labels2.append(sentiments[a2])

    # procentualna saglasnost

    same = sum(
        x == y for x, y in zip(labels1, labels2)
    )

    agreement = round(
        same / len(labels1) * 100,
        2
    )

    agreement_matrix.loc[a1, a2] = agreement
    agreement_matrix.loc[a2, a1] = agreement
    sum_agreement += agreement

    # Cohen kappa

    cohen = cohen_kappa_score(
        labels1,
        labels2
    )

    # Scott pi

    cm = confusion_matrix(
        labels1,
        labels2,
        labels=["negative", "neutral", "positive"]
    )

    po = np.trace(cm) / np.sum(cm)

    row_probs = np.sum(cm, axis=1) / np.sum(cm)
    col_probs = np.sum(cm, axis=0) / np.sum(cm)

    p = (row_probs + col_probs) / 2

    pe = np.sum(p ** 2)

    if pe == 1:
        scott = 1
    else:
        scott = (po - pe) / (1 - pe)

    cohen_matrix.loc[a1, a2] = round(cohen, 4)
    cohen_matrix.loc[a2, a1] = round(cohen, 4)

    scott_matrix.loc[a1, a2] = round(scott, 4)
    scott_matrix.loc[a2, a1] = round(scott, 4)

save_heatmap(
    agreement_matrix,
    "procentualna saglasnost",
    "procentualna_saglasnost.png",
    0,
    100
)

save_heatmap(
    cohen_matrix,
    "Cohen's Kappa",
    "cohen_kappa.png",
    -1,
    1
)

save_heatmap(
    scott_matrix,
    "Scott's Pi",
    "scott_pi.png",
    -1,
    1
)


# srednja vrednost procentualne saglasnosti
number_of_annotators = len(annotators)
group_average = sum_agreement / (number_of_annotators * (number_of_annotators - 1) / 2)

# Fleiss Kappa

ratings = []

for item in dataset:

    counts = [
        0,
        0,
        0
    ]

    for annotator in annotators:

        value = item["sentimenti"][annotator]

        if value == "negative":
            counts[0] += 1

        elif value == "neutral":
            counts[1] += 1

        elif value == "positive":
            counts[2] += 1

    ratings.append(counts)

ratings = np.array(ratings)

fleiss = fleiss_kappa(
    ratings
)

# Krippendorff Alpha

reliability_data = []

for annotator in annotators:

    reliability_data.append(
        [
            item["sentimenti"][annotator]
            for item in dataset
        ]
    )

reliability_data = np.array(
    reliability_data
)

alpha = krippendorff.alpha(
    reliability_data=reliability_data,
    level_of_measurement="nominal"
)

global_results = pd.DataFrame(
    [
        {
            "Mera": "Prosečna procentualna saglasnost (%)",
            "Vrednost": round(group_average, 2)
        },
        {
            "Mera": "Fleiss' kappa",
            "Vrednost": round(fleiss, 4)
        },
        {
            "Mera": "Krippendorff's alpha",
            "Vrednost": round(alpha, 4)
        }
    ]
)


save_table_as_png(
    global_results,
    "Globalne mere saglasnosti",
    "globalne_mere_saglasnosti.png",
    figsize=(7, 2.5)
)

print("\nProcentualna saglasnost:\n")
print(agreement_matrix)

print("\nCohen Kappa:\n")
print(cohen_matrix)

print("\nScott Pi:\n")
print(scott_matrix)

print("\nGlobalne mere:\n")
print(global_results)


# Statistika klasa

class_results = []


for annotator in annotators:

    counts = {
        "negative":0,
        "neutral":0,
        "positive":0
    }

    for item in dataset:

        sentiment = item["sentimenti"].get(
            annotator
        )

        if sentiment in counts:
            counts[sentiment] += 1

    class_results.append(
        {
            "Anotator": annotator,
            "Positive": counts["positive"],
            "Negative": counts["negative"],
            "Neutral": counts["neutral"]
        }
    )

# ukupno

class_results.append(
    {
        "Anotator": "Ukupno",
        "Positive": sum(
            x["Positive"]
            for x in class_results
        ),
        "Negative": sum(
            x["Negative"]
            for x in class_results
        ),
        "Neutral": sum(
            x["Neutral"]
            for x in class_results
        )
    }
)

df_classes = pd.DataFrame(class_results)

save_table_as_png(
    df_classes,
    "Statistika klasa po anotatorima",
    "statistika_po_anotatorima.png",
    figsize=(8, 3)
)
