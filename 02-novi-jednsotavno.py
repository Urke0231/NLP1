import csv
import json
import sys
import traceback
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from joblib import Memory
from scipy.stats import friedmanchisquare, wilcoxon
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfTransformer,
    TfidfVectorizer,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.svm import LinearSVC

from data import load_dataset, present_labels
from preprocessing import lemmatize_batch, stem_batch, strip_named_entities_batch

plt.switch_backend("Agg")

# Putanja do fajla sa podacima (JSON ili JSONL). Ovaj fajl treba da sadrži listu objekata sa poljima "tekst" i "sentiment".
DATA_PATH = Path("anotacije-2026-07-26.json")

# Ovde biramo koje nacine pretprocesiranja zelimo da primenimo.
PREPROCESSING_VARIANTS = [
    "lower",  # Pretvaranje svih karaktera u mala slova
    "lower+stem",  # Pretvaranje svih karaktera u mala slova i stemming
    "lower+lemma",  # Pretvaranje svih karaktera u mala slova i lematizacija
]

# Ovde biramo koji filter zelimo da primenimo na tekstove pre pretprocesiranja.
# Ner filter radi tako sto uklanja sve entitete iz teksta (npr. imena, lokacije, organizacije itd.).
TEXT_FILTER = "ner"

# Ovde biramo koje odlike zelimo da izracunamo.
FEATURE_VARIANTS = [
    "BOW",  # Broj pojavljivanja svake reci u svakom tekstu
    "TF",  # Broj pojavljivanja podeljen brojem reci u tekstu
    "LOG_TF",  # Logaritamski normalizovan broj pojavljivanja
    "IDF",  # Veca vrednost za reci koje se pojavljuju u manje tekstova
    "TF_IDF",  # Proizvod TF i IDF vrednosti
    "TFIDF_1-2",  # TF-IDF za reci i parove uzastopnih reci
    "TFIDF_1-3",  # TF-IDF za reci, bigrame i trigramе
    "CHAR_3-5",  # TF-IDF za karakterске n-grame duzine 3 do 5
    "REC+CHAR",  # Unija recnih i karakterskih TF-IDF odlika
]

# Ovde biramo modele komentarisanjem stavki. "hyperparameters": None pokrece
# model direktno, a recnik hiperparametara ukljucuje unutrasnji GridSearchCV.
MODEL_VARIANTS = {
    "MultinomialNB": {
        "classifier": lambda: MultinomialNB(),
        "hyperparameters": {
            "classifier__alpha": [0.01, 0.1, 0.5, 1.0, 2.0],
        },
    },
    "LogisticRegression": {
        "classifier": lambda: LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            random_state=RANDOM_SEED,
        ),
        "hyperparameters": None,
    },
    "LogisticRegression (GridSearch)": {
        "classifier": lambda: LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            random_state=RANDOM_SEED,
        ),
        "hyperparameters": {
            "classifier__C": [0.1, 1.0, 10.0],
        },
    },
    "LinearSVM": {
        "classifier": lambda: LinearSVC(
            penalty="l2",
            loss="squared_hinge",
            dual=True,
            max_iter=100000,
            random_state=RANDOM_SEED,
        ),
        "hyperparameters": None,
    },
    "LinearSVM (GridSearch)": {
        "classifier": lambda: LinearSVC(
            penalty="l2",
            loss="squared_hinge",
            dual=True,
            max_iter=100000,
            random_state=RANDOM_SEED,
        ),
        "hyperparameters": {
            "classifier__C": [0.1, 1.0, 10.0],
        },
    },
}

MIN_WORD_FREQUENCY = 2
MIN_CHARACTER_FREQUENCY = 3
TOKEN_PATTERN = r"(?u)\S+"

CROSS_VALIDATION_FOLDS = 10
INNER_CROSS_VALIDATION_FOLDS = 2
RANDOM_SEED = 42

# Pipeline ovde kesira odlike posebno za svaki trening fold. Na taj nacin
# validacioni fold ne utice na vokabular koji se uci tokom treniranja.
FEATURE_CACHE_DIR = Path(".cache") / "model_features"
OUTPUT_DIR = Path("rezultati_novi_jednostavno")
LOG_PATH = OUTPUT_DIR / "izvrsavanje.log"


class Tee:
    """Isti izlaz odmah upisuje u terminal i log fajl."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

FEATURE_EXTRACTORS = {
    "BOW": lambda: CountVectorizer(token_pattern=TOKEN_PATTERN),
    "TF": lambda: Pipeline([
        ("counts", CountVectorizer(token_pattern=TOKEN_PATTERN)),
        ("tf", Normalizer(norm="l1")),
    ]),
    "LOG_TF": lambda: TfidfVectorizer(
        token_pattern=TOKEN_PATTERN, use_idf=False,
        sublinear_tf=True, norm=None,
    ),
    "IDF": lambda: TfidfVectorizer(
        token_pattern=TOKEN_PATTERN, binary=True, use_idf=True, norm=None,
    ),
    "TF_IDF": lambda: Pipeline([
        ("counts", CountVectorizer(token_pattern=TOKEN_PATTERN)),
        ("tf", Normalizer(norm="l1")),
        ("idf", TfidfTransformer(use_idf=True, norm=None)),
    ]),
    "TFIDF_1-2": lambda: TfidfVectorizer(
        token_pattern=TOKEN_PATTERN, min_df=MIN_WORD_FREQUENCY,
        ngram_range=(1, 2), sublinear_tf=True,
    ),
    "TFIDF_1-3": lambda: TfidfVectorizer(
        token_pattern=TOKEN_PATTERN, min_df=MIN_CHARACTER_FREQUENCY,
        ngram_range=(1, 3), sublinear_tf=True,
    ),
    "CHAR_3-5": lambda: TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5),
        min_df=MIN_CHARACTER_FREQUENCY, sublinear_tf=True,
    ),
    "REC+CHAR": lambda: FeatureUnion([
        ("word", TfidfVectorizer(
            token_pattern=TOKEN_PATTERN, min_df=MIN_WORD_FREQUENCY,
            ngram_range=(1, 2), sublinear_tf=True,
        )),
        ("char", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5),
            min_df=MIN_CHARACTER_FREQUENCY, sublinear_tf=True,
        )),
    ]),
}


def filter_texts(texts: list[str], filter_name: str) -> list[str]:
    """Filtrira tekstove izabranim metodom."""
    if filter_name == "ner":
        return strip_named_entities_batch(texts, use_gpu=False)

    raise ValueError(f"Nepoznat filter: {filter_name}")


# Preprocessing funkcija koja vraca obradjene tekstove za svaku izabranu varijantu.
def do_preprocessing(texts: Iterable[str]) -> dict[str, list[str]]:
    """Vraca obradjene tekstove za svaku izabranu varijantu."""
    texts = list(texts)
    filtered_texts = filter_texts(texts, TEXT_FILTER)
    lower_texts = [text.lower() for text in filtered_texts]
    preprocessed_texts = {}

    for variant in PREPROCESSING_VARIANTS:
        if variant == "lower":
            preprocessed_texts[variant] = lower_texts
        elif variant == "lower+stem":
            preprocessed_texts[variant] = stem_batch(lower_texts)
        elif variant == "lower+lemma":
            preprocessed_texts[variant] = lemmatize_batch(lower_texts)

    return preprocessed_texts


def calculate_features_for_texts(
    texts: list[str], feature_variant: str
) -> tuple[list[str], object]:
    """Racuna jednu izabranu varijantu odlika za jednu listu tekstova."""
    extractor = FEATURE_EXTRACTORS[feature_variant]()
    feature_matrix = extractor.fit_transform(texts)
    return extractor.get_feature_names_out().tolist(), feature_matrix


def calculate_features(
    preprocessed_texts: dict[str, list[str]],
) -> dict[str, dict[str, tuple[list[str], object]]]:
    """Racuna odlike za svaku varijantu pretprocesiranja."""
    return {
        preprocessing_variant: {
            feature_variant: calculate_features_for_texts(texts, feature_variant)
            for feature_variant in FEATURE_VARIANTS
        }
        for preprocessing_variant, texts in preprocessed_texts.items()
    }


def evaluate_model(
    texts: list[str], labels: list[str], preprocessing_variant: str,
    feature_variant: str, model: str, classifier,
    hyperparameters: dict[str, list[object]] | None, outer_cross_validation,
) -> dict[str, object]:
    """Bira hiperparametre unutrasnjom, a ocenjuje model spoljasnjom CV."""
    FEATURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    inner_cross_validation = StratifiedKFold(
        n_splits=INNER_CROSS_VALIDATION_FOLDS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    start_time = perf_counter()
    accuracy_scores = []
    macro_f1_scores = []
    predictions = [None] * len(labels)
    best_hyperparameters = []
    configuration = f"{preprocessing_variant} + {feature_variant} + {model}"
    number_of_outer_folds = len(outer_cross_validation)
    print(f"\nPokrecem: {configuration}", flush=True)

    for fold, (train_indices, test_indices) in enumerate(
        outer_cross_validation, start=1
    ):
        train_texts = [texts[index] for index in train_indices]
        train_labels = [labels[index] for index in train_indices]
        test_texts = [texts[index] for index in test_indices]
        test_labels = [labels[index] for index in test_indices]

        pipeline = Pipeline(
            [
                ("features", FEATURE_EXTRACTORS[feature_variant]()),
                ("classifier", classifier()),
            ],
            memory=Memory(FEATURE_CACHE_DIR, verbose=0),
        )
        if hyperparameters is None:
            pipeline.fit(train_texts, train_labels)
            fold_predictions = pipeline.predict(test_texts)
            fold_best_hyperparameters = {}
        else:
            search = GridSearchCV(
                pipeline,
                param_grid=hyperparameters,
                cv=inner_cross_validation,
                scoring="f1_macro",
                n_jobs=-1,
            )
            search.fit(train_texts, train_labels)
            fold_predictions = search.predict(test_texts)
            fold_best_hyperparameters = search.best_params_
        fold_accuracy = accuracy_score(test_labels, fold_predictions)
        fold_macro_f1 = f1_score(test_labels, fold_predictions, average="macro")

        accuracy_scores.append(fold_accuracy)
        macro_f1_scores.append(fold_macro_f1)
        best_hyperparameters.append(fold_best_hyperparameters)
        for index, prediction in zip(test_indices, fold_predictions):
            predictions[index] = prediction

        print(
            f"  Fold {fold:>2}/{number_of_outer_folds}: "
            f"accuracy={fold_accuracy:.4f}, macro-F1={fold_macro_f1:.4f}, "
            f"najbolje={fold_best_hyperparameters}",
            flush=True,
        )

    elapsed_seconds = perf_counter() - start_time

    return {
        "model": model,
        "preprocessing_variant": preprocessing_variant,
        "feature_variant": feature_variant,
        "scores": {
            "test_accuracy": np.asarray(accuracy_scores),
            "test_macro_f1": np.asarray(macro_f1_scores),
        },
        "predictions": predictions,
        "inner_cross_validation_folds": (
            INNER_CROSS_VALIDATION_FOLDS if hyperparameters is not None else 0
        ),
        "hyperparameter_scoring": (
            "f1_macro" if hyperparameters is not None else "bez optimizacije"
        ),
        "hyperparameter_grid": hyperparameters,
        "best_hyperparameters": best_hyperparameters,
        "elapsed_seconds": elapsed_seconds,
    }


def save_model_results(
    model: str, results: list[dict[str, object]], labels: list[str],
    label_order: list[str],
) -> None:
    """Cuva sazetak, classification report i matrice konfuzije jednog modela."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    file_model = model.lower().replace("+", "_").replace("-", "_")
    output_path = OUTPUT_DIR / f"rezultati_{file_model}.csv"

    with output_path.open("w", newline="", encoding="utf-8-sig") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["MODEL", model])
        writer.writerow(["SAZETAK REZULTATA"])
        writer.writerow([
            "preprocessing", "features",
            "accuracy", "accuracy_std", "accuracy_min", "accuracy_max",
            "macro_f1", "macro_f1_std", "macro_f1_min", "macro_f1_max",
            "accuracy_po_foldu", "macro_f1_po_foldu",
            "unutrasnji_cv_foldovi", "kriterijum_izbora",
            "mreza_hiperparametara", "najbolji_hiperparametri_po_foldu",
            "vreme_s",
        ])

        for result in results:
            scores = result["scores"]
            accuracy = scores["test_accuracy"]
            macro_f1 = scores["test_macro_f1"]
            writer.writerow([
                result["preprocessing_variant"], result["feature_variant"],
                f"{accuracy.mean():.4f}", f"{accuracy.std():.4f}",
                f"{accuracy.min():.4f}", f"{accuracy.max():.4f}",
                f"{macro_f1.mean():.4f}", f"{macro_f1.std():.4f}",
                f"{macro_f1.min():.4f}", f"{macro_f1.max():.4f}",
                json.dumps(accuracy.tolist()),
                json.dumps(macro_f1.tolist()),
                result["inner_cross_validation_folds"],
                result["hyperparameter_scoring"],
                json.dumps(result["hyperparameter_grid"], ensure_ascii=False),
                json.dumps(result["best_hyperparameters"], ensure_ascii=False),
                f"{float(result['elapsed_seconds']):.2f}",
            ])

        for result in results:
            preprocessing_variant = str(result["preprocessing_variant"])
            feature_variant = str(result["feature_variant"])
            predictions = result["predictions"]
            configuration = f"{preprocessing_variant} + {feature_variant} + {model}"
            report = classification_report(
                labels, predictions, labels=label_order, output_dict=True,
                zero_division=0,
            )
            matrix = confusion_matrix(labels, predictions, labels=label_order)

            writer.writerow([])
            writer.writerow(["KONFIGURACIJA", configuration])
            writer.writerow(["CLASSIFICATION REPORT"])
            writer.writerow(["klasa", "precision", "recall", "f1-score", "support"])
            for report_label in [*label_order, "macro avg", "weighted avg"]:
                values = report[report_label]
                writer.writerow([
                    report_label, f"{values['precision']:.3f}",
                    f"{values['recall']:.3f}", f"{values['f1-score']:.3f}",
                    int(values["support"]),
                ])
            writer.writerow(["accuracy", "", "", f"{report['accuracy']:.3f}", len(labels)])

            writer.writerow([])
            writer.writerow(["MATRICA KONFUZIJE"])
            writer.writerow(["stvarno \\ predvidjeno", *label_order])
            for actual_label, row in zip(label_order, matrix):
                writer.writerow([actual_label, *row.tolist()])

    print(f"Rezultati za {model} sacuvani u: {output_path}")


def configuration_name(result: dict[str, object]) -> str:
    """Vraca citljiv naziv jedne eksperimentalne konfiguracije."""
    return (
        f"{result['preprocessing_variant']} | {result['feature_variant']} | "
        f"{result['model']}"
    )


def holm_correction(p_values: list[float]) -> list[float]:
    """Racuna Holm-korigovane p-vrednosti za visestruka poredjenja."""
    number_of_tests = len(p_values)
    order = np.argsort(p_values)
    corrected = np.empty(number_of_tests, dtype=float)
    previous = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (number_of_tests - rank) * p_values[index])
        previous = max(previous, value)
        corrected[index] = previous
    return corrected.tolist()


def save_statistical_comparison(results: list[dict[str, object]]) -> None:
    """Poredi sve konfiguracije Friedmanovim i uparenim Wilcoxon testovima."""
    score_arrays = [
        np.asarray(result["scores"]["test_macro_f1"])
        for result in results
    ]
    if len(results) >= 3:
        friedman_statistic, friedman_p = friedmanchisquare(*score_arrays)
        with (OUTPUT_DIR / "friedman_sve_konfiguracije.csv").open(
            "w", newline="", encoding="utf-8-sig"
        ) as output_file:
            writer = csv.writer(output_file)
            writer.writerow(["broj_konfiguracija", "statistika", "p"])
            writer.writerow([len(results), friedman_statistic, friedman_p])
        print(
            f"Friedman test svih konfiguracija: "
            f"statistika={friedman_statistic:.4f}, p={friedman_p:.6g}"
        )
    else:
        print("Friedman test zahteva najmanje tri konfiguracije; preskacem ga.")

    comparisons = []
    raw_p_values = []
    for first_index in range(len(results)):
        for second_index in range(first_index + 1, len(results)):
            first_scores = score_arrays[first_index]
            second_scores = score_arrays[second_index]
            p_value = (
                1.0 if np.allclose(first_scores, second_scores)
                else float(wilcoxon(first_scores, second_scores).pvalue)
            )
            comparisons.append((first_index, second_index, p_value))
            raw_p_values.append(p_value)

    corrected_p_values = holm_correction(raw_p_values)
    output_path = OUTPUT_DIR / "wilcoxon_sve_konfiguracije.csv"
    with output_path.open("w", newline="", encoding="utf-8-sig") as output_file:
        writer = csv.writer(output_file)
        writer.writerow([
            "konfiguracija_1", "konfiguracija_2",
            "macro_f1_1", "macro_f1_2", "razlika",
            "p", "p_holm", "znacajno_p_holm_0.05",
        ])
        for comparison, corrected_p in zip(comparisons, corrected_p_values):
            first_index, second_index, p_value = comparison
            first_mean = score_arrays[first_index].mean()
            second_mean = score_arrays[second_index].mean()
            writer.writerow([
                configuration_name(results[first_index]),
                configuration_name(results[second_index]),
                f"{first_mean:.6f}", f"{second_mean:.6f}",
                f"{first_mean - second_mean:.6f}",
                f"{p_value:.6f}", f"{corrected_p:.6f}",
                corrected_p < 0.05,
            ])

    print(f"Wilcoxon poredjenja svih parova sacuvana u: {output_path}")


def fit_final_model(
    result: dict[str, object], texts: list[str], labels: list[str],
):
    """Trenira najbolju konfiguraciju na celom skupu radi tumacenja odlika."""
    model_configuration = MODEL_VARIANTS[str(result["model"])]
    pipeline = Pipeline([
        ("features", FEATURE_EXTRACTORS[str(result["feature_variant"])]()),
        ("classifier", model_configuration["classifier"]()),
    ])
    hyperparameters = model_configuration["hyperparameters"]
    if hyperparameters is None:
        return pipeline.fit(texts, labels)

    inner_cross_validation = StratifiedKFold(
        n_splits=INNER_CROSS_VALIDATION_FOLDS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )
    search = GridSearchCV(
        pipeline, hyperparameters, cv=inner_cross_validation,
        scoring="f1_macro", n_jobs=-1,
    )
    search.fit(texts, labels)
    return search.best_estimator_


def save_informative_features(
    best_result: dict[str, object], preprocessed_texts: dict[str, list[str]],
    labels: list[str],
) -> None:
    """Cuva 20 najinformativnijih odlika za svaku klasu najboljeg modela."""
    texts = preprocessed_texts[str(best_result["preprocessing_variant"])]
    pipeline = fit_final_model(best_result, texts, labels)
    extractor = pipeline.named_steps["features"]
    classifier = pipeline.named_steps["classifier"]
    feature_names = np.asarray(extractor.get_feature_names_out())
    classes = np.asarray(classifier.classes_)

    if hasattr(classifier, "feature_log_prob_"):
        class_values = classifier.feature_log_prob_
        scores = class_values - (
            (class_values.sum(axis=0, keepdims=True) - class_values)
            / max(1, len(classes) - 1)
        )
        value_name = "relativna_log_verovatnoca"
    elif hasattr(classifier, "coef_"):
        scores = classifier.coef_
        if scores.shape[0] == 1 and len(classes) == 2:
            scores = np.vstack([-scores[0], scores[0]])
        value_name = "koeficijent"
    else:
        print("Najbolji model nema vrednosti za tumacenje odlika.")
        return

    output_path = OUTPUT_DIR / "najinformativnije_odlike.csv"
    with output_path.open("w", newline="", encoding="utf-8-sig") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["klasa", "rang", "odlika", value_name])
        for class_index, class_name in enumerate(classes):
            top_indices = np.argsort(scores[class_index])[-20:][::-1]
            for rank, feature_index in enumerate(top_indices, start=1):
                writer.writerow([
                    class_name, rank, feature_names[feature_index],
                    f"{scores[class_index, feature_index]:.6f}",
                ])
    print(f"Najinformativnije odlike sacuvane u: {output_path}")


def save_final_analysis(
    results: list[dict[str, object]], preprocessed_texts: dict[str, list[str]],
    labels: list[str], label_order: list[str],
) -> None:
    """Pravi zavrsnu analizu nakon evaluacije svih konfiguracija."""
    ranked_results = sorted(
        results,
        key=lambda result: -np.mean(result["scores"]["test_macro_f1"]),
    )
    top_results = ranked_results[:10]
    best_result = top_results[0]

    top_rows = []
    for rank, result in enumerate(top_results, start=1):
        accuracy = np.asarray(result["scores"]["test_accuracy"])
        macro_f1 = np.asarray(result["scores"]["test_macro_f1"])
        top_rows.append([
            rank, configuration_name(result), result["model"],
            result["preprocessing_variant"], result["feature_variant"],
            macro_f1.mean(), macro_f1.std(), accuracy.mean(),
            result["elapsed_seconds"],
        ])

    top_path = OUTPUT_DIR / "finalna_top10.csv"
    with top_path.open("w", newline="", encoding="utf-8-sig") as output_file:
        writer = csv.writer(output_file)
        writer.writerow([
            "rang", "konfiguracija", "model", "preprocessing", "features",
            "macro_f1", "macro_f1_std", "accuracy", "vreme_s",
        ])
        for row in top_rows:
            writer.writerow([
                *row[:5], f"{row[5]:.4f}", f"{row[6]:.4f}",
                f"{row[7]:.4f}", f"{float(row[8]):.2f}",
            ])

    print("\n=== TOP 10 KONFIGURACIJA ===")
    for row in top_rows:
        print(
            f"{row[0]:>2}. {row[1]}: macro-F1={row[5]:.4f} "
            f"(std={row[6]:.4f}), accuracy={row[7]:.4f}"
        )

    plot_rows = top_rows[::-1]
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(
        [row[1] for row in plot_rows], [row[5] for row in plot_rows],
        xerr=[row[6] for row in plot_rows], color="#397367",
        error_kw={"ecolor": "#222222", "capsize": 3},
    )
    ax.set_xlabel(f"macro-F1 ({CROSS_VALIDATION_FOLDS}-slojna CV)")
    ax.set_title("Top 10 konfiguracija")
    ax.set_xlim(left=max(0, min(row[5] for row in plot_rows) - 0.15))
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "finalna_top10.png", dpi=150)
    plt.close(fig)

    matrix = confusion_matrix(
        labels, best_result["predictions"], labels=label_order,
    )
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(label_order)), labels=label_order, rotation=30)
    ax.set_yticks(range(len(label_order)), labels=label_order)
    ax.set_xlabel("predvidjeno")
    ax.set_ylabel("stvarno")
    ax.set_title(f"Matrica konfuzije\n{configuration_name(best_result)}")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            ax.text(
                column_index, row_index, matrix[row_index, column_index],
                ha="center", va="center",
            )
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "matrica_konfuzije_najbolje.png", dpi=150)
    plt.close(fig)

    save_informative_features(best_result, preprocessed_texts, labels)
    save_statistical_comparison(ranked_results)
    print(f"Finalna top 10 tabela sacuvana u: {top_path}")
    print(f"Najbolja konfiguracija: {configuration_name(best_result)}")


def main() -> None:
    dataset = load_dataset(DATA_PATH)

    original_texts = dataset["tekst"].tolist()
    preprocessed_texts = do_preprocessing(original_texts)

    print(f"Ucitano tekstova: {len(original_texts)}")

    labels = dataset["sentiment"].tolist()
    label_order = present_labels(dataset)
    outer_cross_validation = list(StratifiedKFold(
        n_splits=CROSS_VALIDATION_FOLDS,
        shuffle=True,
        random_state=RANDOM_SEED,
    ).split(original_texts, labels))
    results_by_model = {model: [] for model in MODEL_VARIANTS}
    for preprocessing_variant in PREPROCESSING_VARIANTS:
        for feature_variant in FEATURE_VARIANTS:
            for model_variant, model_configuration in MODEL_VARIANTS.items():
                result = evaluate_model(
                    preprocessed_texts[preprocessing_variant], labels,
                    preprocessing_variant, feature_variant, model_variant,
                    model_configuration["classifier"],
                    model_configuration["hyperparameters"],
                    outer_cross_validation,
                )

                results_by_model[model_variant].append(result)

    for model_variant, results in results_by_model.items():
        save_model_results(model_variant, results, labels, label_order)

    all_results = [
        result
        for model_results in results_by_model.values()
        for result in model_results
    ]
    save_final_analysis(all_results, preprocessed_texts, labels, label_order)


def run_with_log() -> None:
    """Pokrece eksperiment uz paralelni ispis u terminal i log fajl."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    terminal_stdout = sys.stdout
    terminal_stderr = sys.stderr
    exit_code = 0

    with LOG_PATH.open("w", encoding="utf-8", buffering=1) as log_file:
        sys.stdout = Tee(terminal_stdout, log_file)
        sys.stderr = Tee(terminal_stderr, log_file)
        try:
            print(
                "Pocetak izvrsavanja: "
                f"{datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')}"
            )
            main()
            print(
                "Kraj izvrsavanja: "
                f"{datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')}"
            )
        except KeyboardInterrupt:
            traceback.print_exc()
            exit_code = 130
        except Exception:  # noqa: BLE001 - log mora da zabelezi svaki neocekivani pad
            traceback.print_exc()
            exit_code = 1
        finally:
            sys.stdout = terminal_stdout
            sys.stderr = terminal_stderr

    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    run_with_log()