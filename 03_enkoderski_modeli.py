"""
Faza 3b - testiranje efikasnosti Ollama modela za multilingvalne recenicne
vektore ("paraphrase-multilingual"), koji radi lokalno na localhost:11434.

Ovaj model se NE fino podesava - on samo pretvara tekst u fiksni vektor
(embedding). Njegova efikasnost meri se posredno: iznad zamrznutih vektora
treniraju se jednostavni klasifikatori (logisticka regresija, linearni i
RBF SVM) i poredi se njihov skor sa osnovnim TF-IDF modelima.

Protokol evaluacije:
  - Ne izdvaja se poseban test skup.
  - Ceo skup se ocenjuje stratifikovanom 10-slojnom unakrsnom validacijom.
  - Svaki primer se tacno jednom nalazi u validacionom foldu.
  - Matrica konfuzije i klasifikacioni izvestaj racunaju se iz objedinjenih
    out-of-fold predikcija svih deset foldova.
  - Koriste se fiksni hiperparametri da se izbegne veoma sporo ugnjezdeno
    podesavanje SVM modela pomocu GridSearchCV.

Pokretanje:
  1) Pokrenuti Ollama servis:
       ollama serve
  2) Uveriti se da je model povucen:
       ollama pull paraphrase-multilingual
  3) Pokrenuti skriptu:
       python 03_enkoderski_modeli_10fold_cv.py

Nema parametara iz komandne linije - sve se podesava u bloku KONFIGURACIJA.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

from data import load_dataset, present_labels
from preprocessing import basic_clean

# =============================================================================
#                              K O N F I G U R A C I J A
# =============================================================================

# --- OBAVEZNO ----------------------------------------------------------------

PUTANJA_PODACI = "./anotacije-2026-07-26.json"
IZLAZNI_DIREKTORIJUM = "rezultati_enkoder_konacno_revizija"

# --- OLLAMA ------------------------------------------------------------------

OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "paraphrase-multilingual"
OLLAMA_BATCH = 16   # koliko tekstova ide u jedan HTTP zahtev (/api/embed)
OLLAMA_NITI = 4     # broj paralelnih HTTP zahteva ka Ollama-i

# Vektori se kesiraju u .cache/, pa ponovno pokretanje ne salje iste tekstove
# ponovo Ollama-i. Postavite na True da se kes zanemari.
IGNORISI_KES = False

# --- STA SE POREDI -----------------------------------------------------------

MODELI = [
    "Vecinski",
    "LogRegresija",
    "LinearSVM",
    "RBF-SVM",
]

# --- PROTOKOL EVALUACIJE -----------------------------------------------------

BROJ_VALIDACIONIH_FOLDOVA = 10
SLUCAJNO_SEME = 42

# Fiksni hiperparametri uklanjaju spori GridSearchCV unutar svakog CV folda.
LOGREG_C = 1.0
LINEAR_SVM_C = 1.0
RBF_SVM_C = 1.0
RBF_SVM_GAMMA = "scale"

# LibSVM koristi memorijski kes. Smanjite vrednost ako racunar nema dovoljno RAM-a.
RBF_SVM_CACHE_MB = 2048

# =============================================================================
#                    kraj konfiguracije - ispod ne treba menjati
# =============================================================================

CACHE = Path(__file__).parent / ".cache"
CACHE.mkdir(exist_ok=True)


def _http_post(path: str, payload: dict, timeout: int = 120) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{OLLAMA_URL}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def _embed_batch(texts: List[str]) -> List[List[float]]:
    """Vraca embeddinge za jedan paket tekstova uz nekoliko ponovnih pokusaja."""
    for attempt in range(4):
        try:
            output = _http_post(
                "/api/embed",
                {"model": OLLAMA_MODEL, "input": texts},
            )
            return output["embeddings"]
        except (urllib.error.HTTPError, KeyError):
            break
        except urllib.error.URLError:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)

    embeddings = []
    for text in texts:
        for attempt in range(4):
            try:
                output = _http_post(
                    "/api/embeddings",
                    {"model": OLLAMA_MODEL, "prompt": text},
                )
                embeddings.append(output["embedding"])
                break
            except urllib.error.URLError:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
    return embeddings


def get_embeddings(texts: List[str]) -> np.ndarray:
    """Vraca i kesira matricu Ollama vektora oblika broj_tekstova x dimenzija."""
    key = hashlib.md5(
        (OLLAMA_MODEL + "|" + json.dumps(texts, ensure_ascii=False)).encode("utf-8")
    ).hexdigest()
    cache_file = CACHE / f"ollama_emb_{key}.npy"

    if cache_file.exists() and not IGNORISI_KES:
        print(f"Ucitavam kesirane vektore: {cache_file.name}")
        return np.load(cache_file)

    try:
        urllib.request.urlopen(OLLAMA_URL, timeout=5)
    except urllib.error.URLError:
        sys.exit(
            f"GRESKA: Ollama nije dostupna na {OLLAMA_URL}.\n"
            f"  Pokrenite je sa 'ollama serve' i uverite se da je model "
            f"povucen: 'ollama pull {OLLAMA_MODEL}'"
        )

    print(
        f"Racunam vektore preko Ollama ({OLLAMA_MODEL}) za {len(texts)} tekstova..."
    )
    chunks = [
        texts[i:i + OLLAMA_BATCH]
        for i in range(0, len(texts), OLLAMA_BATCH)
    ]
    results = [None] * len(chunks)
    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=OLLAMA_NITI) as executor:
        futures = {
            executor.submit(_embed_batch, chunk): index
            for index, chunk in enumerate(chunks)
        }
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            results[futures[future]] = future.result()
            completed += 1
            print(
                f"  {completed}/{len(chunks)} paketa obradjeno",
                end="\r",
                flush=True,
            )
    print()

    embeddings = np.array(
        [vector for chunk in results for vector in chunk],
        dtype=np.float32,
    )
    print(
        f"Gotovo za {time.time() - start_time:.1f}s. "
        f"Dimenzija vektora: {embeddings.shape[1]}"
    )
    np.save(cache_file, embeddings)
    return embeddings


def _model_defs():
    return {
        "Vecinski": (
            DummyClassifier(strategy="most_frequent"),
            {},
            False,
        ),
        "LogRegresija": (
            LogisticRegression(
                C=LOGREG_C,
                max_iter=3000,
                class_weight="balanced",
                random_state=SLUCAJNO_SEME,
            ),
            {"C": LOGREG_C},
            True,
        ),
        "LinearSVM": (
            LinearSVC(
                C=LINEAR_SVM_C,
                class_weight="balanced",
                random_state=SLUCAJNO_SEME,
                dual=False,
                max_iter=20000,
            ),
            {"C": LINEAR_SVM_C},
            True,
        ),
        "RBF-SVM": (
            SVC(
                C=RBF_SVM_C,
                gamma=RBF_SVM_GAMMA,
                kernel="rbf",
                class_weight="balanced",
                cache_size=RBF_SVM_CACHE_MB,
            ),
            {"C": RBF_SVM_C, "gamma": RBF_SVM_GAMMA},
            True,
        ),
    }


def evaluate_model(mname: str, clf, hyperparameters: dict, use_scaler: bool, X: np.ndarray, y: np.ndarray, labels: List[str]):
    """Ocenjuje model pomocu 10-fold CV i vraca out-of-fold predikcije."""
    cv = StratifiedKFold(
        n_splits=BROJ_VALIDACIONIH_FOLDOVA,
        shuffle=True,
        random_state=SLUCAJNO_SEME,
    )

    predictions_cv = np.empty_like(y)
    fold_scores = []
    fold_accuracies = []
    start_time = time.time()

    for fold_number, (train_idx, validation_idx) in enumerate(cv.split(X, y), start=1):
        steps = [("clf", clone(clf))]
        if use_scaler:
            steps.insert(0, ("scale", StandardScaler()))
        pipeline = Pipeline(steps)

        pipeline.fit(X[train_idx], y[train_idx])
        fold_predictions = pipeline.predict(X[validation_idx])
        predictions_cv[validation_idx] = fold_predictions

        fold_score = f1_score(
            y[validation_idx],
            fold_predictions,
            average="macro",
            labels=labels,
            zero_division=0,
        )
        fold_accuracy = float(
            np.mean(fold_predictions == y[validation_idx])
        )
        fold_scores.append(fold_score)
        fold_accuracies.append(fold_accuracy)

        print(
            f"  {mname:<15} fold {fold_number:>2}/"
            f"{BROJ_VALIDACIONIH_FOLDOVA}: "
            f"macro-F1 = {fold_score:.4f}"
        )

    fold_scores_array = np.asarray(fold_scores, dtype=float)
    fold_accuracies_array = np.asarray(fold_accuracies, dtype=float)

    row = {
        "model": mname,
        "cv_macroF1_sr": fold_scores_array.mean(),
        "cv_macroF1_std": fold_scores_array.std(),
        "cv_tacnost_sr": fold_accuracies_array.mean(),
        "cv_tacnost_std": fold_accuracies_array.std(),
        "cv_macroF1_oof": f1_score(
            y,
            predictions_cv,
            average="macro",
            labels=labels,
            zero_division=0,
        ),
        "cv_tacnost_oof": float(np.mean(predictions_cv == y)),
        "hiperparametri": json.dumps(hyperparameters),
        "fold_skorovi": fold_scores_array.tolist(),
        "fold_tacnosti": fold_accuracies_array.tolist(),
        "vreme_s": round(time.time() - start_time, 1),
    }
    return row, predictions_cv


def main():
    data_path = Path(PUTANJA_PODACI)
    if not data_path.exists():
        sys.exit(
            f"GRESKA: ne postoji fajl sa podacima: {data_path.resolve()}\n"
            f"Ispravite PUTANJA_PODACI na vrhu skripte."
        )

    output_directory = Path(IZLAZNI_DIREKTORIJUM)
    output_directory.mkdir(parents=True, exist_ok=True)

    dataframe = load_dataset(data_path)
    labels = present_labels(dataframe)
    texts = [basic_clean(text) for text in dataframe["tekst"]]
    y = dataframe["sentiment"].to_numpy()

    minimum_class_size = int(dataframe["sentiment"].value_counts().min())
    if minimum_class_size < BROJ_VALIDACIONIH_FOLDOVA:
        sys.exit(
            "GRESKA: svaka klasa mora imati najmanje "
            f"{BROJ_VALIDACIONIH_FOLDOVA} primera za stratifikovanu "
            f"{BROJ_VALIDACIONIH_FOLDOVA}-fold validaciju. "
            f"Najmanja klasa trenutno ima {minimum_class_size} primera."
        )

    print(f"Ucitano {len(dataframe)} primera, klase: {labels}")
    print(
        f"Evaluacija: ceo skup, stratifikovana "
        f"{BROJ_VALIDACIONIH_FOLDOVA}-fold cross-validacija, bez test skupa."
    )

    X = get_embeddings(texts)

    models = {
        name: definition
        for name, definition in _model_defs().items()
        if name in MODELI
    }
    if not models:
        sys.exit("GRESKA: prazna lista modela. Proverite konfiguraciju.")

    rows = []
    cv_prediction_store = {}
    start_time = time.time()

    for model_name, (classifier, hyperparameters, use_scaler) in models.items():
        row, predictions_cv = evaluate_model(
            model_name,
            classifier,
            hyperparameters,
            use_scaler,
            X,
            y,
            labels,
        )
        rows.append(row)
        cv_prediction_store[model_name] = predictions_cv

        print(
            f"{model_name:<15} CV macro-F1 = {row['cv_macroF1_sr']:.4f} "
            f"+/- {row['cv_macroF1_std']:.4f} | "
            f"OOF macro-F1 = {row['cv_macroF1_oof']:.4f} "
            f"({row['vreme_s']:.1f}s)"
        )

    results = pd.DataFrame(rows).sort_values(
        "cv_macroF1_sr",
        ascending=False,
    )
    results.drop(columns=["fold_skorovi", "fold_tacnosti"]).to_csv(
        output_directory / "rezultati_enkoderi.csv",
        index=False,
        encoding="utf-8",
    )

    (output_directory / "enkoderi_fold_skorovi.json").write_text(
        json.dumps(
            {
                row["model"]: {
                    "macro_f1": row["fold_skorovi"],
                    "tacnost": row["fold_tacnosti"],
                }
                for row in rows
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nUkupno vreme: {(time.time() - start_time) / 60:.1f} min")
    print(f"\n=== REZULTATI (Ollama model: {OLLAMA_MODEL}) ===")
    print(
        results[
            [
                "model",
                "cv_macroF1_sr",
                "cv_macroF1_std",
                "cv_macroF1_oof",
                "cv_tacnost_oof",
            ]
        ].to_string(index=False)
    )

    _analiza_najboljeg(
        results,
        cv_prediction_store,
        y,
        labels,
        output_directory,
    )
    _statisticko_poredjenje(rows, output_directory)
    _grafikon(results, output_directory)


def _analiza_najboljeg(res, cv_prediction_store, y, labels, outdir: Path):
    best = res.iloc[0]
    predictions_cv = cv_prediction_store[best.model]

    print(f"\n=== NAJBOLJI MODEL: {best.model} ===")
    print(
        classification_report(
            y,
            predictions_cv,
            labels=labels,
            digits=3,
            zero_division=0,
        )
    )

    figure, axes = plt.subplots(figsize=(6, 5))
    matrix = confusion_matrix(y, predictions_cv, labels=labels)
    ConfusionMatrixDisplay(
        matrix,
        display_labels=labels,
    ).plot(
        ax=axes,
        cmap="Blues",
        colorbar=False,
        values_format="d",
    )
    axes.set_title(
        f"Matrica konfuzije "
        f"({BROJ_VALIDACIONIH_FOLDOVA}-fold CV, OOF)\n"
        f"{OLLAMA_MODEL} + {best.model}"
    )
    axes.set_xlabel("predvidjeno")
    axes.set_ylabel("stvarno")
    figure.tight_layout()
    figure.savefig(
        outdir / "enkoderi_matrica_konfuzije_cv.png",
        dpi=150,
    )
    plt.close(figure)


def _statisticko_poredjenje(rows, outdir: Path):
    """Poredi modele uparenim Wilcoxonovim testom nad istim CV foldovima."""
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        print("(za statisticko poredjenje: pip install scipy)")
        return

    comparable_rows = [row for row in rows if row["model"] != "Vecinski"]
    if len(comparable_rows) < 2:
        return

    comparable_rows = sorted(
        comparable_rows,
        key=lambda row: -row["cv_macroF1_sr"],
    )
    base = comparable_rows[0]
    comparison_rows = []

    for row in comparable_rows[1:]:
        base_scores = np.asarray(base["fold_skorovi"])
        compared_scores = np.asarray(row["fold_skorovi"])
        p_value = (
            1.0
            if np.allclose(base_scores, compared_scores)
            else wilcoxon(base_scores, compared_scores).pvalue
        )
        comparison_rows.append(
            {
                "model": row["model"],
                "cv_macroF1": round(row["cv_macroF1_sr"], 4),
                "razlika": round(
                    base["cv_macroF1_sr"] - row["cv_macroF1_sr"],
                    4,
                ),
                "p_vrednost": round(p_value, 4),
                "znacajno_p<0.05": p_value < 0.05,
            }
        )

    comparison_dataframe = pd.DataFrame(comparison_rows)
    comparison_dataframe.to_csv(
        outdir / "enkoderi_statisticko_poredjenje.csv",
        index=False,
        encoding="utf-8",
    )
    print(
        f"\n=== POREDJENJE SA NAJBOLJIM MODELOM "
        f"({base['model']}, Wilcoxon) ==="
    )
    print(comparison_dataframe.to_string(index=False))


def _grafikon(res: pd.DataFrame, outdir: Path):
    top = res.iloc[::-1]
    figure, axes = plt.subplots(figsize=(8, 4))
    axes.barh(
        top.model,
        top.cv_macroF1_sr,
        xerr=top.cv_macroF1_std,
        color="#4c72b0",
        error_kw={"ecolor": "#333", "capsize": 3},
    )
    axes.set_xlabel(
        f"macro-F1 ({BROJ_VALIDACIONIH_FOLDOVA}-slojna CV)"
    )
    axes.set_title(
        f"Klasifikatori nad Ollama vektorima ({OLLAMA_MODEL})"
    )
    figure.tight_layout()
    figure.savefig(
        outdir / "enkoderi_poredjenje.png",
        dpi=150,
    )
    plt.close(figure)
    print(f"\nGrafikoni sacuvani u: {outdir}/")


if __name__ == "__main__":
    main()