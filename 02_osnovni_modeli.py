"""
Faza 3a - osnovni (linearni) modeli.

Poredi varijante pretprocesiranja (lowercasing, stemovanje, lematizacija) i
varijante odlika (TF, IDF, TF-IDF, n-grami reci, karakterski n-grami) kroz
10-slojnu stratifikovanu unakrsnu validaciju sa ugnezdenom optimizacijom
hiperparametara.

Pokretanje:  python 02_osnovni_modeli.py
Nema parametara iz komandne linije - sve se podesava u bloku KONFIGURACIJA.
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from data import load_dataset, present_labels
from preprocessing import USING_EXTERNAL_STEMMER, build_variants

warnings.filterwarnings("ignore")

# =============================================================================
#                              K O N F I G U R A C I J A
# =============================================================================

# --- OBAVEZNO ----------------------------------------------------------------

# Putanja do anotiranog skupa (JSON lista objekata ili JSONL).
PUTANJA_PODACI = "./stemovani_podaci.json"

# Direktorijum za tabele i grafikone. Kreira se automatski.
IZLAZNI_DIREKTORIJUM = "rezultati"

# --- STA SE POREDI ------------------------------------------------------------

# Varijante pretprocesiranja. Zakomentarisite red da biste je izbacili.
# "lower+lema" zahteva CLASSLA i traje najduze - prvi put je pustite preko noci
# ili je privremeno iskljucite dok testirate ostatak.
VARIJANTE_PRETPROCESIRANJA = [
    "sirovo",       # samo transliteracija u latinicu + sredjivanje razmaka
    "lower",        # + lowercasing
    "lower+stem",   # + stemovanje (SerbianStemmer)
    "lower+lema",   # + lematizacija (CLASSLA)
]

# Varijante odlika. Nazivi moraju odgovarati kljucevima iz _feature_defs().
VARIJANTE_ODLIKA = [
    "TF",           # ciste frekvencije termina, bez IDF-a
    "IDF",          # binarno prisustvo x IDF
    "TFIDF",        # klasican TF-IDF, unigrami
    "TFIDF_1-2",    # + bigrami
    "TFIDF_1-3",    # + trigrami
    "CHAR_3-5",     # karakterski n-grami (dobri za bogatu morfologiju)
    "REC+CHAR",     # unija recnih i karakterskih odlika
]

# Modeli. "Vecinski" je baseline i racuna se samo jednom.
MODELI = [
    "Vecinski",
    "MultinomialNB",
    "LogRegresija",
    "LinearSVM",
]

# --- PROTOKOL EVALUACIJE -----------------------------------------------------

BROJ_SPOLJASNJIH_FOLDOVA = 10   # postavka projekta trazi 10
BROJ_UNUTRASNJIH_FOLDOVA = 5    # za optimizaciju hiperparametara
SLUCAJNO_SEME = 42              # isti seed => isti foldovi za sve konfiguracije

# Mreze hiperparametara koje se pretrazuju ugnezdenom validacijom.
MREZA_C = [0.01, 0.1, 1, 10, 100]        # LogRegresija i LinearSVM
MREZA_ALPHA = [0.01, 0.1, 0.5, 1.0, 2.0]  # MultinomialNB

# Minimalna frekvencija termina u vektorizatorima. Ako imate mali skup
# (< 500 primera), spustite na 1.
MIN_DF_REC = 2
MIN_DF_KARAKTER = 3

# --- PERFORMANSE -------------------------------------------------------------

# Broj procesorskih jezgara za pretragu hiperparametara. -1 = sva.
BROJ_JEZGARA = -1

# BRZI_TEST = True redukuje mrezu na 2 pretprocesiranja x 2 odlike i manju
# mrezu hiperparametara. Koristite dok proveravate da li sve radi.
BRZI_TEST = False

# =============================================================================
#                    kraj konfiguracije - ispod ne treba menjati
# =============================================================================

TOK = r"(?u)\S+"  # tekstovi su vec tokenizovani i razdvojeni razmakom


def _feature_defs() -> Dict[str, callable]:
    return {
        "TF": lambda: CountVectorizer(token_pattern=TOK, min_df=MIN_DF_REC),
        "IDF": lambda: TfidfVectorizer(token_pattern=TOK, min_df=MIN_DF_REC,
                                       binary=True, use_idf=True),
        "TFIDF": lambda: TfidfVectorizer(token_pattern=TOK, min_df=MIN_DF_REC,
                                         sublinear_tf=True),
        "TFIDF_1-2": lambda: TfidfVectorizer(token_pattern=TOK,
                                             min_df=MIN_DF_REC,
                                             ngram_range=(1, 2),
                                             sublinear_tf=True),
        "TFIDF_1-3": lambda: TfidfVectorizer(token_pattern=TOK,
                                             min_df=MIN_DF_KARAKTER,
                                             ngram_range=(1, 3),
                                             sublinear_tf=True),
        "CHAR_3-5": lambda: TfidfVectorizer(analyzer="char_wb",
                                            ngram_range=(3, 5),
                                            min_df=MIN_DF_KARAKTER,
                                            sublinear_tf=True),
        "REC+CHAR": lambda: FeatureUnion([
            ("rec", TfidfVectorizer(token_pattern=TOK, min_df=MIN_DF_REC,
                                    ngram_range=(1, 2), sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                     min_df=MIN_DF_KARAKTER,
                                     sublinear_tf=True)),
        ]),
    }


def _model_defs() -> Dict[str, tuple]:
    c_grid = [0.1, 1, 10] if BRZI_TEST else MREZA_C
    a_grid = [0.1, 1.0] if BRZI_TEST else MREZA_ALPHA
    return {
        "Vecinski": (DummyClassifier(strategy="most_frequent"), {}),
        "MultinomialNB": (MultinomialNB(), {"clf__alpha": a_grid}),
        "LogRegresija": (
            LogisticRegression(max_iter=3000, class_weight="balanced",
                               random_state=SLUCAJNO_SEME),
            {"clf__C": c_grid}),
        "LinearSVM": (
            LinearSVC(class_weight="balanced", random_state=SLUCAJNO_SEME,
                      max_iter=5000),
            {"clf__C": c_grid}),
    }


def nested_cv(X: List[str], y: np.ndarray, vec_factory, clf, grid,
              labels: List[str]):
    """Vraca (po-fold macro-F1, out-of-fold predikcije, izabrani hiperparametri)."""
    outer = StratifiedKFold(n_splits=BROJ_SPOLJASNJIH_FOLDOVA, shuffle=True,
                            random_state=SLUCAJNO_SEME)
    inner = StratifiedKFold(n_splits=BROJ_UNUTRASNJIH_FOLDOVA, shuffle=True,
                            random_state=SLUCAJNO_SEME)

    X = np.asarray(X, dtype=object)
    oof = np.empty(len(y), dtype=object)
    scores, chosen = [], []

    for tr, te in outer.split(X, y):
        pipe = Pipeline([("vec", vec_factory()), ("clf", clf)])
        if grid:
            search = GridSearchCV(pipe, grid, scoring="f1_macro", cv=inner,
                                  n_jobs=BROJ_JEZGARA, refit=True)
            search.fit(X[tr], y[tr])
            est, best = search.best_estimator_, search.best_params_
        else:
            pipe.fit(X[tr], y[tr])
            est, best = pipe, {}
        pred = est.predict(X[te])
        oof[te] = pred
        scores.append(f1_score(y[te], pred, average="macro", labels=labels,
                               zero_division=0))
        chosen.append(best)

    return np.array(scores), oof.astype(str), chosen


def main():
    put = Path(PUTANJA_PODACI)
    if not put.exists():
        sys.exit(f"GRESKA: ne postoji fajl sa podacima: {put.resolve()}\n"
                 f"Ispravite PUTANJA_PODACI na vrhu skripte.")

    outdir = Path(IZLAZNI_DIREKTORIJUM)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(put)
    labels = present_labels(df)
    y = df["sentiment"].to_numpy()
    print(f"Ucitano {len(df)} primera, klase: {labels}")
    if not USING_EXTERNAL_STEMMER and "lower+stem" in VARIJANTE_PRETPROCESIRANJA:
        print("UPOZORENJE: SerbianStemmer.py nije pronadjen - koristi se "
              "ugradjeni rezervni stemer. Vidi README.")

    trazi_lemu = "lower+lema" in VARIJANTE_PRETPROCESIRANJA
    variants = build_variants(df["tekst"].tolist(), include_lemma=trazi_lemu)
    variants = {k: v for k, v in variants.items()
                if k in VARIJANTE_PRETPROCESIRANJA}
    features = {k: v for k, v in _feature_defs().items()
                if k in VARIJANTE_ODLIKA}
    models = {k: v for k, v in _model_defs().items() if k in MODELI}

    if BRZI_TEST:
        variants = dict(list(variants.items())[:2])
        features = dict(list(features.items())[:2])
        print("BRZI_TEST = True: mreza je redukovana.")

    if not variants or not features or not models:
        sys.exit("GRESKA: prazna lista varijanti. Proverite konfiguraciju.")

    rows, oof_store = [], {}
    prva_odlika = list(features)[0]
    total = len(variants) * len(features) * len(models)
    i = 0
    t0 = time.time()

    for pname, texts in variants.items():
        for fname, fac in features.items():
            for mname, (clf, grid) in models.items():
                i += 1
                # Vecinski baseline ne zavisi od odlika - racunaj ga jednom.
                if mname == "Vecinski" and fname != prva_odlika:
                    continue
                t = time.time()
                scores, oof, chosen = nested_cv(texts, y, fac, clf, grid, labels)
                key = f"{pname} | {fname} | {mname}"
                oof_store[key] = oof
                rows.append({
                    "pretprocesiranje": pname,
                    "odlike": fname,
                    "model": mname,
                    "macroF1_sr": scores.mean(),
                    "macroF1_std": scores.std(),
                    "tacnost": (oof == y).mean(),
                    "hiperparametri": (
                        pd.Series([json.dumps(c) for c in chosen])
                        .value_counts().index[0] if grid else "{}"),
                    "fold_skorovi": scores.tolist(),
                    "vreme_s": round(time.time() - t, 1),
                })
                print(f"[{i:>3}/{total}] {key:<45} macro-F1 = "
                      f"{scores.mean():.4f} +/- {scores.std():.4f} "
                      f"({time.time() - t:.1f}s)", flush=True)

    res = pd.DataFrame(rows).sort_values("macroF1_sr", ascending=False)
    res.drop(columns=["fold_skorovi"]).to_csv(
        outdir / "rezultati_osnovni_modeli.csv", index=False, encoding="utf-8")
    (outdir / "fold_skorovi.json").write_text(
        json.dumps({r["pretprocesiranje"] + "|" + r["odlike"] + "|" + r["model"]:
                    r["fold_skorovi"] for r in rows}, indent=2), encoding="utf-8")

    print(f"\nUkupno vreme: {(time.time() - t0) / 60:.1f} min")
    print("\n=== TOP 10 KONFIGURACIJA ===")
    print(res.head(10)[["pretprocesiranje", "odlike", "model",
                        "macroF1_sr", "macroF1_std"]].to_string(index=False))

    _analiza_najboljeg(res, oof_store, y, labels, variants, features, models,
                       outdir)
    _statisticko_poredjenje(rows, outdir)
    _grafikoni(res, outdir)


def _analiza_najboljeg(res, oof_store, y, labels, variants, features, models,
                       outdir: Path):
    best = res.iloc[0]
    key = f"{best.pretprocesiranje} | {best.odlike} | {best.model}"
    oof = oof_store[key]

    print(f"\n=== NAJBOLJA KONFIGURACIJA: {key} ===")
    print(classification_report(y, oof, labels=labels, digits=3, zero_division=0))

    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y, oof, labels=labels)
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(f"Matrica konfuzije (out-of-fold)\n{key}")
    ax.set_xlabel("predvidjeno")
    ax.set_ylabel("stvarno")
    fig.tight_layout()
    fig.savefig(outdir / "matrica_konfuzije.png", dpi=150)

    # Najinformativnije odlike po klasi (samo za linearne modele).
    if best.model not in ("LogRegresija", "LinearSVM"):
        return
    clf, grid = models[best.model]
    Xb = np.asarray(variants[best.pretprocesiranje], dtype=object)
    pipe = Pipeline([("vec", features[best.odlike]()), ("clf", clf)])
    if grid:
        pipe = GridSearchCV(
            pipe, grid, scoring="f1_macro",
            cv=StratifiedKFold(BROJ_UNUTRASNJIH_FOLDOVA, shuffle=True,
                               random_state=SLUCAJNO_SEME),
            n_jobs=BROJ_JEZGARA).fit(Xb, y).best_estimator_
    else:
        pipe.fit(Xb, y)

    try:
        names = np.array(pipe.named_steps["vec"].get_feature_names_out())
        coef = pipe.named_steps["clf"].coef_
        classes = pipe.named_steps["clf"].classes_
        lines = []
        for ci, cls in enumerate(classes):
            w = coef[ci] if coef.shape[0] > 1 else coef[0]
            lines.append(f"\n### Klasa: {cls}")
            lines.append("ZA:     " + ", ".join(names[np.argsort(w)[-20:][::-1]]))
            lines.append("PROTIV: " + ", ".join(names[np.argsort(w)[:20]]))
        txt = "\n".join(lines)
        print("\n=== NAJINFORMATIVNIJE ODLIKE ===" + txt)
        (outdir / "najinformativnije_odlike.txt").write_text(txt, encoding="utf-8")
    except Exception as e:
        print(f"(preskacem prikaz odlika: {e})")


def _statisticko_poredjenje(rows, outdir: Path):
    """Uparen Wilcoxonov test po foldovima, u odnosu na najbolju konfiguraciju.

    Foldovi su identicni za sve konfiguracije (isto SLUCAJNO_SEME), pa je
    uparen test opravdan. Ovo ide u dokumentaciju umesto golog poredjenja
    proseka.
    """
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        print("(za statisticko poredjenje: pip install scipy)")
        return

    rows = sorted(rows, key=lambda r: -r["macroF1_sr"])
    base = rows[0]
    out = []
    for r in rows[1:]:
        a, b = np.array(base["fold_skorovi"]), np.array(r["fold_skorovi"])
        p = 1.0 if np.allclose(a, b) else wilcoxon(a, b).pvalue
        out.append({
            "konfiguracija": f"{r['pretprocesiranje']} | {r['odlike']} | {r['model']}",
            "macroF1": round(r["macroF1_sr"], 4),
            "razlika": round(base["macroF1_sr"] - r["macroF1_sr"], 4),
            "p_vrednost": round(p, 4),
            "znacajno_p<0.05": p < 0.05,
        })
    dfp = pd.DataFrame(out)
    dfp.to_csv(outdir / "statisticko_poredjenje.csv", index=False, encoding="utf-8")
    print("\n=== POREDJENJE SA NAJBOLJOM KONFIGURACIJOM (Wilcoxon) ===")
    print(dfp.head(15).to_string(index=False))


def _grafikoni(res: pd.DataFrame, outdir: Path):
    real = res[res.model != "Vecinski"]
    if real.empty:
        return

    # 1) Toplotna mapa: pretprocesiranje x odlike, po modelu
    modeli = sorted(real.model.unique())
    fig, axes = plt.subplots(1, len(modeli), figsize=(6 * len(modeli), 4.5),
                             squeeze=False)
    for ax, m in zip(axes[0], modeli):
        piv = real[real.model == m].pivot_table(
            index="pretprocesiranje", columns="odlike", values="macroF1_sr")
        im = ax.imshow(piv.values, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels(piv.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels(piv.index)
        ax.set_title(m)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                            color="w", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Macro-F1: efekat pretprocesiranja i odlika")
    fig.tight_layout()
    fig.savefig(outdir / "toplotna_mapa.png", dpi=150)

    # 2) Najbolje konfiguracije sa intervalima
    top = res.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    lbl = top.pretprocesiranje + " | " + top.odlike + " | " + top.model
    ax.barh(lbl, top.macroF1_sr, xerr=top.macroF1_std, color="#4c72b0",
            error_kw=dict(ecolor="#333", capsize=3))
    ax.set_xlabel(f"macro-F1 ({BROJ_SPOLJASNJIH_FOLDOVA}-slojna CV)")
    ax.set_title("Najbolje konfiguracije osnovnih modela")
    ax.set_xlim(left=max(0, top.macroF1_sr.min() - 0.15))
    fig.tight_layout()
    fig.savefig(outdir / "top_konfiguracije.png", dpi=150)

    # 3) Marginalni efekat svake tehnike
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    real.boxplot(column="macroF1_sr", by="pretprocesiranje", ax=axes[0])
    axes[0].set_title("Efekat pretprocesiranja")
    axes[0].set_xlabel("")
    real.boxplot(column="macroF1_sr", by="odlike", ax=axes[1])
    axes[1].set_title("Efekat odlika")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=45)
    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(outdir / "marginalni_efekti.png", dpi=150)

    print(f"\nGrafikoni sacuvani u: {outdir}/")


if __name__ == "__main__":
    main()
