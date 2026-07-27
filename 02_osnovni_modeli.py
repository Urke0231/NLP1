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

import concurrent.futures
import json
import os
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
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_score, train_test_split)
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
PUTANJA_PODACI = "./anotacije-2026-07-26.json"

# Direktorijum za tabele i grafikone. Kreira se automatski.
IZLAZNI_DIREKTORIJUM = "rezultati_jednostavni_konacni_10_CV_NER"

# --- STA SE POREDI ------------------------------------------------------------

# Varijante pretprocesiranja. Zakomentarisite red da biste je izbacili.
# "lower+lema" zahteva CLASSLA i traje najduze - prvi put je pustite preko noci
# ili je privremeno iskljucite dok testirate ostatak.
VARIJANTE_PRETPROCESIRANJA = [
    # "sirovo",       # samo transliteracija u latinicu + sredjivanje razmaka
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

# 1/10 podataka se izdvaja kao test skup i ne koristi se ni za treniranje ni
# za podesavanje hiperparametara - sluzi samo za finalnu, nepristrasnu procenu.
VELICINA_TEST = 0.1

# Preostalih 9/10 se deli na 5 delova: 4 za trening, 1 za validaciju. Ta
# raspodela rotira (5-slojna unakrsna validacija) - koristi se i za
# podesavanje hiperparametara (GridSearchCV) i za procenu stabilnosti modela.
BROJ_VALIDACIONIH_FOLDOVA = 10

SLUCAJNO_SEME = 42   # isti seed => isti test skup i isti foldovi za sve konfiguracije

# Mreze hiperparametara koje se pretrazuju ugnezdenom validacijom.
MREZA_C = [0.01, 0.1, 1, 10, 100]        # LogRegresija i LinearSVM
MREZA_ALPHA = [0.01, 0.1, 0.5, 1.0, 2.0]  # MultinomialNB

# Minimalna frekvencija termina u vektorizatorima. Ako imate mali skup
# (< 500 primera), spustite na 1.
MIN_DF_REC = 2
MIN_DF_KARAKTER = 3

# --- PERFORMANSE -------------------------------------------------------------

# Broj tredova za paralelizaciju spoljasnje petlje po konfiguracijama
# (pretprocesiranje x odlike x model). Svaka konfiguracija se trenira i
# validira nezavisno, pa se dobro paralelizuje.
BROJ_NITI = 12

# Broj procesorskih jezgara za pretragu hiperparametara UNUTAR jedne
# konfiguracije (GridSearchCV). -1 = sva. Kada je BROJ_NITI > 1, ovo se
# automatski deli medju tredovima da ne bi doslo do preraspodele jezgara.
BROJ_JEZGARA = -1

# BRZI_TEST = True redukuje mrezu na 2 pretprocesiranja x 2 odlike i manju
# mrezu hiperparametara. Koristite dok proveravate da li sve radi.
BRZI_TEST = False

# =============================================================================
#                    kraj konfiguracije - ispod ne treba menjati
# =============================================================================

TOK = r"(?u)\S+"  # tekstovi su vec tokenizovani i razdvojeni razmakom

1
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


def evaluate_config(pname: str, texts: List[str], fname: str, vec_factory,
                    mname: str, clf, grid: dict, y: np.ndarray,
                    train_idx: np.ndarray, test_idx: np.ndarray,
                    labels: List[str], n_jobs: int):
    """Trenira i podesava jednu konfiguraciju na 90% podataka (5-slojna
    unakrsna validacija: 4 dela za trening, 1 za validaciju, rotira se), a
    zatim je JEDNOM ocenjuje na izdvojenom test skupu (10%), koji nije video
    ni trening ni podesavanje hiperparametara.

    Bezbedno za pozivanje iz vise niti - klasifikator se klonira, pa dve niti
    nikad ne dele isti mutabilni estimator objekat.
    """
    X = np.asarray(texts, dtype=object)
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    cv = StratifiedKFold(n_splits=BROJ_VALIDACIONIH_FOLDOVA, shuffle=True,
                         random_state=SLUCAJNO_SEME)
    pipe = Pipeline([("vec", vec_factory()), ("clf", clone(clf))])

    t = time.time()
    if grid:
        search = GridSearchCV(pipe, grid, scoring="f1_macro", cv=cv,
                              n_jobs=n_jobs, refit=True)
        search.fit(X_tr, y_tr)
        est, best = search.best_estimator_, search.best_params_
        idx = search.best_index_
        fold_skorovi = [search.cv_results_[f"split{k}_test_score"][idx]
                        for k in range(BROJ_VALIDACIONIH_FOLDOVA)]
    else:
        fold_skorovi = cross_val_score(pipe, X_tr, y_tr, scoring="f1_macro",
                                       cv=cv, n_jobs=n_jobs).tolist()
        pipe.fit(X_tr, y_tr)
        est, best = pipe, {}

    fold_skorovi = np.array(fold_skorovi)
    pred_test = est.predict(X_te)

    row = {
        "pretprocesiranje": pname,
        "odlike": fname,
        "model": mname,
        "cv_macroF1_sr": fold_skorovi.mean(),
        "cv_macroF1_std": fold_skorovi.std(),
        "test_macroF1": f1_score(y_te, pred_test, average="macro",
                                 labels=labels, zero_division=0),
        "test_tacnost": (pred_test == y_te).mean(),
        "hiperparametri": json.dumps(best),
        "fold_skorovi": fold_skorovi.tolist(),
        "vreme_s": round(time.time() - t, 1),
    }
    return row, pred_test.astype(str)


def _pitaj_geo_filter() -> str | None:
    """Pita korisnika na pocetku pokretanja da li da se uklone geografska/
    politicka imena (npr. drzave, gradovi) pre vektorizacije, da ne bi lazno
    uticala na ocenu sentimenta (npr. "Srbija" korelira sa sentimentom samo
    zato sto se cesce pojavljuje u domacim vestima, ne zato sto nosi sentiment).
    """
    print("\nUkloniti geografska/politicka imena iz teksta pre vektorizacije?")
    print("  1 = rucna lista korena (brzo)")
    print("  2 = automatska NER detekcija (CLASSLA, sporije)")
    print("  Enter = ne uklanjaj nista")
    izbor = input("Izbor [1/2/Enter]: ").strip()
    mapa = {"1": "stopwords", "2": "ner"}
    if izbor and izbor not in mapa:
        print(f"Nepoznat unos '{izbor}' - nastavljam bez uklanjanja imena.")
    return mapa.get(izbor)


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

    geo_filter = _pitaj_geo_filter()

    # Svaka varijanta pretprocesiranja (lower, lower+stem, lower+lema) se
    # racuna TACNO JEDNOM ovde - stem/lema su i kesirani na disku
    # (preprocessing.py) - i deli se izmedju svih niti i konfiguracija ispod.
    trazi_lemu = "lower+lema" in VARIJANTE_PRETPROCESIRANJA
    variants = build_variants(df["tekst"].tolist(), include_lemma=trazi_lemu, lemma_use_gpu=True,
                              geo_filter=geo_filter)
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

    # Fiksni test skup (1/10) - isti za sve konfiguracije, izdvaja se PRE bilo
    # kakvog treniranja ili podesavanja hiperparametara.
    train_idx, test_idx = train_test_split(
        np.arange(len(y)), test_size=VELICINA_TEST, stratify=y,
        random_state=SLUCAJNO_SEME)
    print(f"Test skup: {len(test_idx)} primera ({VELICINA_TEST:.0%}), "
         f"trening+validacija: {len(train_idx)} primera "
         f"({BROJ_VALIDACIONIH_FOLDOVA}-slojna krosvalidacija)")

    n_jobs = BROJ_JEZGARA if BROJ_NITI <= 1 else max(
        1, (os.cpu_count() or BROJ_NITI) // BROJ_NITI)

    prva_odlika = list(features)[0]
    tasks = []
    for pname, texts in variants.items():
        for fname, fac in features.items():
            for mname, (clf, grid) in models.items():
                # Vecinski baseline ne zavisi od odlika - racunaj ga jednom.
                if mname == "Vecinski" and fname != prva_odlika:
                    continue
                tasks.append((pname, texts, fname, fac, mname, clf, grid))

    total = len(tasks)
    t0 = time.time()
    print(f"Pokrecem {total} konfiguracija na {BROJ_NITI} niti(i), "
         f"{n_jobs} jezgro/a po niti...")

    rows, test_pred_store = [], {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=BROJ_NITI) as ex:
        futures = [
            ex.submit(evaluate_config, pname, texts, fname, fac, mname, clf,
                      grid, y, train_idx, test_idx, labels, n_jobs)
            for pname, texts, fname, fac, mname, clf, grid in tasks
        ]
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            row, pred_test = fut.result()
            key = f"{row['pretprocesiranje']} | {row['odlike']} | {row['model']}"
            test_pred_store[key] = pred_test
            rows.append(row)
            print(f"[{i:>3}/{total}] {key:<45} CV macro-F1 = "
                 f"{row['cv_macroF1_sr']:.4f} +/- {row['cv_macroF1_std']:.4f} "
                 f"| TEST macro-F1 = {row['test_macroF1']:.4f} "
                 f"({row['vreme_s']:.1f}s)", flush=True)

    # Rangiranje po CV skoru (validacija) - test skup se NE koristi za izbor
    # najbolje konfiguracije, samo za finalno nepristrasno izvestavanje.
    res = pd.DataFrame(rows).sort_values("cv_macroF1_sr", ascending=False)
    res.drop(columns=["fold_skorovi"]).to_csv(
        outdir / "rezultati_osnovni_modeli.csv", index=False, encoding="utf-8")
    (outdir / "fold_skorovi.json").write_text(
        json.dumps({r["pretprocesiranje"] + "|" + r["odlike"] + "|" + r["model"]:
                    r["fold_skorovi"] for r in rows}, indent=2), encoding="utf-8")

    print(f"\nUkupno vreme: {(time.time() - t0) / 60:.1f} min")
    print("\n=== TOP 10 KONFIGURACIJA (rangirano po CV macro-F1) ===")
    print(res.head(10)[["pretprocesiranje", "odlike", "model",
                        "cv_macroF1_sr", "cv_macroF1_std",
                        "test_macroF1"]].to_string(index=False))

    _analiza_najboljeg(res, test_pred_store, y, train_idx, test_idx, labels,
                       variants, features, models, outdir)
    _statisticko_poredjenje(rows, outdir)
    _grafikoni(res, outdir)


def _analiza_najboljeg(res, test_pred_store, y, train_idx, test_idx, labels,
                       variants, features, models, outdir: Path):
    best = res.iloc[0]
    key = f"{best.pretprocesiranje} | {best.odlike} | {best.model}"
    pred_test = test_pred_store[key]
    y_test = y[test_idx]

    print(f"\n=== NAJBOLJA KONFIGURACIJA: {key} ===")
    print(classification_report(y_test, pred_test, labels=labels, digits=3,
                                zero_division=0))

    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_test, pred_test, labels=labels)
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(f"Matrica konfuzije (test skup)\n{key}")
    ax.set_xlabel("predvidjeno")
    ax.set_ylabel("stvarno")
    fig.tight_layout()
    fig.savefig(outdir / "matrica_konfuzije.png", dpi=150)

    # Najinformativnije odlike po klasi (samo za linearne modele).
    if best.model not in ("LogRegresija", "LinearSVM"):
        return
    clf, grid = models[best.model]
    X_tr = np.asarray(variants[best.pretprocesiranje], dtype=object)[train_idx]
    y_tr = y[train_idx]
    pipe = Pipeline([("vec", features[best.odlike]()), ("clf", clone(clf))])
    if grid:
        pipe = GridSearchCV(
            pipe, grid, scoring="f1_macro",
            cv=StratifiedKFold(BROJ_VALIDACIONIH_FOLDOVA, shuffle=True,
                               random_state=SLUCAJNO_SEME),
            n_jobs=BROJ_JEZGARA).fit(X_tr, y_tr).best_estimator_
    else:
        pipe.fit(X_tr, y_tr)

    try:
        names = np.array(pipe.named_steps["vec"].get_feature_names_out())
        coef = pipe.named_steps["clf"].coef_
        classes = pipe.named_steps["clf"].classes_
        lines = []
        for ci, cls in enumerate(classes):
            w = coef[ci] if coef.shape[0] > 1 else coef[0]
            za_idx = np.argsort(w)[-20:][::-1]
            protiv_idx = np.argsort(w)[:20]
            lines.append(f"\n### Klasa: {cls}")
            lines.append("ZA:")
            lines.extend(f"    {names[i]:<25} {w[i]:+.4f}" for i in za_idx)
            lines.append("PROTIV:")
            lines.extend(f"    {names[i]:<25} {w[i]:+.4f}" for i in protiv_idx)
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

    rows = sorted(rows, key=lambda r: -r["cv_macroF1_sr"])
    base = rows[0]
    out = []
    for r in rows[1:]:
        a, b = np.array(base["fold_skorovi"]), np.array(r["fold_skorovi"])
        p = 1.0 if np.allclose(a, b) else wilcoxon(a, b).pvalue
        out.append({
            "konfiguracija": f"{r['pretprocesiranje']} | {r['odlike']} | {r['model']}",
            "cv_macroF1": round(r["cv_macroF1_sr"], 4),
            "razlika": round(base["cv_macroF1_sr"] - r["cv_macroF1_sr"], 4),
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
            index="pretprocesiranje", columns="odlike", values="cv_macroF1_sr")
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
    ax.barh(lbl, top.cv_macroF1_sr, xerr=top.cv_macroF1_std, color="#4c72b0",
            error_kw=dict(ecolor="#333", capsize=3))
    ax.set_xlabel(f"macro-F1 ({BROJ_VALIDACIONIH_FOLDOVA}-slojna CV)")
    ax.set_title("Najbolje konfiguracije osnovnih modela")
    ax.set_xlim(left=max(0, top.cv_macroF1_sr.min() - 0.15))
    fig.tight_layout()
    fig.savefig(outdir / "top_konfiguracije.png", dpi=150)

    # 3) Marginalni efekat svake tehnike
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    real.boxplot(column="cv_macroF1_sr", by="pretprocesiranje", ax=axes[0])
    axes[0].set_title("Efekat pretprocesiranja")
    axes[0].set_xlabel("")
    real.boxplot(column="cv_macroF1_sr", by="odlike", ax=axes[1])
    axes[1].set_title("Efekat odlika")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=45)
    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(outdir / "marginalni_efekti.png", dpi=150)

    print(f"\nGrafikoni sacuvani u: {outdir}/")


if __name__ == "__main__":
    main()
