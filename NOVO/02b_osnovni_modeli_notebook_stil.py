"""
Faza 3a (varijanta) - osnovni (linearni) modeli, u stilu Batanovic/nesto.ipynb.

Za razliku od 02_osnovni_modeli.py (koji sve poredi kroz jednu opstu
evaluate_config funkciju), ovde se koristi ISTI, jednostavan kod kao u
notebook-u za svaki model: goli CountVectorizer/TfidfVectorizer, Pipeline sa
koracima 'vectorizer'/'classifier', direktan cross_val_score i GridSearchCV
(ugnezden preko cross_val_score(GridSearchCV(...))) - primenjen na projektni
anotirani skup (data.py) umesto na SerbMR-2C.csv.

I dalje se, kao i u 02_osnovni_modeli.py, poredi:
  - VARIJANTE PRETPROCESIRANJA (lower / lower+stem / lower+lema, preko
    preprocessing.build_variants) - geografska/politicka imena se UVEK
    automatski uklanjaju NER-om (CLASSLA) pre svega ostalog, vidi GEO_FILTER;
  - VARIJANTE ODLIKA (TF/IDF/TFIDF/n-grami/karakterski n-grami);
  - MODELE (MultinomialNB, LogRegresija, LinearSVM, sa i bez GridSearchCV
    optimizacije hiperparametara) - svaki racunat notebook-skim kodom.

Pokretanje:  python 02b_osnovni_modeli_notebook_stil.py
Nema parametara iz komandne linije - sve se podesava u bloku KONFIGURACIJA.
"""

from __future__ import annotations

import os

# MORA da se postavi PRE uvoza numpy/scipy/sklearn (i u glavnom i u svakom
# spawn-ovanom procesu, jer Windows ProcessPoolExecutor iznova izvrsava ovaj
# modul od vrha za svaki worker). Bez ovoga, BLAS biblioteka (OpenBLAS/MKL)
# podrazumevano otvara sopstvene niti po procesu - sa BROJ_PROCESA=20 to
# stvara na hiljade konkurentnih native niti, sto na Windows-u pouzdano
# obara worker proces (BrokenProcessPool) jer je svaki proces vec ogranicen
# na n_jobs=1 preko BROJ_JEZGARA, pa dodatne BLAS niti ne pomazu nista.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import concurrent.futures
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_predict, cross_val_score,
                                     train_test_split)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from data import load_dataset, present_labels
from preprocessing import USING_EXTERNAL_STEMMER, build_variants

warnings.filterwarnings("ignore")

# =============================================================================
#                              K O N F I G U R A C I J A
# =============================================================================

# Putanja do anotiranog skupa (JSON lista objekata ili JSONL).
PUTANJA_PODACI = "./anotacije-2026-07-26.json"

# Direktorijum za tabele i grafikone. Kreira se automatski.
IZLAZNI_DIREKTORIJUM = "rezultati_osnovni_modeli_notebook_stil1"

# Varijante pretprocesiranja koje se porede (isto sto i u 02_osnovni_modeli.py).
# "lower+lema" zahteva CLASSLA i traje najduze - prvi put je pustite preko
# noci ili je privremeno iskljucite dok testirate ostatak.
VARIJANTE_PRETPROCESIRANJA = [
    # "sirovo",       # samo transliteracija u latinicu + sredjivanje razmaka
    "lower",        # + lowercasing
    "lower+stem",   # + stemovanje (SerbianStemmer)
    "lower+lema",   # + lematizacija (CLASSLA)
]

# Geografska/politicka imena (drzave, gradovi...) mogu lazno korelisati sa
# sentimentom (npr. domace vs. strane vesti), a da sama rec nema sentiment.
# Uklanjaju se automatski, PRE svega ostalog pretprocesiranja:
#   "ner"       - automatska NER detekcija (CLASSLA) - podrazumevano
#   "stopwords" - rucna lista korena (brze, bez NER modela)
#   None        - ne uklanjaj nista
GEO_FILTER = "ner"

# Varijante odlika koje se porede (isto sto i u 02_osnovni_modeli.py).
VARIJANTE_ODLIKA = [
    "TF",           # ciste frekvencije termina, bez IDF-a (CountVectorizer)
    "IDF",          # binarno prisustvo x IDF
    "TFIDF",        # klasican TF-IDF, unigrami (kao celija 4 notebook-a)
    "TFIDF_1-2",    # + bigrami
    "TFIDF_1-3",    # + trigrami
    "CHAR_3-5",     # karakterski n-grami (dobri za bogatu morfologiju)
    "REC+CHAR",     # unija recnih i karakterskih odlika
]

# Minimalna frekvencija termina u vektorizatorima. Ako imate mali skup
# (< 500 primera), spustite na 1.
MIN_DF_REC = 2
MIN_DF_KARAKTER = 3

# Broj SPOLJNIH foldova - isti za sve modele (i obicne i GridSearch varijante),
# tako da su sve konfiguracije direktno uporedive. Notebook je za razlicite
# celije koristio razlicite vrednosti (cv=10 za celije 2-4, cv=5 za 5-6); ovde
# je to namerno objedinjeno na 10 radi dosledne, fer evaluacije.
BROJ_FOLDOVA = 10

# Broj UNUTRASNJIH foldova za sam GridSearchCV (bira hiperparametar C) - manji
# je namerno, jer se koristi samo za odabir C i ponavlja se za svaki spoljni
# fold (kao cv=2 u notebook celijama 5-6).
BROJ_FOLDOVA_UNUTRASNJI = 2

# Mreza hiperparametra C - identicna onoj iz notebook-a.
MREZA_C = [0.1, 1.0, 10]

# Isti seed kao train_test_split(..., random_state=42) u notebooku - koristi se
# i za stratifikovane foldove da rezultati budu ponovljivi.
SLUCAJNO_SEME = 42

# --- PERFORMANSE -------------------------------------------------------------

# Broj OS PROCESA za paralelizaciju spoljasnje petlje po konfiguracijama
# (pretprocesiranje x odlike x model). Svaka konfiguracija se racuna
# nezavisno, pa se dobro paralelizuje - isto kao u 02_osnovni_modeli.py.
# NAMERNO su ovo pravi procesi (ProcessPoolExecutor), a ne thread-ovi:
# CountVectorizer/TfidfVectorizer tokenizuju tekst u cistom Python-u (regex),
# sto drzi GIL i NE moze da se paralelizuje thread-ovima u istom procesu -
# thread-ovi bi samo naizmenicno cekali na GIL bez stvarnog ubrzanja. Svaki
# proces ima svoj GIL, pa se ovo realno paralelizuje na vise jezgara.
# Idite blizu broja fizickih jezgara (npr. 16-24 na 24-jezgarnom CPU-u).
BROJ_PROCESA = 8

# Broj jezgara za GridSearchCV UNUTAR jednog procesa (n_jobs). Ostaje na 1 -
# BROJ_PROCESA vec koristi sve fizicke procese/jezgra na spoljnom nivou, pa bi
# n_jobs>1 ovde pokrenuo UGNJEZDENE pod-procese (proces u procesu) i samo
# preraspodelio ista jezgra, bez dobitka.
BROJ_JEZGARA = 1

# =============================================================================
#                    kraj konfiguracije - ispod ne treba menjati
# =============================================================================

# Tekst je vec tokenizovan i razdvojen razmakom (preprocessing.py), pa
# vektorizatori treba da gledaju tacno te tokene umesto da sami tokenizuju.
TOK = r"(?u)\S+"


def _feature_defs() -> Dict[str, callable]:
    """Fabrike vektorizatora po varijanti odlika - isto sto i u
    02_osnovni_modeli.py.

    Bez parametara. Vraca recnik {naziv_varijante: fabrika}, gde je fabrika
    funkcija bez argumenata koja pravi NOVU, nefitovanu instancu vektorizatora
    pri svakom pozivu (bitno jer svaka konfiguracija/fold/proces mora dobiti
    svoju svezu instancu, ne deljenu vec-fitovanu)."""

    def napravi_tf():
        return CountVectorizer(token_pattern=TOK, min_df=MIN_DF_REC)

    def napravi_idf():
        return TfidfVectorizer(
            token_pattern=TOK,
            min_df=MIN_DF_REC,
            binary=True,
            use_idf=True,
        )

    def napravi_tfidf():
        return TfidfVectorizer(
            token_pattern=TOK,
            min_df=MIN_DF_REC,
            sublinear_tf=True,
            use_idf=True,
        )

    def napravi_tfidf_1_2():
        return TfidfVectorizer(
            token_pattern=TOK,
            min_df=MIN_DF_REC,
            ngram_range=(1, 2),
            sublinear_tf=True,
            use_idf=True,
        )

    def napravi_tfidf_1_3():
        return TfidfVectorizer(
            token_pattern=TOK,
            min_df=MIN_DF_KARAKTER,
            ngram_range=(1, 3),
            sublinear_tf=True,
            use_idf=True,
        )

    def napravi_char_3_5():
        return TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=MIN_DF_KARAKTER,
            sublinear_tf=True,
        )

    def napravi_rec_char_uniju():
        rec_odlike = TfidfVectorizer(
            token_pattern=TOK,
            min_df=MIN_DF_REC,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        char_odlike = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=MIN_DF_KARAKTER,
            sublinear_tf=True,
        )
        return FeatureUnion([("rec", rec_odlike), ("char", char_odlike)])

    return {
        "TF": napravi_tf,
        "IDF": napravi_idf,
        "TFIDF": napravi_tfidf,
        "TFIDF_1-2": napravi_tfidf_1_2,
        "TFIDF_1-3": napravi_tfidf_1_3,
        "CHAR_3-5": napravi_char_3_5,
        "REC+CHAR": napravi_rec_char_uniju,
    }


def _cv_report(naziv: str, pretprocesiranje: str, model: str, odlika: str,
              estimator, corpus, y, cv, f1_takodje: bool, n_jobs: int) -> Dict:
    """Isto sto i notebook celije: cross_val_score(scoring=...).mean().

    Ako je f1_takodje=True, racuna i f1_macro drugim pozivom cross_val_score-a
    (tacno kao u celiji 3 notebook-a, gde se za LogRegresiju racunaju i
    accuracy i F-measure).

    Poziva se iz zasebnog OS procesa (ProcessPoolExecutor) - svaki proces
    dobija svoju kopiju estimator/corpus/y (pickle preko IPC-a), pa dve
    konfiguracije nikad ne dele stanje. Kada je estimator vec GridSearchCV
    (ima svoj n_jobs za unutrasnju pretragu), spoljasnji cross_val_score se
    namerno pusta sekvencijalno da ne bi doslo do ugnjezdenih pod-procesa.

    Parametri:
        naziv: puni citljiv naziv konfiguracije (npr. "lower | TFIDF |
            MultinomialNB") - ide u kolonu "konfiguracija" reda rezultata i
            u ispis progresa.
        pretprocesiranje: naziv varijante pretprocesiranja (pname, npr.
            "lower+stem") - samo se upisuje u red rezultata, radi kasnijeg
            grupisanja/toplotne mape.
        model: naziv modela (npr. "LogRegresija (GridSearch)") - upisuje se
            u red rezultata.
        odlika: naziv varijante odlika (fname, npr. "CHAR_3-5") - upisuje se
            u red rezultata.
        estimator: sklearn Pipeline (ili GridSearchCV oko pipeline-a) koji
            se fituje/evaluira preko cross_val_score-a.
        corpus: lista (vec pretprocesiranih) tekstova nad kojima se
            estimator trenira/testira kroz CV.
        y: serija labela (sentiment) - target za klasifikaciju.
        cv: splitter (ovde outer_cv, StratifiedKFold) koji definise spoljne
            foldove za cross_val_score.
        f1_takodje: ako je True, racuna se i drugi cross_val_score poziv sa
            scoring="f1_macro", pored accuracy-ja.
        n_jobs: broj jezgara za spoljni cross_val_score - interno se
            prisiljava na 1 ako je estimator vec GridSearchCV, da se izbegnu
            ugnjezdeni pod-procesi.
    """
    t0 = time.time()

    # Ako je estimator vec GridSearchCV, ne pokrecemo dodatne paralelne
    # procese ovde - unutrasnja pretraga vec koristi n_jobs (videti main()).
    if isinstance(estimator, GridSearchCV):
        outer_n_jobs = 1
    else:
        outer_n_jobs = n_jobs

    acc = cross_val_score(
        estimator,
        corpus,
        y,
        cv=cv,
        scoring="accuracy",
        n_jobs=outer_n_jobs,
    )
    row = {
        "konfiguracija": naziv,
        "pretprocesiranje": pretprocesiranje,
        "model": model,
        "odlika": odlika,
        "cv_accuracy_sr": acc.mean(),
        "cv_accuracy_std": acc.std(),
        "cv_f1_sr": None,
        "cv_f1_std": None,
        "fold_accuracy": acc.tolist(),
        "vreme_s": None,
    }

    if f1_takodje:
        f1 = cross_val_score(
            estimator,
            corpus,
            y,
            cv=cv,
            scoring="f1_macro",
            n_jobs=outer_n_jobs,
        )
        row["cv_f1_sr"] = f1.mean()
        row["cv_f1_std"] = f1.std()

    row["vreme_s"] = round(time.time() - t0, 1)
    return row


def main():
    """Glavna orkestraciona funkcija - bez parametara (sve se cita iz
    konstanti u bloku KONFIGURACIJA na vrhu fajla).

    Ucitava podatke, gradi sve varijante pretprocesiranja i odlika, sastavlja
    listu svih kombinacija (pretprocesiranje x odlike x model) u `tasks`,
    pokrece ih paralelno preko ProcessPoolExecutor pozivajuci _cv_report za
    svaku, cuva CSV sa rezultatima i poziva _najbolji_model/_grafikon/
    _heatmapa za izlazne grafikone."""
    put = Path(PUTANJA_PODACI)
    if not put.exists():
        sys.exit(f"GRESKA: ne postoji fajl sa podacima: {put.resolve()}\n"
                 f"Ispravite PUTANJA_PODACI na vrhu skripte.")

    outdir = Path(IZLAZNI_DIREKTORIJUM)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(put, drop_duplicates=True)
    labels = present_labels(df)
    y = df["sentiment"]
    print(f"Ucitano {len(df)} primera, klase: {labels}")
    if not USING_EXTERNAL_STEMMER and "lower+stem" in VARIJANTE_PRETPROCESIRANJA:
        print("UPOZORENJE: SerbianStemmer.py nije pronadjen - koristi se "
             "ugradjeni rezervni stemer. Vidi README.")

    # Svaka varijanta pretprocesiranja (lower, lower+stem, lower+lema) se
    # racuna TACNO JEDNOM ovde - stem/lema su i kesirani na disku
    # (preprocessing.py) - i deli se (pickle-om) izmedju svih procesa i
    # konfiguracija ispod.
    trazi_lemu = "lower+lema" in VARIJANTE_PRETPROCESIRANJA
    svi_tekstovi = df["tekst"].tolist()
    sve_varijante = build_variants(
        svi_tekstovi,
        include_lemma=trazi_lemu,
        lemma_use_gpu=True,
        geo_filter=GEO_FILTER,
    )

    # Zadrzavamo samo varijante pretprocesiranja koje su ukljucene u
    # konfiguraciji (VARIJANTE_PRETPROCESIRANJA) - build_variants ume da
    # vrati i vise od toga.
    variants = {}
    for naziv_varijante, tekstovi_varijante in sve_varijante.items():
        if naziv_varijante in VARIJANTE_PRETPROCESIRANJA:
            variants[naziv_varijante] = tekstovi_varijante

    # Isto tako, od svih definisanih fabrika odlika (_feature_defs) zadrzavamo
    # samo one ukljucene u konfiguraciji (VARIJANTE_ODLIKA).
    features = {}
    for naziv_odlike, fabrika_vektorizatora in _feature_defs().items():
        if naziv_odlike in VARIJANTE_ODLIKA:
            features[naziv_odlike] = fabrika_vektorizatora

    if not variants or not features:
        sys.exit("GRESKA: prazna VARIJANTE_PRETPROCESIRANJA/VARIJANTE_ODLIKA. "
                 "Proverite konfiguraciju.")

    # --- Celija 1 iz notebooka: brza provera na jednoj train/test podeli -----
    # (samo sanity-check na sirovom tekstu, ne ulazi u tabelu rezultata)
    vectorizer = CountVectorizer()
    clf = MultinomialNB()
    corpus_train, corpus_test, y_train, y_test = train_test_split(
        df["tekst"],
        y,
        test_size=0.2,
        random_state=SLUCAJNO_SEME,
    )
    X_train = vectorizer.fit_transform(corpus_train)
    clf.fit(X_train, y_train)
    X_test = vectorizer.transform(corpus_test)
    y_pred = clf.predict(X_test)

    broj_tacnih = (y_test == y_pred).sum()
    accuracy = broj_tacnih / X_test.shape[0]
    print("\n--- Brza provera (jedna train/test podela): "
         "MultinomialNB + CountVectorizer (sirov tekst) ---")
    print("Accuracy: ", accuracy)

    n_jobs = BROJ_JEZGARA

    # Jedan te isti spoljni fold-splitter za BAS SVE konfiguracije (stratifikovan,
    # promesan, fiksni seme) - direktna uporedivost rezultata. Unutrasnji
    # splitter se koristi SAMO unutar GridSearchCV-a, za odabir C.
    outer_cv = StratifiedKFold(
        n_splits=BROJ_FOLDOVA,
        shuffle=True,
        random_state=SLUCAJNO_SEME,
    )
    inner_cv = StratifiedKFold(
        n_splits=BROJ_FOLDOVA_UNUTRASNJI,
        shuffle=True,
        random_state=SLUCAJNO_SEME,
    )

    # --- Gradimo listu (naziv, pretprocesiranje, model, odlika, estimator,
    #     corpus, cv, f1_takodje) za sve konfiguracije ---------------------
    estimators: Dict[str, Tuple[object, List[str]]] = {}
    tasks: List[Tuple[str, str, str, str, object, List[str], object, bool]] = []

    for pname, corpus in variants.items():
        for fname, vec_factory in features.items():

            # --- Celija 2: MultinomialNB, 10-slojna CV ----------------------
            naziv = f"{pname} | {fname} | MultinomialNB"
            p_clf = Pipeline([
                ("vectorizer", vec_factory()),
                ("classifier", MultinomialNB()),
            ])
            tasks.append((
                naziv,          # citljiv naziv konfiguracije
                pname,          # naziv varijante pretprocesiranja (lower, lower+stem, ...)
                "MultinomialNB",  # naziv modela
                fname,          # naziv varijante odlika (TF, TFIDF, ...)
                p_clf,          # sklearn pipeline (estimator) koji se evaluira
                corpus,         # tekstovi ove varijante pretprocesiranja
                outer_cv,       # spoljni CV splitter (isti za sve konfiguracije)
                True,           # f1_takodje - racunaj i macro-F1, ne samo accuracy
            ))
            estimators[naziv] = (p_clf, corpus)

            # --- Celija 3/4: LogRegresija, 10-slojna CV (acc i F1) ----------
            # napomena: notebook koristi solver='liblinear', ali on ne
            # podrzava multiklasnu klasifikaciju (nas skup ima >=3 klase) -
            # lbfgs radi multiklasno nativno.
            naziv = f"{pname} | {fname} | LogRegresija"
            clf = LogisticRegression(solver="lbfgs", max_iter=1000)
            p_clf = Pipeline([
                ("vectorizer", vec_factory()),
                ("classifier", clf),
            ])
            tasks.append((
                naziv,
                pname,
                "LogRegresija",
                fname,
                p_clf,
                corpus,
                outer_cv,
                True,
            ))
            estimators[naziv] = (p_clf, corpus)

            # --- Celija 5: LogRegresija, optimizacija hiperparametara -------
            # GridSearchCV bira C na inner_cv (2 folda); vec optimizovan
            # pipeline se onda ocenjuje na outer_cv (10 foldova) - to je
            # ugnjezdena (nested) CV, pa je odabir C i finalna ocena strogo
            # razdvojeni. GridSearchCV dobija n_jobs da paralelizuje
            # unutrasnju pretragu (C-mreza x unutrasnji foldovi); spoljni
            # cross_val_score ostaje sekvencijalan (vidi _cv_report).
            naziv = f"{pname} | {fname} | LogRegresija (GridSearch)"
            clf = LogisticRegression(solver="lbfgs", max_iter=1000)
            p_grid_lr = {"classifier__C": MREZA_C}
            p_clf = Pipeline([
                ("vectorizer", vec_factory()),
                ("classifier", clf),
            ])
            gs_clf = GridSearchCV(
                estimator=p_clf,
                param_grid=p_grid_lr,
                cv=inner_cv,
                scoring="accuracy",
                n_jobs=n_jobs,
            )
            tasks.append((
                naziv,
                pname,
                "LogRegresija (GridSearch)",
                fname,
                gs_clf,   # ovde je estimator GridSearchCV, ne "goli" pipeline
                corpus,
                outer_cv,
                True,
            ))
            estimators[naziv] = (gs_clf, corpus)

            # --- Celija 6: LinearSVM, sa i bez optimizacije ------------------
            # SVM bez kernela, L2 regularizacija, L2 funkcija gubitka,
            # resavanje u primalnom domenu.
            clf = LinearSVC(
                penalty="l2",
                loss="squared_hinge",
                dual=True,
                max_iter=100000,
            )
            p_clf = Pipeline([
                ("vectorizer", vec_factory()),
                ("classifier", clf),
            ])

            naziv = f"{pname} | {fname} | LinearSVM"
            tasks.append((
                naziv,
                pname,
                "LinearSVM",
                fname,
                p_clf,
                corpus,
                outer_cv,
                True,
            ))
            estimators[naziv] = (p_clf, corpus)

            # Isto nested-CV nacelo kao za LogRegresiju iznad: inner_cv bira
            # C, outer_cv daje finalnu, nepristrasnu ocenu.
            naziv = f"{pname} | {fname} | LinearSVM (GridSearch)"
            p_grid_svm = {"classifier__C": MREZA_C}
            gs_clf = GridSearchCV(
                estimator=p_clf,
                param_grid=p_grid_svm,
                cv=inner_cv,
                scoring="accuracy",
                n_jobs=n_jobs,
            )
            tasks.append((
                naziv,
                pname,
                "LinearSVM (GridSearch)",
                fname,
                gs_clf,
                corpus,
                outer_cv,
                True,
            ))
            estimators[naziv] = (gs_clf, corpus)

    # --- Izvrsavanje konfiguracija paralelno preko BROJ_PROCESA OS procesa -----
    total = len(tasks)
    t0 = time.time()
    print(f"\nPokrecem {total} konfiguracija na {BROJ_PROCESA} procesa, "
         f"{n_jobs} jezgro/a po procesu...")

    rows: List[Dict] = []
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=BROJ_PROCESA) as ex:
            # Svaki task je (naziv, pname, model, odlika, estimator,
            # task_corpus, cv, f1_takodje) - raspakujemo ga i predajemo
            # kao zasebne argumente _cv_report-u, u zasebnom OS procesu.
            futures = {}
            for zadatak in tasks:
                naziv, pname, model, odlika, estimator, task_corpus, cv, f1_takodje = zadatak
                buduci_rezultat = ex.submit(
                    _cv_report,
                    naziv,
                    pname,
                    model,
                    odlika,
                    estimator,
                    task_corpus,
                    y,
                    cv,
                    f1_takodje,
                    n_jobs,
                )
                futures[buduci_rezultat] = naziv

            for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
                row = fut.result()
                rows.append(row)
                f1_txt = f", macro-F1 = {row['cv_f1_sr']:.4f}" if row["cv_f1_sr"] is not None else ""
                print(f"[{i:>3}/{total}] {row['konfiguracija']:<65} "
                     f"accuracy = {row['cv_accuracy_sr']:.4f}{f1_txt} "
                     f"({row['vreme_s']:.1f}s)", flush=True)
    except concurrent.futures.process.BrokenProcessPool:
        # Neki worker proces je oboren (najcesce OS-nivo, npr. crash u
        # nativnoj biblioteci) - konfiguracija koja je bas bila u obradi se
        # ne moze pouzdano identifikovati (proces je umro bez rezultata), ali
        # rezultati vec zavrsenih konfiguracija se ne odbacuju.
        print(f"\nUPOZORENJE: proces u pool-u je prekinut (BrokenProcessPool) "
             f"nakon {len(rows)}/{total} zavrsenih konfiguracija. Cuvam "
             f"delimicne rezultate i prekidam.", file=sys.stderr)
        if rows:
            delimicni_rezultati = pd.DataFrame(rows)
            delimicni_rezultati = delimicni_rezultati.drop(columns=["fold_accuracy"])
            delimicni_rezultati.to_csv(
                outdir / "rezultati_osnovni_modeli_notebook_stil_DELIMICNO.csv",
                index=False,
                encoding="utf-8",
            )
        raise

    print(f"\nUkupno vreme: {(time.time() - t0) / 60:.1f} min")

    # --- Rezultati -------------------------------------------------------------
    res = pd.DataFrame(rows).sort_values("cv_f1_sr", ascending=False)

    res_za_csv = res.drop(columns=["fold_accuracy"])
    res_za_csv.to_csv(
        outdir / "rezultati_osnovni_modeli_notebook_stil.csv",
        index=False,
        encoding="utf-8",
    )

    top_15 = res.head(15)[["konfiguracija", "cv_accuracy_sr", "cv_f1_sr", "cv_f1_std"]]
    print(f"\n=== TOP 15 KONFIGURACIJA (rangirano po CV macro-F1, od {len(res)}) ===")
    print(top_15.to_string(index=False))

    _najbolji_model(res, estimators, y, labels, outdir)
    _grafikon(res, outdir)
    _heatmapa(res, outdir)

    print(f"\nRezultati sacuvani u: {outdir}/")


def _najbolji_model(res: pd.DataFrame, estimators: Dict[str, Tuple[object, List[str]]],
                    y, labels: List[str], outdir: Path):
    """Matrica konfuzije i classification_report za najbolju konfiguraciju,
    racunati preko cross_val_predict (van-uzorka predikcije) - bez odvojenog
    test skupa, isto kao i ostatak evaluacije.

    Parametri:
        res: DataFrame rezultata svih konfiguracija, sortiran po cv_f1_sr -
            uzima se prvi red (res.iloc[0]) kao najbolja konfiguracija.
        estimators: recnik {naziv: (estimator, corpus)} - koristi se da se
            za naziv najbolje konfiguracije dohvati konkretan (nefitovan)
            estimator i njegov corpus, radi ponovnog cross_val_predict-a.
        y: serija labela - target za classification_report i matricu
            konfuzije.
        labels: lista mogucih klasa (npr. ["negative","neutral","positive"])
            - redosled/oznake za classification_report i ose matrice
            konfuzije.
        outdir: Path direktorijuma u koji se cuva matrica_konfuzije.png.
    """
    best_naziv = res.iloc[0]["konfiguracija"]
    best_est, best_corpus = estimators[best_naziv]

    print(f"\n=== NAJBOLJA KONFIGURACIJA: {best_naziv} ===")
    cv = StratifiedKFold(
        n_splits=BROJ_FOLDOVA,
        shuffle=True,
        random_state=SLUCAJNO_SEME,
    )
    pred_cv = cross_val_predict(best_est, best_corpus, y, cv=cv)
    izvestaj = classification_report(
        y, pred_cv, labels=labels, digits=3, zero_division=0,
    )
    print(izvestaj)

    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y, pred_cv, labels=labels)
    cm_prikaz = ConfusionMatrixDisplay(cm, display_labels=labels)
    cm_prikaz.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(f"Matrica konfuzije (unakrsna validacija)\n{best_naziv}")
    ax.set_xlabel("predvidjeno")
    ax.set_ylabel("stvarno")
    fig.tight_layout()
    fig.savefig(outdir / "matrica_konfuzije.png", dpi=150)


def _heatmapa(res: pd.DataFrame, outdir: Path):
    """Toplotna mapa pretprocesiranje x odlike, po modelu - isto sto i
    toplotna_mapa.png u 02_osnovni_modeli.py.

    Parametri:
        res: DataFrame svih rezultata - pivotira se po pretprocesiranje/
            odlika/cv_f1_sr, odvojeno za svaki model (jedna podmapa po
            modelu).
        outdir: Path direktorijuma u koji se cuva toplotna_mapa.png.
    """
    modeli = sorted(res.model.unique())
    fig, axes = plt.subplots(1, len(modeli), figsize=(6 * len(modeli), 4.5),
                             squeeze=False)
    for ax, m in zip(axes[0], modeli):
        piv = res[res.model == m].pivot_table(
            index="pretprocesiranje", columns="odlika", values="cv_f1_sr")
        im = ax.imshow(piv.values, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels(piv.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels(piv.index)
        ax.set_title(m)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                if not pd.isna(v):
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                           color="w", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Macro-F1: efekat pretprocesiranja i odlika")
    fig.tight_layout()
    fig.savefig(outdir / "toplotna_mapa.png", dpi=150)


def _grafikon(res: pd.DataFrame, outdir: Path):
    """Horizontalni bar-grafikon top 15 konfiguracija po macro-F1 (sa
    error-bar-ovima std).

    Parametri:
        res: DataFrame rezultata - uzima se res.head(15) (vec sortirano po
            cv_f1_sr) i redosled se obrce radi prikaza (najbolji na vrhu).
        outdir: Path direktorijuma u koji se cuva top_konfiguracije.png.
    """
    top = res.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top.konfiguracija, top.cv_f1_sr, xerr=top.cv_f1_std,
           color="#4c72b0", error_kw=dict(ecolor="#333", capsize=3))
    ax.set_xlabel("macro-F1 (unakrsna validacija)")
    ax.set_title("Najbolje konfiguracije (notebook stil)")
    ax.set_xlim(left=max(0, top.cv_f1_sr.min() - 0.15))
    fig.tight_layout()
    fig.savefig(outdir / "top_konfiguracije.png", dpi=150)


if __name__ == "__main__":
    main()
