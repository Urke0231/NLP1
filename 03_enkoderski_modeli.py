"""
Faza 3b - testiranje efikasnosti Ollama modela za multilingvalne recenicne
vektore ("paraphrase-multilingual"), koji radi lokalno na localhost:11434.

Ovaj model se NE fino podesava - on samo pretvara tekst u fiksni vektor
(embedding). Njegova efikasnost se zato meri posredno: iznad zamrznutih
vektora treniraju se jednostavni klasifikatori (logisticka regresija,
linearni i RBF SVM) i poredi se njihov skor sa osnovnim TF-IDF modelima
iz 02_osnovni_modeli.py.

Protokol evaluacije (isti kao u 02_osnovni_modeli.py):
  - 1/10 podataka se izdvaja kao test skup i ne koristi se ni za treniranje
    ni za podesavanje hiperparametara - sluzi samo za finalnu, nepristrasnu
    procenu.
  - Preostalih 9/10 se deli na 5 delova (5-slojna stratifikovana unakrsna
    validacija) - koristi se i za podesavanje hiperparametara (GridSearchCV)
    i za procenu stabilnosti modela.

Pokretanje:
  1) Pokrenuti Ollama servis (podrazumevano vec sluza na :11434):
       ollama serve
  2) Uveriti se da je model povucen:
       ollama pull paraphrase-multilingual
  3) python 03_enkoderski_modeli.py
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
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

from data import load_dataset, present_labels
from preprocessing import basic_clean

# =============================================================================
#                              K O N F I G U R A C I J A
# =============================================================================

# --- OBAVEZNO ----------------------------------------------------------------

PUTANJA_PODACI = "./anotacije-2026-07-26.json"
IZLAZNI_DIREKTORIJUM = "rezultati_enkoder_konacno"

# --- OLLAMA --------------------------------------------------------------

OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "paraphrase-multilingual"
OLLAMA_BATCH = 16   # koliko tekstova ide u jedan HTTP zahtev (/api/embed)
OLLAMA_NITI = 4     # broj paralelnih HTTP zahteva ka Ollama-i

# Vektori se kesiraju u .cache/, pa ponovno pokretanje ne saljе iste tekstove
# ponovo Ollama-i. Postavite na True da se kes zanemari.
IGNORISI_KES = False

# --- STA SE POREDI -------------------------------------------------------

# Klasifikatori koji se treniraju IZNAD zamrznutih Ollama vektora.
# "Vecinski" je baseline i racuna se samo jednom. Zakomentarisite red da ga
# izbacite iz poredjenja.
MODELI = [
    "Vecinski",
    "LogRegresija",
    "LinearSVM",
    "RBF-SVM",
]

# --- PROTOKOL EVALUACIJE (isti kao 02_osnovni_modeli.py) -----------------

# 1/10 podataka se izdvaja kao test skup - ne koristi se ni za treniranje ni
# za podesavanje hiperparametara.
VELICINA_TEST = 0.1

# Preostalih 9/10 se deli na 5 delova: 4 za trening, 1 za validaciju, rotira
# se (5-slojna unakrsna validacija) - koristi se za GridSearchCV i za procenu
# stabilnosti.
BROJ_VALIDACIONIH_FOLDOVA = 5

SLUCAJNO_SEME = 42   # isti seed => isti test skup i isti foldovi za sve modele

MREZA_C = [0.01, 0.1, 1, 10, 100]              # LogRegresija i LinearSVM
MREZA_C_RBF = [0.1, 1, 10, 100]                # RBF-SVM
MREZA_GAMMA_RBF = ["scale", 0.01, 0.001]       # RBF-SVM

# =============================================================================
#                    kraj konfiguracije - ispod ne treba menjati
# =============================================================================

CACHE = Path(__file__).parent / ".cache"
CACHE.mkdir(exist_ok=True)


def _http_post(path: str, payload: dict, timeout: int = 120) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_URL}{path}", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _embed_batch(texts: List[str]) -> List[List[float]]:
    """Pokusava batch endpoint /api/embed; ako ga verzija Ollama-e nema,
    prelazi na pojedinacne pozive /api/embeddings. Nekoliko pokusaja sa
    pauzom, jer lokalni server ume da bude privremeno zauzet."""
    for attempt in range(4):
        try:
            out = _http_post("/api/embed",
                             {"model": OLLAMA_MODEL, "input": texts})
            return out["embeddings"]
        except (urllib.error.HTTPError, KeyError):
            break  # batch endpoint ne postoji - probaj pojedinacne pozive
        except urllib.error.URLError:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)

    embs = []
    for t in texts:
        for attempt in range(4):
            try:
                out = _http_post("/api/embeddings",
                                 {"model": OLLAMA_MODEL, "prompt": t})
                embs.append(out["embedding"])
                break
            except urllib.error.URLError:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
    return embs


def get_embeddings(texts: List[str]) -> np.ndarray:
    """Vraca matricu (broj_tekstova x dimenzija) Ollama vektora. Kesira se na
    disku jer su HTTP pozivi mnogo sporiji od ucitavanja fajla."""
    key = hashlib.md5(
        (OLLAMA_MODEL + "|" + json.dumps(texts, ensure_ascii=False))
        .encode("utf-8")).hexdigest()
    cache_file = CACHE / f"ollama_emb_{key}.npy"
    if cache_file.exists() and not IGNORISI_KES:
        print(f"Ucitavam kesirane vektore: {cache_file.name}")
        return np.load(cache_file)

    try:
        urllib.request.urlopen(OLLAMA_URL, timeout=5)
    except urllib.error.URLError:
        sys.exit(f"GRESKA: Ollama nije dostupna na {OLLAMA_URL}.\n"
                 f"  Pokrenite je sa 'ollama serve' i uverite se da je model "
                 f"povucen: 'ollama pull {OLLAMA_MODEL}'")

    print(f"Racunam vektore preko Ollama ({OLLAMA_MODEL}) za {len(texts)} "
         f"tekstova...")
    chunks = [texts[i:i + OLLAMA_BATCH] for i in range(0, len(texts), OLLAMA_BATCH)]
    results = [None] * len(chunks)
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=OLLAMA_NITI) as ex:
        futures = {ex.submit(_embed_batch, ch): i for i, ch in enumerate(chunks)}
        done = 0
        for fut in concurrent.futures.as_completed(futures):
            results[futures[fut]] = fut.result()
            done += 1
            print(f"  {done}/{len(chunks)} paketa obradjeno", end="\r", flush=True)
    print()

    emb = np.array([v for chunk in results for v in chunk], dtype=np.float32)
    print(f"Gotovo za {time.time() - t0:.1f}s. Dimenzija vektora: {emb.shape[1]}")
    np.save(cache_file, emb)
    return emb


def _model_defs():
    return {
        "Vecinski": (DummyClassifier(strategy="most_frequent"), {}, False),
        "LogRegresija": (
            LogisticRegression(max_iter=3000, class_weight="balanced",
                               random_state=SLUCAJNO_SEME),
            {"clf__C": MREZA_C}, True),
        "LinearSVM": (
            LinearSVC(class_weight="balanced", random_state=SLUCAJNO_SEME,
                      dual=False, max_iter=20000),
            {"clf__C": MREZA_C}, True),
        "RBF-SVM": (
            SVC(kernel="rbf", class_weight="balanced", random_state=SLUCAJNO_SEME),
            {"clf__C": MREZA_C_RBF, "clf__gamma": MREZA_GAMMA_RBF}, True),
    }


def evaluate_model(mname: str, clf, grid: dict, use_scaler: bool,
                   X: np.ndarray, y: np.ndarray, train_idx: np.ndarray,
                   test_idx: np.ndarray, labels: List[str]):
    """Podesava (ako ima mreze hiperparametara) i ocenjuje jedan klasifikator
    nad zamrznutim Ollama vektorima, po istom protokolu kao osnovni modeli:
    5-slojna CV na 90% podataka, jednom finalno na izdvojenih 10%."""
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    cv = StratifiedKFold(n_splits=BROJ_VALIDACIONIH_FOLDOVA, shuffle=True,
                         random_state=SLUCAJNO_SEME)
    steps = [("clf", clone(clf))]
    if use_scaler:
        steps.insert(0, ("scale", StandardScaler()))
    pipe = Pipeline(steps)

    t = time.time()
    if grid:
        search = GridSearchCV(pipe, grid, scoring="f1_macro", cv=cv, n_jobs=-1,
                              refit=True)
        search.fit(X_tr, y_tr)
        est, best = search.best_estimator_, search.best_params_
        idx = search.best_index_
        fold_skorovi = [search.cv_results_[f"split{k}_test_score"][idx]
                        for k in range(BROJ_VALIDACIONIH_FOLDOVA)]
    else:
        fold_skorovi = cross_val_score(pipe, X_tr, y_tr, scoring="f1_macro",
                                       cv=cv, n_jobs=-1).tolist()
        pipe.fit(X_tr, y_tr)
        est, best = pipe, {}

    fold_skorovi = np.array(fold_skorovi)
    pred_test = est.predict(X_te)

    row = {
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
    return row, pred_test


def main():
    put = Path(PUTANJA_PODACI)
    if not put.exists():
        sys.exit(f"GRESKA: ne postoji fajl sa podacima: {put.resolve()}\n"
                 f"Ispravite PUTANJA_PODACI na vrhu skripte.")

    outdir = Path(IZLAZNI_DIREKTORIJUM)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(put)
    labels = present_labels(df)
    texts = [basic_clean(t) for t in df["tekst"]]
    y = df["sentiment"].to_numpy()
    print(f"Ucitano {len(df)} primera, klase: {labels}")

    X = get_embeddings(texts)

    train_idx, test_idx = train_test_split(
        np.arange(len(y)), test_size=VELICINA_TEST, stratify=y,
        random_state=SLUCAJNO_SEME)
    print(f"Test skup: {len(test_idx)} primera ({VELICINA_TEST:.0%}), "
         f"trening+validacija: {len(train_idx)} primera "
         f"({BROJ_VALIDACIONIH_FOLDOVA}-slojna krosvalidacija)")

    models = {k: v for k, v in _model_defs().items() if k in MODELI}
    if not models:
        sys.exit("GRESKA: prazna lista modela. Proverite konfiguraciju.")

    rows, test_pred_store = [], {}
    t0 = time.time()
    for mname, (clf, grid, use_scaler) in models.items():
        row, pred_test = evaluate_model(mname, clf, grid, use_scaler, X, y,
                                        train_idx, test_idx, labels)
        test_pred_store[mname] = pred_test
        rows.append(row)
        print(f"{mname:<15} CV macro-F1 = {row['cv_macroF1_sr']:.4f} "
             f"+/- {row['cv_macroF1_std']:.4f} | TEST macro-F1 = "
             f"{row['test_macroF1']:.4f} ({row['vreme_s']:.1f}s)")

    res = pd.DataFrame(rows).sort_values("cv_macroF1_sr", ascending=False)
    res.drop(columns=["fold_skorovi"]).to_csv(
        outdir / "rezultati_enkoderi.csv", index=False, encoding="utf-8")
    (outdir / "enkoderi_fold_skorovi.json").write_text(
        json.dumps({r["model"]: r["fold_skorovi"] for r in rows}, indent=2),
        encoding="utf-8")

    print(f"\nUkupno vreme: {(time.time() - t0) / 60:.1f} min")
    print(f"\n=== REZULTATI (Ollama model: {OLLAMA_MODEL}) ===")
    print(res[["model", "cv_macroF1_sr", "cv_macroF1_std",
               "test_macroF1", "test_tacnost"]].to_string(index=False))

    _analiza_najboljeg(res, test_pred_store, y, test_idx, labels, outdir)
    _statisticko_poredjenje(rows, outdir)
    _grafikon(res, outdir)


def _analiza_najboljeg(res, test_pred_store, y, test_idx, labels, outdir: Path):
    best = res.iloc[0]
    pred_test = test_pred_store[best.model]
    y_test = y[test_idx]

    print(f"\n=== NAJBOLJI MODEL: {best.model} ===")
    print(classification_report(y_test, pred_test, labels=labels, digits=3,
                                zero_division=0))

    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_test, pred_test, labels=labels)
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(f"Matrica konfuzije (test skup)\n{OLLAMA_MODEL} + {best.model}")
    ax.set_xlabel("predvidjeno")
    ax.set_ylabel("stvarno")
    fig.tight_layout()
    fig.savefig(outdir / "enkoderi_matrica_konfuzije.png", dpi=150)


def _statisticko_poredjenje(rows, outdir: Path):
    """Uparen Wilcoxonov test po foldovima, u odnosu na najbolji model.
    Foldovi su identicni za sve modele (isto SLUCAJNO_SEME), pa je uparen
    test opravdan."""
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        print("(za statisticko poredjenje: pip install scipy)")
        return

    rows = [r for r in rows if r["model"] != "Vecinski"]
    if len(rows) < 2:
        return
    rows = sorted(rows, key=lambda r: -r["cv_macroF1_sr"])
    base = rows[0]
    out = []
    for r in rows[1:]:
        a, b = np.array(base["fold_skorovi"]), np.array(r["fold_skorovi"])
        p = 1.0 if np.allclose(a, b) else wilcoxon(a, b).pvalue
        out.append({
            "model": r["model"],
            "cv_macroF1": round(r["cv_macroF1_sr"], 4),
            "razlika": round(base["cv_macroF1_sr"] - r["cv_macroF1_sr"], 4),
            "p_vrednost": round(p, 4),
            "znacajno_p<0.05": p < 0.05,
        })
    dfp = pd.DataFrame(out)
    dfp.to_csv(outdir / "enkoderi_statisticko_poredjenje.csv", index=False,
              encoding="utf-8")
    print(f"\n=== POREDJENJE SA NAJBOLJIM MODELOM ({base['model']}, Wilcoxon) ===")
    print(dfp.to_string(index=False))


def _grafikon(res: pd.DataFrame, outdir: Path):
    top = res.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(top.model, top.cv_macroF1_sr, xerr=top.cv_macroF1_std, color="#4c72b0",
            error_kw=dict(ecolor="#333", capsize=3))
    ax.set_xlabel(f"macro-F1 ({BROJ_VALIDACIONIH_FOLDOVA}-slojna CV)")
    ax.set_title(f"Klasifikatori nad Ollama vektorima ({OLLAMA_MODEL})")
    fig.tight_layout()
    fig.savefig(outdir / "enkoderi_poredjenje.png", dpi=150)
    print(f"\nGrafikoni sacuvani u: {outdir}/")


if __name__ == "__main__":
    main()
