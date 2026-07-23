"""
Faza 2 - deskriptivna statistika anotiranih podataka + slaganje anotatora.

Pokretanje:  python 01_statistika.py
Nema parametara iz komandne linije - sve se podesava u bloku KONFIGURACIJA.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from data import load_dataset, present_labels
from preprocessing import basic_clean, tokenize

# =============================================================================
#                              K O N F I G U R A C I J A
#            (jedino ovo menjate; putanje su relativne u odnosu na
#             direktorijum iz koga pokrecete skriptu)
# =============================================================================

# --- OBAVEZNO ----------------------------------------------------------------

# Putanja do glavnog anotiranog skupa. JSON lista objekata ili JSONL.
# Obavezna polja: "tekst", "sentiment". Opciona: "duzina", "uneo".
PUTANJA_PODACI = "./anotacije-2026-07-23 (2).json"

# Direktorijum u koji se upisuju tabele i grafikoni. Kreira se automatski.
IZLAZNI_DIREKTORIJUM = "rezultati"

# --- OPCIONO -----------------------------------------------------------------

# Kalibracioni skup: po jedan fajl za SVAKOG clana grupe, isti tekstovi
# anotirani nezavisno. Ostavite praznu listu [] ako jos nemate kalibraciju.
# Naziv fajla se koristi kao ime anotatora u izvestaju.
PUTANJE_KALIBRACIJA = [
    # "podaci/kalibracija_anotator1.json",
    # "podaci/kalibracija_anotator2.json",
    # "podaci/kalibracija_anotator3.json",
    # "podaci/kalibracija_anotator4.json",
]

# Da li iz glavnog skupa izbaciti tekstove koji se ponavljaju.
# Postavka projekta trazi da finalni skup bude prociscen od duplikata.
UKLONI_DUPLIKATE = True

# Broj najfrekventnijih reci koji se ispisuje po klasi.
BROJ_NAJCESCIH_RECI = 15

# =============================================================================
#                    kraj konfiguracije - ispod ne treba menjati
# =============================================================================


def opisna_statistika(df: pd.DataFrame, outdir: Path) -> None:
    labels = present_labels(df)
    print("\n=== OSNOVNI PODACI ===")
    print(f"Broj primera: {len(df)}")
    print(f"Klase: {labels}")

    print("\n=== RASPODELA SENTIMENTA ===")
    dist = df["sentiment"].value_counts().reindex(labels)
    dist_pct = (dist / len(df) * 100).round(1)
    print(pd.DataFrame({"broj": dist, "%": dist_pct}))
    # Odnos vecinske i manjinske klase - kljucno za izbor metrike.
    print(f"Neuravnotezenost (max/min): {dist.max() / dist.min():.2f}")
    print(f"Tacnost vecinskog klasifikatora: {dist.max() / len(df) * 100:.1f}%")

    print("\n=== DUZINA TEKSTA ===")
    print(df[["broj_reci", "broj_karaktera"]].describe().round(1))

    if "duzina" in df.columns:
        print("\n=== RUCNA OZNAKA DUZINE ===")
        print(df["duzina"].value_counts())
        ct = pd.crosstab(df["duzina"], df["sentiment"]).reindex(
            columns=labels, fill_value=0)
        print("\nUnakrsna tabela duzina x sentiment:")
        print(ct)
        # Ako duzina korelira sa sentimentom, to je bias u podacima
        # i treba ga pomenuti u dokumentaciji.
        try:
            from scipy.stats import chi2_contingency
            chi2, p, dof, _ = chi2_contingency(ct)
            print(f"Hi-kvadrat test nezavisnosti: chi2={chi2:.2f}, p={p:.4f}")
        except ImportError:
            print("(za hi-kvadrat test: pip install scipy)")
        print("\nStvarni broj reci po rucnoj oznaci duzine:")
        print(df.groupby("duzina")["broj_reci"].describe().round(1))

    if "uneo" in df.columns:
        print("\n=== DOPRINOS PO ANOTATORU ===")
        per_ann = pd.crosstab(df["uneo"], df["sentiment"]).reindex(
            columns=labels, fill_value=0)
        per_ann["ukupno"] = per_ann.sum(axis=1)
        print(per_ann)
        # Ako se raspodela oznaka bitno razlikuje po anotatoru, to je znak
        # da uputstva za anotaciju nisu dovoljno precizna.

    print("\n=== RECNIK ===")
    tokens = [t for txt in df["tekst"] for t in tokenize(basic_clean(txt).lower())]
    vocab = pd.Series(tokens).value_counts()
    print(f"Ukupno tokena: {len(tokens)}, velicina recnika: {len(vocab)}")
    print(f"Hapax legomena (frekvencija 1): {(vocab == 1).sum()} "
          f"({(vocab == 1).sum() / len(vocab) * 100:.1f}%)")

    print(f"\n=== NAJCESCE RECI PO KLASI (top {BROJ_NAJCESCIH_RECI}) ===")
    for lab in labels:
        toks = [t for txt in df.loc[df.sentiment == lab, "tekst"]
                for t in tokenize(basic_clean(txt).lower())]
        top = pd.Series(toks).value_counts().head(BROJ_NAJCESCIH_RECI)
        print(f"{lab:>9}: {', '.join(top.index)}")

    _grafikoni(df, labels, outdir)


def _grafikoni(df: pd.DataFrame, labels, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.ravel()

    df["sentiment"].value_counts().reindex(labels).plot(
        kind="bar", ax=axes[0], color="#4c72b0", rot=0)
    axes[0].set_title("Raspodela klasa sentimenta")
    axes[0].set_ylabel("broj primera")

    df["broj_reci"].plot(kind="hist", bins=30, ax=axes[1], color="#55a868")
    axes[1].set_title("Raspodela duzine teksta (broj reci)")
    axes[1].set_xlabel("broj reci")

    df.boxplot(column="broj_reci", by="sentiment", ax=axes[2])
    axes[2].set_title("Duzina teksta po klasi")
    axes[2].set_xlabel("")
    fig.suptitle("")

    if "duzina" in df.columns:
        ct = pd.crosstab(df["duzina"], df["sentiment"]).reindex(
            columns=labels, fill_value=0)
        ct.plot(kind="bar", stacked=True, ax=axes[3], rot=0)
        axes[3].set_title("Sentiment po rucnoj oznaci duzine")
    else:
        axes[3].axis("off")

    fig.tight_layout()
    fig.savefig(outdir / "deskriptivna_statistika.png", dpi=150)
    print(f"\nGrafikoni sacuvani u {outdir / 'deskriptivna_statistika.png'}")


def slaganje_anotatora(files, outdir: Path) -> None:
    """Kalibracioni skup: po jedan fajl za svakog clana grupe, isti tekstovi."""
    ann = {}
    for f in files:
        f = Path(f)
        if not f.exists():
            print(f"UPOZORENJE: kalibracioni fajl ne postoji: {f} - preskacem.")
            continue
        d = load_dataset(f, drop_duplicates=False)
        ann[f.stem] = d.set_index("tekst")["sentiment"]

    if len(ann) < 2:
        print("Potrebna su bar dva kalibraciona fajla za racunanje slaganja.")
        return

    common = sorted(set.intersection(*[set(s.index) for s in ann.values()]))
    if not common:
        print("UPOZORENJE: kalibracioni fajlovi nemaju nijedan zajednicki tekst.")
        return
    print(f"\n=== SLAGANJE ANOTATORA (kalibracioni skup, {len(common)} primera) ===")

    names = list(ann)
    kappas = {}
    for a, b in itertools.combinations(names, 2):
        ya = [ann[a][t] for t in common]
        yb = [ann[b][t] for t in common]
        k = cohen_kappa_score(ya, yb)
        agree = sum(x == y for x, y in zip(ya, yb)) / len(common)
        kappas[(a, b)] = k
        print(f"{a:>20} vs {b:<20} Cohen kappa = {k:.3f} | "
              f"sirovo slaganje = {agree:.3f}")

    prosek = sum(kappas.values()) / len(kappas)
    print(f"\nGrupni prosek binarnih stepena saglasnosti: kappa = {prosek:.3f}")
    print("Tumacenje (Landis & Koch): <0.20 slabo, 0.21-0.40 osrednje, "
          "0.41-0.60 umereno, 0.61-0.80 znacajno, >0.80 gotovo savrseno")

    # Fleiss kappa za sve anotatore odjednom.
    try:
        from statsmodels.stats.inter_rater import aggregate_raters, fleiss_kappa
        matrix = np.array([[ann[n][t] for n in names] for t in common])
        table, _ = aggregate_raters(matrix)
        print(f"Fleiss kappa (svi anotatori): {fleiss_kappa(table):.3f}")
    except ImportError:
        print("(za Fleiss kappa: pip install statsmodels)")

    # Koje se klase najcesce mesaju - direktan input za doradu uputstava.
    a, b = names[0], names[1]
    labs = sorted({v for n in names for v in ann[n][common]})
    cm = confusion_matrix([ann[a][t] for t in common],
                          [ann[b][t] for t in common], labels=labs)
    print(f"\nMatrica konfuzije {a} (redovi) x {b} (kolone):")
    print(pd.DataFrame(cm, index=labs, columns=labs))

    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "slaganje_anotatora.json").write_text(
        json.dumps({f"{a}|{b}": v for (a, b), v in kappas.items()},
                   ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    put = Path(PUTANJA_PODACI)
    if not put.exists():
        sys.exit(f"GRESKA: ne postoji fajl sa podacima: {put.resolve()}\n"
                 f"Ispravite PUTANJA_PODACI na vrhu skripte.")

    outdir = Path(IZLAZNI_DIREKTORIJUM)
    df = load_dataset(put, drop_duplicates=UKLONI_DUPLIKATE)
    opisna_statistika(df, outdir)

    if PUTANJE_KALIBRACIJA:
        slaganje_anotatora(PUTANJE_KALIBRACIJA, outdir)
    else:
        print("\n(PUTANJE_KALIBRACIJA je prazno - preskacem analizu slaganja "
              "anotatora.)")


if __name__ == "__main__":
    main()
