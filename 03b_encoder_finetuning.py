"""
Faza 3b - enkoderski jezicki modeli, PRAVO fino podesavanje (fine-tuning).

Za razliku od 03_enkoderski_modeli.py (koji koristi ZAMRZNUTE recenicne
vektore iz Ollama-e i iznad njih trenira klasicne klasifikatore), ova
skripta stvarno dotreniava tezine samog enkoder modela (BERT-olikog
transformera) na anotiranom skupu - to je "fine-tuning" u uzem smislu i
odgovara originalnom zahtevu iz README.md ("bar jedan monolingvalni i bar
jedan visejezicni model").

Protokol: 10-slojna stratifikovana unakrsna validacija; unutar svakog folda
model se evaluira POSLE SVAKE EPOHE, pa se dobija kriva
"broj epoha -> macro-F1" koju trazi postavka projekta. Nema posebno
izdvojenog test skupa (za razliku od 02/03) - to je namerno, jer bi na
malom anotiranom skupu dodatno cepanje ostavilo premalo primera po foldu za
fino podesavanje transformera. Ogranicenje: broj epoha se bira post hoc na
osnovu ovih istih foldova - to obavezno navesti u izvestaju (vidi README.md).

Rucna petlja obucavanja (bez Trainer-a) da bi skripta radila nezavisno od
verzije biblioteke transformers.

Pokretanje (NVIDIA GPU ili CPU):
  1) pip install torch transformers
  2) python 03b_encoder_finetuning.py
Nema parametara iz komandne linije - sve se podesava u bloku KONFIGURACIJA.
Preporuka: Google Colab sa GPU-om ili studentski krediti na MS Azure - na
CPU-u fino podesavanje transformera traje satima.

Pokretanje (Windows + AMD GPU, npr. RX 7900 XT) - PROVERENO RADI:
  Od ROCm 7.2.1 postoje zvanicni PyTorch wheel-ovi za NATIVE Windows (ne
  treba vise WSL2). Rade samo sa Python-om 3.12 (cp312), zato poseban
  .venv312 pored ostalih virtuelnih okruzenja. NAPOMENA: RX 7900 XT (RDNA3)
  zvanicno NIJE na AMD-ovoj listi podrzanih GPU-ova za Windows (samo RDNA4
  je zvanicno podrzan) - u praksi ipak radi (testirano: torch.cuda prepoznaje
  karticu, treniranje na GPU-u prolazi).
    1) py -3.12 -m venv .venv312
    2) .venv312\\Scripts\\python.exe -m pip install --no-cache-dir ^
         https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_core-7.2.1-py3-none-win_amd64.whl ^
         https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl ^
         https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl ^
         https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm-7.2.1.tar.gz
    3) .venv312\\Scripts\\python.exe -m pip install --no-cache-dir ^
         "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl" ^
         "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torchaudio-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl" ^
         "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torchvision-0.24.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl"
    4) .venv312\\Scripts\\python.exe -m pip install "transformers==4.46.3" pandas matplotlib scikit-learn scipy
       VAZNO: ova ROCm-Windows verzija PyTorch-a je bez torch.distributed
       podrske. transformers >= 5.x pri importu AutoModelForSequenceClassification
       bezuslovno pokusava da ucita torch.distributed.fsdp i puca sa
       "ModuleNotFoundError: torch._C._distributed_c10d". transformers 4.46.3
       je poslednja poznata verzija bez tog problema - NE nadogradjivati.
    5) .venv312\\Scripts\\python.exe 03b_encoder_finetuning.py
  Skripta ne zahteva izmene koda za ROCm - torch.cuda.is_available() ROCm
  uredjaj automatski prijavljuje kao "cuda" (ispis "Uredjaj:" na startu
  razlikuje CUDA/ROCm/CPU i ispisuje naziv GPU-a).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedKFold

from data import load_dataset, present_labels
from preprocessing import basic_clean

# =============================================================================
#                              K O N F I G U R A C I J A
# =============================================================================

# --- OBAVEZNO ----------------------------------------------------------------

PUTANJA_PODACI = "./opj_dataset_no_duplicates.json"
IZLAZNI_DIREKTORIJUM = "sasvim_novi_rezultati_finetuning_konacno2"

# --- IZBOR MODELA ------------------------------------------------------------

# Postavka trazi bar jedan monolingvalni i bar jedan visejezicni model.
# Zakomentarisite red da biste model izbacili iz eksperimenta.
MODELI = [
    "classla/bcms-bertic",           # MONOLINGVALNI (BCMS), ELECTRA, ~110M par.
    "xlm-roberta-base",              # VISEJEZICNI, primarni izbor, 278M par.
    "bert-base-multilingual-cased",  # VISEJEZICNI, referentna tacka
    # "classla/xlm-r-bertic",        # XLM-R-large dotreniran na juznoslovenskim
    #                                # jezicima; trazi ~24 GB VRAM-a, na
    #                                # besplatnom Colabu najcesce ne staje
]

# --- HIPERPARAMETRI FINOG PODESAVANJA ----------------------------------------

# Maksimalan broj epoha. Model se evaluira posle svake, pa ovo istovremeno
# odredjuje koliko tacaka ima kriva "epohe -> macro-F1".
BROJ_EPOHA = 5

# Broj foldova unakrsne validacije. Postavka trazi 10.
# Za brzu probu privremeno spustite na 2 ili 3.
BROJ_FOLDOVA = 10

# Velicina batcha. Ako dobijete CUDA out of memory, prepolovite na 8 ili 4.
VELICINA_BATCHA = 16

# Stopa ucenja. Uobicajen opseg za fino podesavanje BERT-olikih modela:
# 1e-5 do 5e-5.
STOPA_UCENJA = 2e-5

# Maksimalna duzina sekvence u podrec-tokenima. 256 obicno pokriva par
# recenica; smanjenje na 128 znacajno ubrzava obucavanje.
MAKSIMALNA_DUZINA = 128

# Da li se u funkciji gubitka koriste tezine klasa (preporuceno kod
# neuravnotezenog sentimenta).
KORISTI_TEZINE_KLASA = True

SLUCAJNO_SEME = 42

# =============================================================================
#                    kraj konfiguracije - ispod ne treba menjati
# =============================================================================


def set_seed(s: int):
    import random
    import torch
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def make_loader(texts, labels, tokenizer, shuffle):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    enc = tokenizer(list(texts), truncation=True, padding="max_length",
                    max_length=MAKSIMALNA_DUZINA, return_tensors="pt")
    ds = TensorDataset(enc["input_ids"], enc["attention_mask"],
                       torch.tensor(labels, dtype=torch.long))
    return DataLoader(ds, batch_size=VELICINA_BATCHA, shuffle=shuffle)


def fine_tune_fold(model_name, X_tr, y_tr, X_te, y_te, n_labels, device):
    """Obucava jedan fold i vraca (po-epoha macro-F1, po-epoha predikcije)."""
    import torch
    from torch.optim import AdamW
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              get_linear_schedule_with_warmup)

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=n_labels).to(device)

    tr = make_loader(X_tr, y_tr, tok, shuffle=True)
    te = make_loader(X_te, y_te, tok, shuffle=False)

    opt = AdamW(model.parameters(), lr=STOPA_UCENJA, weight_decay=0.01)
    total = len(tr) * BROJ_EPOHA
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)

    if KORISTI_TEZINE_KLASA:
        counts = np.bincount(y_tr, minlength=n_labels).astype(float)
        w = torch.tensor(len(y_tr) / (n_labels * np.maximum(counts, 1)),
                         dtype=torch.float, device=device)
        lossf = torch.nn.CrossEntropyLoss(weight=w)
    else:
        lossf = torch.nn.CrossEntropyLoss()

    per_epoch_f1, per_epoch_pred = [], []
    for ep in range(1, BROJ_EPOHA + 1):
        model.train()
        for ids, mask, y in tr:
            ids, mask, y = ids.to(device), mask.to(device), y.to(device)
            opt.zero_grad()
            logits = model(input_ids=ids, attention_mask=mask).logits
            lossf(logits, y).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

        model.eval()
        preds = []
        with torch.no_grad():
            for ids, mask, _ in te:
                logits = model(input_ids=ids.to(device),
                               attention_mask=mask.to(device)).logits
                preds.extend(logits.argmax(-1).cpu().numpy().tolist())
        f1 = f1_score(y_te, preds, average="macro", zero_division=0)
        per_epoch_f1.append(f1)
        per_epoch_pred.append(preds)
        print(f"    epoha {ep}: macro-F1 = {f1:.4f}", flush=True)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return per_epoch_f1, per_epoch_pred


def main():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        sys.exit("GRESKA: nedostaju biblioteke.\n"
                 "  pip install torch transformers")

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        backend = "ROCm (AMD)" if getattr(torch.version, "hip", None) else "CUDA (NVIDIA)"
        print(f"Uredjaj: {device} [{backend}] - {torch.cuda.get_device_name(0)}")
    else:
        print(f"Uredjaj: {device}")
        print("UPOZORENJE: bez GPU-a ovo traje satima. "
              "Koristite Google Colab, MS Azure ili WSL2+ROCm (vidi docstring).")

    put = Path(PUTANJA_PODACI)
    if not put.exists():
        sys.exit(f"GRESKA: ne postoji fajl sa podacima: {put.resolve()}\n"
                 f"Ispravite PUTANJA_PODACI na vrhu skripte.")

    outdir = Path(IZLAZNI_DIREKTORIJUM)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(put,drop_duplicates=True)
    labels = present_labels(df)
    lab2id = {l: i for i, l in enumerate(labels)}
    X = np.array([basic_clean(t) for t in df["tekst"]], dtype=object)
    y = np.array([lab2id[s] for s in df["sentiment"]])
    print(f"{len(X)} primera, {len(labels)} klasa: {labels}")
    print(f"Prosecan broj reci: {df['broj_reci'].mean():.1f}, "
          f"maksimum: {df['broj_reci'].max()} "
          f"(MAKSIMALNA_DUZINA = {MAKSIMALNA_DUZINA} podrec-tokena)")

    if not MODELI:
        sys.exit("GRESKA: prazna lista modela. Proverite konfiguraciju.")

    skf = StratifiedKFold(n_splits=BROJ_FOLDOVA, shuffle=True,
                          random_state=SLUCAJNO_SEME)
    rezultati, oof_all = [], {}

    for mname in MODELI:
        print(f"\n{'=' * 70}\nMODEL: {mname}\n{'=' * 70}")
        set_seed(SLUCAJNO_SEME)
        fold_f1 = np.zeros((BROJ_FOLDOVA, BROJ_EPOHA))
        oof = np.full((BROJ_EPOHA, len(y)), -1, dtype=int)
        t0 = time.time()

        for k, (tr, te) in enumerate(skf.split(X, y), start=1):
            print(f"  Fold {k}/{BROJ_FOLDOVA}", flush=True)
            f1s, preds = fine_tune_fold(mname, X[tr], y[tr], X[te], y[te],
                                        len(labels), device)
            fold_f1[k - 1] = f1s
            for e in range(BROJ_EPOHA):
                oof[e, te] = preds[e]

        mean_f1, std_f1 = fold_f1.mean(axis=0), fold_f1.std(axis=0)
        best_ep = int(np.argmax(mean_f1)) + 1
        print(f"\n  Najbolji broj epoha: {best_ep} "
              f"(macro-F1 = {mean_f1[best_ep - 1]:.4f} "
              f"+/- {std_f1[best_ep - 1]:.4f})")
        print(f"  Vreme: {(time.time() - t0) / 60:.1f} min")

        for e in range(BROJ_EPOHA):
            rezultati.append({
                "model": mname, "epohe": e + 1,
                "macroF1_sr": mean_f1[e], "macroF1_std": std_f1[e],
                "tacnost": (oof[e] == y).mean(),
                "fold_skorovi": fold_f1[:, e].tolist(),
            })
        oof_all[mname] = oof[best_ep - 1]
        print(classification_report(y, oof[best_ep - 1], target_names=labels,
                                    digits=3, zero_division=0))

    res = pd.DataFrame(rezultati)
    res.drop(columns=["fold_skorovi"]).to_csv(
        outdir / "rezultati_finetuning.csv", index=False, encoding="utf-8")
    (outdir / "finetuning_fold_skorovi.json").write_text(
        json.dumps(rezultati, indent=2, default=float), encoding="utf-8")
    np.savez(outdir / "finetuning_oof.npz", y=y, labels=np.array(labels), **oof_all)

    print("\n=== SVI REZULTATI ===")
    print(res[["model", "epohe", "macroF1_sr", "macroF1_std"]].to_string(index=False))
    _grafikoni(res, oof_all, y, labels, outdir)


def _grafikoni(res, oof_all, y, labels, outdir: Path):
    # 1) Kriva: broj epoha -> macro-F1, po modelu
    fig, ax = plt.subplots(figsize=(8, 5))
    for m, g in res.groupby("model"):
        ax.errorbar(g.epohe, g.macroF1_sr, yerr=g.macroF1_std,
                    marker="o", capsize=3, label=m.split("/")[-1])
    ax.set_xlabel("broj epoha finog podesavanja")
    ax.set_ylabel("macro-F1 (unakrsna validacija)")
    ax.set_title("Uticaj duzine finog podesavanja")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "finetuning_epohe.png", dpi=150)

    # 2) Najbolji rezultat po modelu
    best = res.loc[res.groupby("model").macroF1_sr.idxmax()]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh([m.split("/")[-1] for m in best.model], best.macroF1_sr,
            xerr=best.macroF1_std, color="#c44e52",
            error_kw=dict(ecolor="#333", capsize=3))
    ax.set_xlabel("macro-F1")
    ax.set_title("Najbolja varijanta po modelu")
    fig.tight_layout()
    fig.savefig(outdir / "finetuning_poredjenje.png", dpi=150)

    # 3) Matrice konfuzije
    n = len(oof_all)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), squeeze=False)
    for ax, (m, pred) in zip(axes[0], oof_all.items()):
        cm = confusion_matrix(y, pred, labels=range(len(labels)))
        ConfusionMatrixDisplay(cm, display_labels=labels).plot(
            ax=ax, cmap="Reds", colorbar=False, values_format="d")
        ax.set_title(m.split("/")[-1])
    fig.tight_layout()
    fig.savefig(outdir / "finetuning_matrice_konfuzije.png", dpi=150)
    print(f"\nGrafikoni sacuvani u: {outdir}/")


if __name__ == "__main__":
    main()
