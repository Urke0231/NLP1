"""
Faza 3c - dekoderski (generativni) veliki jezicki modeli.

Ovi modeli se ne obucavaju - ceo skup podataka je evaluacioni skup.
Sistematski se varira:
  1. JEZIK UPITA:  srpski vs engleski
  2. TIP UPITA:    zero-shot / zero-shot sa definicijama oznaka / few-shot

Pokretanje:  python 04_dekoderski_modeli.py
Nema parametara iz komandne linije - sve se podesava u bloku KONFIGURACIJA.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix, f1_score)

from data import load_dataset, present_labels
from preprocessing import basic_clean

# =============================================================================
#                              K O N F I G U R A C I J A
# =============================================================================

# --- OBAVEZNO ----------------------------------------------------------------

PUTANJA_PODACI = "podaci/anotirano.json"
IZLAZNI_DIREKTORIJUM = "rezultati"

# --- IZBOR MODELA ------------------------------------------------------------

# "openai" | "gemini" | "ollama" | "dummy"
#   openai -> pip install openai
#   gemini -> pip install google-genai
#   ollama -> lokalni server na http://localhost:11434 (bez API kljuca)
#   dummy  -> nasumicne oznake; sluzi da proverite da skripta radi bez trosenja
#             kredita. Rezultati su besmisleni, ne stavljajte ih u izvestaj.
BACKEND = "dummy"

# Naziv modela u okviru izabranog backend-a.
#   openai: "gpt-4o-mini", "gpt-4o"
#   gemini: "gemini-2.0-flash"
#   ollama: "qwen2.5:7b-instruct", "llama3.1:8b"
NAZIV_MODELA = "gpt-4o-mini"

# API kljuc. Ostavite prazan string ako ga postavljate kroz promenljivu
# okruzenja (OPENAI_API_KEY odnosno GOOGLE_API_KEY). Ne ostavljajte pravi
# kljuc u fajlu koji predajete uz projekat.
API_KLJUC = ""

# --- STA SE EVALUIRA ---------------------------------------------------------

# Koje varijante upita pustiti. Zakomentarisite red da izbacite varijantu.
UPITI = [
    "sr_zero",   # srpski, bez primera i bez definicija
    "sr_def",    # srpski, sa definicijama oznaka
    "sr_few",    # srpski, sa primerima (few-shot)
    "en_zero",   # engleski, bez primera i bez definicija
    "en_def",    # engleski, sa definicijama oznaka
    "en_few",    # engleski, sa primerima (few-shot)
]

# Koliko primera po klasi ide u few-shot upit. Ti primeri se automatski
# izbacuju iz evaluacionog skupa, da ne bi doslo do curenja podataka.
BROJ_FEW_SHOT_PRIMERA = 3

# Broj primera na kojima se evaluira. None = ceo skup (tako trazi postavka).
# Stavite npr. 50 dok testirate, da ne biste potrosili kredite.
VELICINA_UZORKA = None

# Pauza izmedju poziva u sekundama. Povecajte ako dobijate rate-limit greske.
PAUZA_IZMEDJU_POZIVA = 0.0

# Odgovori se kesiraju u .cache/, pa ponovno pokretanje iste kombinacije
# (model + upit + velicina skupa) ne trosi kredite. Postavite na True da se
# kes zanemari i svi pozivi ponove.
IGNORISI_KES = False

# =============================================================================
#                    kraj konfiguracije - ispod ne treba menjati
# =============================================================================

CACHE = Path(__file__).parent / ".cache"
CACHE.mkdir(exist_ok=True)

PROMPTS: Dict[str, Dict[str, str]] = {
    "sr_zero": {
        "sys": "Ti si anotator sentimenta za tekstove na srpskom jeziku.",
        "user": (
            "Odredi sentiment sledece recenice.\n"
            "Odgovori iskljucivo jednom recju: pozitivan, negativan ili neutralan.\n\n"
            "Recenica: {tekst}\n\nSentiment:"
        ),
    },
    "sr_def": {
        "sys": "Ti si anotator sentimenta za tekstove na srpskom jeziku.",
        "user": (
            "Odredi sentiment sledece recenice prema ovim definicijama:\n"
            "- pozitivan: recenica prenosi povoljan ishod, korist ili odobravanje\n"
            "- negativan: recenica prenosi nepovoljan ishod, stetu ili neodobravanje\n"
            "- neutralan: recenica samo iznosi cinjenice, bez vrednosnog stava\n\n"
            "Vazno: ocenjuje se stav koji tekst prenosi, a ne tvoje misljenje o "
            "temi. Odgovori iskljucivo jednom recju.\n\n"
            "Recenica: {tekst}\n\nSentiment:"
        ),
    },
    "sr_few": {
        "sys": "Ti si anotator sentimenta za tekstove na srpskom jeziku.",
        "user": (
            "Odredi sentiment recenice. Odgovori jednom recju: "
            "pozitivan, negativan ili neutralan.\n\n"
            "{primeri}\n"
            "Recenica: {tekst}\nSentiment:"
        ),
    },
    "en_zero": {
        "sys": "You are a sentiment annotator for Serbian texts.",
        "user": (
            "Determine the sentiment of the following Serbian sentence.\n"
            "Answer with exactly one word: positive, negative, or neutral.\n\n"
            "Sentence: {tekst}\n\nSentiment:"
        ),
    },
    "en_def": {
        "sys": "You are a sentiment annotator for Serbian texts.",
        "user": (
            "Determine the sentiment of the following Serbian sentence using "
            "these definitions:\n"
            "- positive: conveys a favourable outcome, benefit or approval\n"
            "- negative: conveys an unfavourable outcome, harm or disapproval\n"
            "- neutral: states facts without an evaluative stance\n\n"
            "Judge the stance conveyed by the text, not your own opinion of the "
            "topic. Answer with exactly one word.\n\n"
            "Sentence: {tekst}\n\nSentiment:"
        ),
    },
    "en_few": {
        "sys": "You are a sentiment annotator for Serbian texts.",
        "user": (
            "Determine the sentiment of the Serbian sentence. Answer with one "
            "word: positive, negative, or neutral.\n\n"
            "{primeri}\n"
            "Sentence: {tekst}\nSentiment:"
        ),
    },
}

_PARSE = [
    (r"\b(pozitiv\w*|positive|pos)\b", "positive"),
    (r"\b(negativ\w*|negative|neg)\b", "negative"),
    (r"\b(neutral\w*|neutraln\w*|neu)\b", "neutral"),
    (r"\b(mesovit\w*|mixed)\b", "mixed"),
]


def parse_label(raw: str, allowed: List[str]) -> str:
    t = (raw or "").strip().lower()
    for pat, lab in _PARSE:
        if lab in allowed and re.search(pat, t):
            return lab
    return "NEPARSIRANO"


def make_backend():
    if BACKEND == "openai":
        from openai import OpenAI
        key = API_KLJUC or os.environ.get("OPENAI_API_KEY")
        if not key:
            sys.exit("GRESKA: nema API kljuca. Popunite API_KLJUC ili "
                     "postavite promenljivu okruzenja OPENAI_API_KEY.")
        client = OpenAI(api_key=key)

        def call(sys_msg, user_msg):
            r = client.chat.completions.create(
                model=NAZIV_MODELA, temperature=0, max_tokens=10,
                messages=[{"role": "system", "content": sys_msg},
                          {"role": "user", "content": user_msg}])
            return r.choices[0].message.content
        return call

    if BACKEND == "gemini":
        from google import genai
        key = API_KLJUC or os.environ.get("GOOGLE_API_KEY")
        if not key:
            sys.exit("GRESKA: nema API kljuca. Popunite API_KLJUC ili "
                     "postavite promenljivu okruzenja GOOGLE_API_KEY.")
        client = genai.Client(api_key=key)

        def call(sys_msg, user_msg):
            r = client.models.generate_content(
                model=NAZIV_MODELA, contents=f"{sys_msg}\n\n{user_msg}")
            return r.text
        return call

    if BACKEND == "ollama":
        import urllib.request

        def call(sys_msg, user_msg):
            body = json.dumps({
                "model": NAZIV_MODELA, "stream": False,
                "options": {"temperature": 0, "num_predict": 10},
                "messages": [{"role": "system", "content": sys_msg},
                             {"role": "user", "content": user_msg}],
            }).encode()
            req = urllib.request.Request(
                "http://localhost:11434/api/chat", data=body,
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=180) as resp:
                return json.loads(resp.read())["message"]["content"]
        return call

    if BACKEND == "dummy":
        rng = np.random.default_rng(0)
        opts = ["pozitivan", "negativan", "neutralan"]
        return lambda s, u: str(rng.choice(opts))

    sys.exit(f"GRESKA: nepoznat BACKEND: {BACKEND}")


def few_shot_block(df, labels, lang: str, seed: int = 7):
    """Bira po n primera po klasi; vraca (tekst bloka, indeksi primera)."""
    parts = [df[df.sentiment == lab].sample(
        min(BROJ_FEW_SHOT_PRIMERA, int((df.sentiment == lab).sum())),
        random_state=seed) for lab in labels]
    ex = pd.concat(parts).sample(frac=1.0, random_state=seed)
    sr_map = {"positive": "pozitivan", "negative": "negativan",
              "neutral": "neutralan", "mixed": "mesovit"}
    head = "Recenica" if lang == "sr" else "Sentence"
    lines = []
    for _, r in ex.iterrows():
        lab = sr_map[r["sentiment"]] if lang == "sr" else r["sentiment"]
        lines.append(f"{head}: {basic_clean(r['tekst'])}\nSentiment: {lab}\n")
    return "\n".join(lines), set(ex.index)


def evaluate_prompt(call, pname, tmpl, df_eval):
    key = f"{BACKEND}_{NAZIV_MODELA}_{pname}_{len(df_eval)}"
    cache_f = CACHE / f"gen_{re.sub(r'[^a-zA-Z0-9_]', '_', key)}.json"
    if cache_f.exists() and not IGNORISI_KES:
        print("  (ucitano iz kesa)")
        return json.loads(cache_f.read_text(encoding="utf-8"))

    raw_out = []
    for i, t in enumerate(df_eval["tekst"]):
        msg = tmpl["user"].format(tekst=basic_clean(t),
                                  primeri=tmpl.get("_primeri", ""))
        for attempt in range(4):
            try:
                raw_out.append(call(tmpl["sys"], msg))
                break
            except Exception as e:
                if attempt == 3:
                    print(f"  greska na primeru {i}: {e}")
                    raw_out.append("")
                else:
                    time.sleep(2 ** attempt)
        if PAUZA_IZMEDJU_POZIVA:
            time.sleep(PAUZA_IZMEDJU_POZIVA)
        if (i + 1) % 50 == 0:
            print(f"    {i + 1}/{len(df_eval)}", flush=True)

    cache_f.write_text(json.dumps(raw_out, ensure_ascii=False), encoding="utf-8")
    return raw_out


def main():
    put = Path(PUTANJA_PODACI)
    if not put.exists():
        sys.exit(f"GRESKA: ne postoji fajl sa podacima: {put.resolve()}\n"
                 f"Ispravite PUTANJA_PODACI na vrhu skripte.")

    outdir = Path(IZLAZNI_DIREKTORIJUM)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(put)
    labels = present_labels(df)
    call = make_backend()
    if BACKEND == "dummy":
        print("PAZNJA: BACKEND = 'dummy' - odgovori su nasumicni. "
              "Promenite BACKEND pre nego sto rezultate stavite u izvestaj.\n")

    rows, preds_store = [], {}
    for pname in UPITI:
        if pname not in PROMPTS:
            print(f"UPOZORENJE: nepoznata varijanta upita '{pname}' - preskacem.")
            continue
        tmpl = dict(PROMPTS[pname])
        df_eval = df
        if pname.endswith("_few"):
            block, used = few_shot_block(df, labels, pname[:2])
            tmpl["_primeri"] = block
            df_eval = df.drop(index=list(used))  # bez curenja podataka
        if VELICINA_UZORKA:
            df_eval = df_eval.sample(min(VELICINA_UZORKA, len(df_eval)),
                                     random_state=1)

        print(f"\n=== UPIT: {pname} ({len(df_eval)} primera) ===")
        raw = evaluate_prompt(call, pname, tmpl, df_eval)
        pred = [parse_label(r, labels) for r in raw]
        gold = df_eval["sentiment"].tolist()

        neparsirano = sum(p == "NEPARSIRANO" for p in pred)
        mask = [p != "NEPARSIRANO" for p in pred]
        g2 = [g for g, m in zip(gold, mask) if m]
        p2 = [p for p, m in zip(pred, mask) if m]

        f1 = f1_score(g2, p2, average="macro", labels=labels, zero_division=0)
        acc = float(np.mean([g == p for g, p in zip(gold, pred)]))
        rows.append({
            "upit": pname,
            "jezik": "srpski" if pname.startswith("sr") else "engleski",
            "tip": pname.split("_")[1],
            "macroF1": f1, "tacnost": acc,
            "neparsirano": neparsirano, "n": len(df_eval),
        })
        preds_store[pname] = (gold, pred)
        print(f"macro-F1 = {f1:.4f} | tacnost = {acc:.4f} | "
              f"neparsiranih odgovora: {neparsirano}")
        print(classification_report(g2, p2, labels=labels, digits=3,
                                    zero_division=0))

    if not rows:
        sys.exit("GRESKA: nijedna varijanta upita nije izvrsena.")

    res = pd.DataFrame(rows).sort_values("macroF1", ascending=False)
    res.to_csv(outdir / "rezultati_dekoder.csv", index=False, encoding="utf-8")
    print("\n=== SVI UPITI ===")
    print(res.to_string(index=False))
    _grafikoni(res, preds_store, labels, outdir)


def _grafikoni(res, preds_store, labels, outdir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    r = res.sort_values("macroF1")
    axes[0].barh(r.upit, r.macroF1, color="#8172b2")
    axes[0].set_xlabel("macro-F1")
    axes[0].set_title(f"Efekat formata upita ({NAZIV_MODELA})")

    piv = res.pivot_table(index="tip", columns="jezik", values="macroF1")
    piv.plot(kind="bar", ax=axes[1], rot=0)
    axes[1].set_ylabel("macro-F1")
    axes[1].set_title("Jezik upita x tip upita")
    fig.tight_layout()
    fig.savefig(outdir / "dekoder_upiti.png", dpi=150)

    best = res.iloc[0].upit
    gold, pred = preds_store[best]
    labs = list(labels) + (["NEPARSIRANO"] if "NEPARSIRANO" in pred else [])
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(gold, pred, labels=labs)
    ConfusionMatrixDisplay(cm, display_labels=labs).plot(
        ax=ax, cmap="Purples", colorbar=False, values_format="d")
    ax.set_title(f"Najbolji upit: {best}")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(outdir / "dekoder_matrica_konfuzije.png", dpi=150)
    print(f"\nGrafikoni sacuvani u: {outdir}/")


if __name__ == "__main__":
    main()
