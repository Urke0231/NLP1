"""
Pretprocesiranje tekstova na srpskom jeziku.

Sadrzi:
  - transliteraciju cirilica -> latinica
  - lowercasing
  - stemovanje (SerbianStemmer, Nikola Milosevic)
  - lematizaciju (CLASSLA)
  - kes na disku, jer su stemovanje i narocito lematizacija spori

Autor koda: generisano kao skelet za projekat iz Obrade prirodnih jezika.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Transliteracija cirilica -> latinica
# ---------------------------------------------------------------------------
# Vazno: i SerbianStemmer i vecina modela ocekuju latinicu.
# Redosled je bitan - viseslovni parovi (dj, lj, nj, dz) se resavaju mapom.

_CYR2LAT = {
    "А": "A", "Б": "B", "В": "V", "Г": "G", "Д": "D", "Ђ": "Đ", "Е": "E",
    "Ж": "Ž", "З": "Z", "И": "I", "Ј": "J", "К": "K", "Л": "L", "Љ": "Lj",
    "М": "M", "Н": "N", "Њ": "Nj", "О": "O", "П": "P", "Р": "R", "С": "S",
    "Т": "T", "Ћ": "Ć", "У": "U", "Ф": "F", "Х": "H", "Ц": "C", "Ч": "Č",
    "Џ": "Dž", "Ш": "Š",
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "ђ": "đ", "е": "e",
    "ж": "ž", "з": "z", "и": "i", "ј": "j", "к": "k", "л": "l", "љ": "lj",
    "м": "m", "н": "n", "њ": "nj", "о": "o", "п": "p", "р": "r", "с": "s",
    "т": "t", "ћ": "ć", "у": "u", "ф": "f", "х": "h", "ц": "c", "ч": "č",
    "џ": "dž", "ш": "š",
}


def cyr_to_lat(text: str) -> str:
    """Prevodi cirilicki tekst u latinicu. Latinicki tekst ostaje nepromenjen."""
    return "".join(_CYR2LAT.get(ch, ch) for ch in text)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def basic_clean(text: str) -> str:
    """Minimalno ciscenje koje se primenjuje na SVE varijante pretprocesiranja."""
    return normalize_whitespace(cyr_to_lat(text))


# ---------------------------------------------------------------------------
# 2. Tokenizacija
# ---------------------------------------------------------------------------
# Jednostavan regex tokenizator - zadrzava reci sa dijakriticima, brojeve i
# procente (bitno za novinski domen: "7,9%", "2026").

_TOKEN_RE = re.compile(r"\w+(?:[.,]\w+)*%?", re.UNICODE)


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text)


# ---------------------------------------------------------------------------
# 3. Stemovanje - adapter za SerbianStemmer
# ---------------------------------------------------------------------------
# Preuzmi SerbianStemmer.py sa https://github.com/nikolamilosevic86/SerbianStemmer
# i stavi ga PORED ovog fajla. Adapter automatski trazi ulaznu funkciju, jer se
# imena funkcija razlikuju izmedju verzija/forkova repozitorijuma.
#
# Alternativa (Java): SCStemmers, https://vukbatanovic.github.io/SCStemmers/
# - sadrzi Keselj-Sipka greedy/optimal stemer, Milosevicevo poboljsanje i
#   Ljubesic-Pandzic stemer za hrvatski. Moze se pozvati preko subprocess-a.

_STEM_CANDIDATE_NAMES = [
    "stem", "stemWord", "stem_word", "stemmer", "stemming",
    "SerbianStemmer", "stem_text", "stemText","stem_arr","stem_str"
]


def _load_external_stemmer() -> Optional[Callable[[str], str]]:
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        import SerbianStemmer as _ss  # type: ignore
    except Exception:
        return None
    for name in _STEM_CANDIDATE_NAMES:
        fn = getattr(_ss, name, None)
        if callable(fn):
            try:
                out = fn("kucama")
                if isinstance(out, str):
                    return fn
            except Exception:
                continue
    return None


# --- Rezervni (fallback) stemer -------------------------------------------
# Koristi se SAMO ako SerbianStemmer.py nije pronadjen, da bi skripta mogla da
# se pokrene. U dokumentaciji obavezno navesti koji je stemer stvarno korscen.
# Ovo je skraceni greedy suffix-stripping, sortiran po duzini sufiksa.

_FALLBACK_SUFFIXES = sorted(
    [
        "ovanje", "ivanje", "avanje", "iranje", "enje", "anje", "acija", "icija",
        "ostima", "ostiju", "ostima", "osti", "ost", "stvo", "stva", "stvu",
        "ijama", "ijom", "ijem", "ima", "ama", "oga", "omu", "ome", "emu",
        "ijih", "ijem", "ijim", "iji", "ija", "iju", "ije", "ijo",
        "ova", "ovi", "ove", "ovu", "ovo", "eva", "evi", "eve", "evu",
        "ala", "alo", "ali", "ale", "ao", "la", "lo", "li", "le",
        "ti", "ci", "ce", "cu", "ka", "ku", "ke", "ki",
        "im", "om", "em", "am", "um", "ah", "ih", "eh", "oh",
        "a", "e", "i", "o", "u", "y",
    ],
    key=len,
    reverse=True,
)


def _fallback_stem(word: str) -> str:
    w = word
    if len(w) <= 3:
        return w
    for suf in _FALLBACK_SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)]
    return w


_EXTERNAL_STEMMER = _load_external_stemmer()
USING_EXTERNAL_STEMMER = _EXTERNAL_STEMMER is not None


def stem_word(word: str) -> str:
    if _EXTERNAL_STEMMER is not None:
        try:
            return _EXTERNAL_STEMMER(word)
        except Exception:
            return _fallback_stem(word)
    return _fallback_stem(word)


def stem_text(text: str) -> str:
    return " ".join(stem_word(t) for t in tokenize(text))


# ---------------------------------------------------------------------------
# 4. Lematizacija - CLASSLA
# ---------------------------------------------------------------------------
# pip install classla
# python -c "import classla; classla.download('sr')"

_CLASSLA_PIPELINE = None


def _get_classla(use_gpu: bool = False):
    global _CLASSLA_PIPELINE
    if _CLASSLA_PIPELINE is None:
        import classla  # lokalni import, da modul radi i bez classla

        try:
            _CLASSLA_PIPELINE = classla.Pipeline(
                "sr", processors="tokenize,pos,lemma", use_gpu=use_gpu
            )
        except Exception:
            classla.download("sr")
            _CLASSLA_PIPELINE = classla.Pipeline(
                "sr", processors="tokenize,pos,lemma", use_gpu=use_gpu
            )
    return _CLASSLA_PIPELINE


def lemmatize_batch(texts: List[str], use_gpu: bool = False,
                    keep_pos: bool = False) -> List[str]:
    """Lematizuje listu tekstova. Rezultat se kesira na disku.

    keep_pos=True dodaje UPOS oznaku uz lemu ("rast_NOUN"), sto ponekad
    poboljsava linearne modele jer razdvaja homonime.
    """
    key = hashlib.md5(
        (json.dumps(texts, ensure_ascii=False) + f"|pos={keep_pos}").encode("utf-8")
    ).hexdigest()
    cache_file = CACHE_DIR / f"lemma_{key}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))

    nlp = _get_classla(use_gpu=use_gpu)
    out: List[str] = []
    for i, t in enumerate(texts):
        doc = nlp(t if t.strip() else "prazno")
        lemmas = []
        for sent in doc.sentences:
            for w in sent.words:
                lem = (w.lemma or w.text).lower()
                lemmas.append(f"{lem}_{w.upos}" if keep_pos else lem)
        out.append(" ".join(lemmas))
        if (i + 1) % 100 == 0:
            print(f"  lematizovano {i + 1}/{len(texts)}", flush=True)

    cache_file.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return out


def stem_batch(texts: List[str]) -> List[str]:
    key = hashlib.md5(
        (json.dumps(texts, ensure_ascii=False) + f"|ext={USING_EXTERNAL_STEMMER}")
        .encode("utf-8")
    ).hexdigest()
    cache_file = CACHE_DIR / f"stem_{key}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))
    out = [stem_text(t) for t in texts]
    cache_file.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# 4b. Uklanjanje geografskih/politickih imena
# ---------------------------------------------------------------------------
# Imena drzava/gradova mogu slucajno korelisati sa sentimentom u skupu
# podataka (npr. vesti o domacoj vs. stranoj ekonomiji), a da sama rec nema
# nikakav sentiment - model bi ih onda naucio kao laznu odliku. Dve opcije:
#   "stopwords" - rucna lista korena (brzo, bez zavisnosti)
#   "ner"       - automatska NER detekcija (CLASSLA), sporije, ali ne trazi
#                 rucno odrzavanje liste. Radi na tekstu SA velikim slovima
#                 (pre lowercasing-a), jer se NER model na to oslanja.

GEO_STOPWORD_ROOTS = [
    "srbij", "beograd", "novi sad", "nis", "kragujev",
    "nemack", "berlin", "keln", "sxtutgart", "stutgart", "minhen",
    "frankfurt", "hamburg",
    "hrvat", "zagreb", "crn gor", "crnogor", "podgoric",
    "bosn", "sarajev", "makedonij", "skoplj", "slovenij", "ljubljan",
    "madxarsk", "budimpesxt", "budimpest", "rumunij", "bukuresxt",
    "bugarsk", "sofij", "grck", "atin", "albanij", "tiran",
    "evrop", "amerik", "kin", "rusij", "moskv",
    "francusk", "pariz", "italij", "rim", "spanij", "madrid",
    "britanij", "london", "svajcarsk", "cirih", "austrij", "becx",
]


def strip_geo_stopwords(text: str) -> str:
    """Uklanja tokene koji pocinju korenom geografskog/politickog imena."""
    kept = [t for t in tokenize(text)
            if not any(t.lower().startswith(r) for r in GEO_STOPWORD_ROOTS)]
    return " ".join(kept)


def strip_geo_stopwords_batch(texts: List[str]) -> List[str]:
    return [strip_geo_stopwords(t) for t in texts]


_NER_PIPELINE = None


def _get_classla_ner(use_gpu: bool = False):
    global _NER_PIPELINE
    if _NER_PIPELINE is None:
        import classla  # lokalni import, da modul radi i bez classla

        try:
            _NER_PIPELINE = classla.Pipeline(
                "sr", processors="tokenize,ner", use_gpu=use_gpu
            )
        except Exception:
            classla.download("sr")
            _NER_PIPELINE = classla.Pipeline(
                "sr", processors="tokenize,ner", use_gpu=use_gpu
            )
    return _NER_PIPELINE


def strip_named_entities_batch(texts: List[str], use_gpu: bool = False,
                               entity_types=("LOC", "ORG")) -> List[str]:
    """Uklanja tokene prepoznate kao imenovani entiteti (drzave, gradovi,
    organizacije...) pomocu CLASSLA NER-a. Rezultat se kesira na disku, kao
    stemovanje i lematizacija.
    """
    key = hashlib.md5(
        (json.dumps(texts, ensure_ascii=False) + f"|ner={entity_types}")
        .encode("utf-8")
    ).hexdigest()
    cache_file = CACHE_DIR / f"ner_{key}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))

    nlp = _get_classla_ner(use_gpu=use_gpu)
    out: List[str] = []
    for i, t in enumerate(texts):
        doc = nlp(t if t.strip() else "prazno")
        kept = []
        for sent in doc.sentences:
            for token in sent.tokens:
                tag = token.ner or "O"
                if not any(tag.endswith(f"-{et}") for et in entity_types):
                    kept.append(token.text)
        out.append(" ".join(kept))
        if (i + 1) % 100 == 0:
            print(f"  NER obrada {i + 1}/{len(texts)}", flush=True)

    cache_file.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# 5. Varijante pretprocesiranja koje se porede u eksperimentu
# ---------------------------------------------------------------------------

def build_variants(raw_texts: List[str],
                   include_lemma: bool = True,
                   lemma_use_gpu: bool = False,
                   geo_filter: Optional[str] = None) -> Dict[str, List[str]]:
    """Vraca recnik: naziv varijante -> lista pretprocesiranih tekstova.

    geo_filter: None (nista), "stopwords" (rucna lista korena) ili "ner"
    (automatska CLASSLA NER detekcija). Uklanja geografska/politicka imena
    PRE lowercasing-a/stemovanja/lematizacije, da ne bi lazno uticala na
    ocenu sentimenta.
    """
    cleaned = [basic_clean(t) for t in raw_texts]

    if geo_filter == "stopwords":
        cleaned = strip_geo_stopwords_batch(cleaned)
    elif geo_filter == "ner":
        print("NER detekcija geografskih/politickih imena u toku "
             "(CLASSLA, koristi se kes)...")
        cleaned = strip_named_entities_batch(cleaned, use_gpu=lemma_use_gpu)
    elif geo_filter is not None:
        raise ValueError(f"Nepoznat geo_filter: {geo_filter!r}")

    lowered = [t.lower() for t in cleaned]

    variants: Dict[str, List[str]] = {
        "sirovo": cleaned,            # samo translit + whitespace
        "lower": lowered,             # + lowercasing
        "lower+stem": stem_batch(lowered),
    }
    if include_lemma:
        print("Lematizacija u toku (CLASSLA je spora, koristi se kes)...")
        variants["lower+lema"] = lemmatize_batch(lowered, use_gpu=lemma_use_gpu)
    return variants


if __name__ == "__main__":
    PUTANJA_PODACI = "./anotacije-2026-07-26.json"

    put = Path(PUTANJA_PODACI)
    if put.exists():
        from data import load_dataset

        df = load_dataset(put)
        print(f"Ucitano {len(df)} primera iz {put.name}")
        demo = df["tekst"].iloc[0]
    else:
        print(f"UPOZORENJE: ne postoji fajl sa podacima: {put.resolve()} "
              f"- koristi se ugradjeni primer.")
        demo = (
            "Највећи раст станарина забележен је у Келну (7,9%), "
            "док су у Берлину порасле за само 1%."
        )

    print("original :", demo)
    print("latinica :", basic_clean(demo))
    print("lower    :", basic_clean(demo).lower())
    print("stem     :", stem_text(basic_clean(demo).lower()))
    print("eksterni SerbianStemmer ucitan:", USING_EXTERNAL_STEMMER)

    if put.exists():
        print("Stemovanje celog skupa i pakovanje u JSON...")
        lowered_all = [basic_clean(t).lower() for t in df["tekst"]]
        stemmed_all = stem_batch(lowered_all)
        duzine = df["duzina"] if "duzina" in df.columns else [None] * len(df)
        uneo = df["uneo"] if "uneo" in df.columns else [None] * len(df)

        out_records = [
            {"tekst": stem, "sentiment": sentiment, "duzina": duzina, "uneo": uneo_}
            for stem, sentiment, duzina, uneo_ in zip(stemmed_all, df["sentiment"], duzine, uneo)
        ]
        out_path = Path("stemovani_podaci_konacni.json")
        out_path.write_text(
            json.dumps(out_records, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Sacuvano {len(out_records)} stemovanih primera u {out_path.resolve()}")
