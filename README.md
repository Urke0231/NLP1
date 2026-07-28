# Analiza ekonomskog sentimenta na srpskom jeziku

Projekat iz predmeta Obrada prirodnog jezika. Zadatak: klasifikacija
sentimenta (**positive** / **negative** / **neutral**) rečenica iz
ekonomskih vesti na srpskom, uz poređenje klasičnih linearnih modela,
enkoderskih transformera (zamrznuti embedinzi i pravi fine-tuning) i
dekoderskih LLM-ova preko promptovanja.

Tim: Uroš, Manojlo, Jelena, Damjan (imena anotatora, vidi `uneo` polje u
podacima).

Nijedna skripta nema parametre iz komandne linije. Svaka na vrhu ima blok
`KONFIGURACIJA` sa promenljivim koje se menjaju, pa se pokreće prosto sa
`python 0X_....py`. `02_osnovni_modeli.py` dodatno na startu postavlja jedno
interaktivno pitanje (uklanjanje geografskih/političkih imena, vidi niže).

## Podaci

- **`anotacije-2026-07-26.json`** — finalni anotirani skup, 2919 rečenica,
  polja `tekst`, `sentiment` (`positive`/`negative`/`neutral`), `duzina`
  (`kratak`/`srednji`/`dugacak`), `uneo` (ime anotatora). Klase su
  približno izbalansirane (964 negative / 1000 neutral / 955 positive).
  `anotacije-2026-07-23 (2).json`, `-07-24`, `-07-25` su međuverzije iz
  ranijih faza anotiranja, ostavljene radi tragova rada.
- **`podaci/anotirano.json`** — verzija skupa korišćena u ranijoj fazi
  (Faza 2, deskriptivna statistika); `01_statistika.py` još uvek pokazuje
  na `anotacije-2026-07-23 (2).json`, a ne na finalni `-07-26` skup —
  ako se statistika ponovo generiše, prvo ažurirati `PUTANJA_PODACI`.
- **`Skripte za anotiranje/le_script_withExtraSteps.py`** — Tkinter GUI
  alat kojim su članovi tima nezavisno anotirali/kalibrisali tekstove
  (upisuje `dataset.json` i `poredjenje_annotacija.json`).
- **`analiza_saglasnosti/`** — slaganje anotatora: Cohen κ (parovi),
  Fleiss κ i Krippendorffov α (svi anotatori), procentualna saglasnost;
  `analiza_anotacije.py` generiše `cohen_kappa.png`,
  `globalne_mere_saglasnosti.png`, `procentualna_saglasnost.png`,
  `statistika_po_anotatorima.png` iz `poredjenje_anotacija.json`.

## Struktura

```
projekat/
├── anotacije-2026-07-26.json          <- finalni anotirani skup (koriste ga 02, 03, 04)
├── podaci/anotirano.json              <- skup iz ranije faze
├── data.py                            <- učitavanje JSON-a, LABEL_MAP
├── preprocessing.py                   <- transliteracija, stemer, lematizator, NER filter
├── SerbianStemmer.py                  <- stemer (Nikola Milošević), već u repou
├── 01_statistika.py                   <- Faza 2: deskriptivna statistika + slaganje
├── 02_osnovni_modeli.py               <- Faza 3a: linearni modeli (TF-IDF i varijante)
├── 03_enkoderski_modeli.py            <- Faza 3b: zamrznuti Ollama embedinzi + klasifikator
├── 03b_encoder_finetuning.py          <- Faza 3b: pravi fine-tuning enkoderskih transformera
├── 04_dekoderski_modeli.py            <- Faza 3c: dekoderski LLM (zero/def/few-shot)
├── rezultati_jednostavni_konacni_10_CV_NER/  <- finalni izlaz iz 02 (sa NER filtriranjem imena)
├── rezultati_jednostavni_konacni_10_CV/      <- izlaz iz 02 bez filtriranja imena
├── rezultati_enkoder_konacno/         <- finalni izlaz iz 03
├── rezultati_finetuning_konacno/      <- finalni izlaz iz 03b (xlm-roberta-base, mBERT)
├── rezultati_dekoder_konacno/         <- finalni izlaz iz 04 (olivilo/zora)
├── opj_bertic/                        <- Colab notebook: fine-tuning classla/bcms-bertic
│   └── opj_bertic.ipynb, bertic1-5.PNG, opj_dataset.json
├── analiza_saglasnosti/               <- slaganje anotatora (kappa/alfa), grafici
├── Skripte za anotiranje/             <- Tkinter alat za anotiranje
└── Poslata dokumentacija/             <- predati izveštaj i prilozi
```

Stariji izlazni direktorijumi (`rezultati/`, `rezultati2/`, `rezultati3/`,
`rezultati_dekoder/`, `rezultati_enkoder/`, `rezultati_jednostavni_konacni/`,
`rezultati_jednostavni_konacni_CLASSLA_ner/`) su bili međurezultati iz ranijih
pokretanja i obrisani su — trenutno važeći su oni sa sufiksom `_konacno` /
`_10_CV[_NER]` navedeni gore.

## Skripte

### `01_statistika.py` — Faza 2
Deskriptivna statistika (dužine, distribucija klasa, najčešće reči po
klasi) i Cohen κ nad kalibracionim skupom. **Napomena:** ovaj skript je
najstariji u pajplajnu i još uvek gleta na predfinalnu verziju podataka
(vidi gore); rezultat u `rezultati/` je obrisan iz repoa jer je zamenjen
detaljnijom analizom u `analiza_saglasnosti/`.

### `02_osnovni_modeli.py` — Faza 3a
Poredi 3 varijante pretprocesiranja (`lower`, `lower+stem`, `lower+lema`;
`sirovo` je zakomentarisano) × 7 varijanti odlika (`TF`, `IDF`, `TFIDF`,
`TFIDF_1-2`, `TFIDF_1-3`, `CHAR_3-5`, `REC+CHAR`) × 4 modela (`Vecinski`,
`MultinomialNB`, `LogRegresija`, `LinearSVM`), 10-slojna stratifikovana
unakrsna validacija sa ugnežđenom `GridSearchCV` optimizacijom i
izdvojenim test skupom (10%). Na startu pita da li da se geografska/
politička imena (npr. „Srbija") uklone pre vektorizacije, da ne bi lažno
korelirala sa sentimentom — ručnom listom korena ili automatskom CLASSLA
NER detekcijom. Finalno pokretanje (`rezultati_jednostavni_konacni_10_CV_NER`)
korišćeno je sa NER filtriranjem.

### `03_enkoderski_modeli.py` — Faza 3b (zamrznuti embedinzi)
Rečenične vektore računa lokalni Ollama servis (`paraphrase-multilingual`,
port 11434, keširano u `.cache/`); model se ne dotrenirava, već se iznad
vektora treniraju `LogRegresija`, `LinearSVM`, `RBF-SVM` (uz `Vecinski`
baseline). Isti protokol evaluacije kao `02` (test 10% + 5-slojna CV).

### `03b_encoder_finetuning.py` — Faza 3b (pravi fine-tuning)
Stvarno dotrenirava težine enkoderskog transformera (ručna petlja
obučavanja bez `Trainer`-a), 10-slojna CV, evaluacija posle svake epohe →
kriva epohe→macro-F1. Lokalno pokrenuto za `xlm-roberta-base` i
`bert-base-multilingual-cased` (višejezični modeli); `classla/bcms-bertic`
(monolingvalni BCMS) je u ovoj skripti zakomentarisan jer je umesto toga
fino podešen posebno na Google Colabu — vidi `opj_bertic/`. Sadrži
uputstvo za pokretanje na AMD GPU (ROCm 7.2.1, Windows, `.venv312`) —
testirano i radi na RX 7900 XT iako zvanično nije podržana.

### `04_dekoderski_modeli.py` — Faza 3c
Dekoderski LLM se ne trenira — ceo skup je evaluacioni. Varira jezik
upita (srpski/engleski) × tip upita (zero-shot / zero-shot sa definicijama
/ few-shot), 6 kombinacija. Finalno pokretanje: `BACKEND = "ollama"`,
`NAZIV_MODELA = "olivilo/zora"` (lokalni srpski model preko Ollama-e).
Odgovori se keširaju u `.cache/`.

Ako `data.py`-jev `LABEL_MAP` ne prepozna neku oznaku, skripta pukne sa
jasnom porukom — dopuniti mapu.

## Rezultati (finalna pokretanja)

| Pristup | Najbolja konfiguracija | test macro-F1 |
|---|---|---|
| Osnovni (TF-IDF + linearni) | `lower+lema` + `REC+CHAR` + LinearSVM | **0.703** |
| Zamrznuti embedinzi (Ollama) + SVM | `paraphrase-multilingual` + LinearSVM | **0.769** |
| Fine-tuning (pravi) | `xlm-roberta-base`, 5 epoha | **0.800** |
| Fine-tuning (pravi) | `bert-base-multilingual-cased`, 5 epoha | 0.721 |
| Dekoderski LLM (olivilo/zora) | engleski, few-shot | 0.737 |
| Dekoderski LLM (olivilo/zora) | srpski, zero-shot | 0.583 |

Najbolji rezultat daje pravi fine-tuning `xlm-roberta-base` (macro-F1
≈0.80), ispred zamrznutih embedinga (≈0.77), promptovanja LLM-a (≈0.74) i
klasičnih TF-IDF modela (≈0.70). `classla/bcms-bertic` (monolingvalni)
fino je podešen odvojeno u `opj_bertic/opj_bertic.ipynb` (Colab); brojevi
nisu izvezeni u CSV, već su u vidu screenshotova (`bertic1-5.PNG`) i u
finalnom izveštaju.

Detaljni rezultati (po foldu, matrice konfuzije, statistička značajnost
Wilcoxonovim uparenim testom, najinformativnije odlike) su u odgovarajućim
`rezultati_*` direktorijumima.

## Finalni izveštaj

`Poslata dokumentacija/` sadrži predati izveštaj
(`OPJ-Analiza-Ekonomskog-Sentimenta.docx`) i izvezene anotacije
(`anotacije_Ceo_skup.txt`, `poredjenje_anotacija.txt`).

## Instalacija

```bash
pip install scikit-learn pandas numpy matplotlib scipy statsmodels seaborn
pip install classla                 # lematizacija i NER filter
python -c "import classla; classla.download('sr')"
pip install transformers torch      # enkoderski modeli / fine-tuning
```

**Ollama** (zamrznuti embedinzi u `03` i dekoderski model u `04`):
`ollama serve`, pa `ollama pull paraphrase-multilingual` i
`ollama pull olivilo/zora` (ili drugi `BACKEND`: `openai` / `gemini` /
`dummy` u `04`).

**Stemer:** `SerbianStemmer.py` (Nikola Milošević,
<https://github.com/nikolamilosevic86/SerbianStemmer>) je već uključen u
repo; `preprocessing.py` ga automatski detektuje.

**Fine-tuning na AMD GPU (Windows):** vidi opširno uputstvo na vrhu
`03b_encoder_finetuning.py` — poseban `.venv312` (Python 3.12) sa ROCm 7.2.1
PyTorch wheel-ovima i `transformers==4.46.3` (novije verzije pucaju na
`torch.distributed.fsdp` uvozu).

## Detalji na koje treba obratiti pažnju

- **Metrika.** Klase su približno izbalansirane, ali glavna metrika je
  ipak macro-F1 (tačnost ide uz nju radi poređenja sa većinskim
  klasifikatorom).
- **Ćirilica/latinica.** Sve se prevodi u latinicu pre svega ostalog.
- **Geografska/politička imena.** Ekonomske vesti često pominju „Srbija",
  „EU" i sl., što bi moglo lažno korelirati sa klasom — `02` nudi
  opciono uklanjanje (ručna lista ili CLASSLA NER); finalni rezultat
  koristi NER varijantu.
- **Curenje podataka.** Vektorizacija je unutar `Pipeline`-a (rečnik/IDF
  računat samo na trening foldu); test skup (10%) se izdvaja pre
  unakrsne validacije i ne koristi se za podešavanje hiperparametara.
- **Statistička značajnost.** `statisticko_poredjenje.csv` u svakom
  `rezultati_*` direktorijumu sadrži Wilcoxonov upareni test po foldovima.
- **Broj epoha kod fine-tuninga.** Kriva epohe→macro-F1 se dobija
  evaluacijom na CV foldovima; izbor „najboljeg" broja epoha je post hoc
  (napomenuto i u izveštaju).
