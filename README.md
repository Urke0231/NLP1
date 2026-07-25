# Analiza sentimenta na srpskom — skelet eksperimenta

Nijedna skripta nema parametre iz komandne linije. Svaka na vrhu ima blok
`KONFIGURACIJA` sa promenljivim koje popunjavate, pa se pokreće prosto sa
`python 0X_....py`.

## Struktura

```
projekat/
├── podaci/
│   ├── anotirano.json              <- vaš glavni skup
│   └── kalibracija_*.json          <- po jedan fajl za svakog člana grupe
├── data.py                         <- učitavanje JSON-a (LABEL_MAP se možda menja)
├── preprocessing.py                <- transliteracija, stemer, lematizator
├── SerbianStemmer.py               <- ručno preuzeti sa GitHub-a (vidi niže)
├── 01_statistika.py
├── 02_osnovni_modeli.py
├── 03_enkoderski_modeli.py
├── 04_dekoderski_modeli.py
└── rezultati/                      <- kreira se automatski
```

## Šta treba da popunite u kojoj skripti

### `01_statistika.py`
| Promenljiva | Šta je |
|---|---|
| `PUTANJA_PODACI` | putanja do glavnog JSON-a |
| `IZLAZNI_DIREKTORIJUM` | gde se upisuju tabele i grafikoni |
| `PUTANJE_KALIBRACIJA` | lista fajlova sa paralelnim anotacijama; `[]` ako je još nemate |
| `UKLONI_DUPLIKATE` | postavka traži pročišćen skup, ostavite `True` |
| `BROJ_NAJCESCIH_RECI` | koliko reči po klasi da ispiše |

### `02_osnovni_modeli.py`
| Promenljiva | Šta je |
|---|---|
| `PUTANJA_PODACI`, `IZLAZNI_DIREKTORIJUM` | isto kao gore |
| `VARIJANTE_PRETPROCESIRANJA` | `sirovo` / `lower` / `lower+stem` / `lower+lema` |
| `VARIJANTE_ODLIKA` | `TF`, `IDF`, `TFIDF`, `TFIDF_1-2`, `TFIDF_1-3`, `CHAR_3-5`, `REC+CHAR` |
| `MODELI` | `Vecinski`, `MultinomialNB`, `LogRegresija`, `LinearSVM` |
| `VELICINA_TEST` | 0.1 — 1/10 podataka se izdvaja kao test skup, ne koristi se za trening/podešavanje |
| `BROJ_VALIDACIONIH_FOLDOVA` | 5 — preostalih 9/10 se deli na 4 trening / 1 validacija, rotira se (krosvalidacija) |
| `BROJ_NITI` | broj tredova za paralelno računanje konfiguracija (pretprocesiranje × odlike × model) |
| `MREZA_C`, `MREZA_ALPHA` | opsezi hiperparametara |
| `MIN_DF_REC`, `MIN_DF_KARAKTER` | spustite na 1 ako imate < 500 primera |
| `BROJ_JEZGARA` | `-1` = sva |
| `BRZI_TEST` | `True` dok proveravate da sve radi |

### `03_enkoderski_modeli.py`
| Promenljiva | Šta je |
|---|---|
| `PUTANJA_PODACI` | na Colabu obično putanja unutar `/content/drive/...` |
| `MODELI` | lista HuggingFace naziva |
| `BROJ_EPOHA` | koliko tačaka ima kriva „epohe → macro-F1" |
| `BROJ_FOLDOVA` | 10; spustite na 2–3 dok testirate |
| `VELICINA_BATCHA` | prepolovite ako dobijete CUDA out of memory |
| `STOPA_UCENJA` | uobičajeno 1e-5 do 5e-5 |
| `MAKSIMALNA_DUZINA` | 256; 128 znatno ubrzava |
| `KORISTI_TEZINE_KLASA` | `True` kod neuravnoteženog sentimenta |

### `04_dekoderski_modeli.py`
| Promenljiva | Šta je |
|---|---|
| `BACKEND` | `openai` / `gemini` / `ollama` / `dummy` |
| `NAZIV_MODELA` | npr. `gpt-4o-mini`, `gemini-2.0-flash`, `qwen2.5:7b-instruct` |
| `API_KLJUC` | ostavite prazno i koristite promenljivu okruženja |
| `UPITI` | 6 kombinacija jezik × tip upita |
| `BROJ_FEW_SHOT_PRIMERA` | po klasi; ti primeri se izbacuju iz evaluacije |
| `VELICINA_UZORKA` | `None` = ceo skup; stavite 50 dok testirate |
| `PAUZA_IZMEDJU_POZIVA` | povećajte kod rate-limit grešaka |
| `IGNORISI_KES` | odgovori se keširaju, pa ponovno pokretanje ne troši kredite |

`BACKEND = "dummy"` daje nasumične oznake — služi samo da proverite da skripta
radi bez trošenja kredita. Ne stavljajte te brojeve u izveštaj.

Ako u `data.py` `LABEL_MAP` ne prepozna vaše oznake, skripta pukne sa jasnom
porukom — dopunite mapu i to je sve.

## Odgovor na komentar predavača

| Primedba | Gde je pokrivena |
|---|---|
| „još neke tehnike pretprocesiranja osim TF-IDF-a" | mreža 4 pretprocesiranja × 7 odlika u `02` |
| lowercasing | varijanta `lower` vs `sirovo` |
| stemovanje | `lower+stem` → SerbianStemmer |
| lematizacija | `lower+lema` → CLASSLA |
| n-gramske odlike | `TFIDF_1-2`, `TFIDF_1-3`, `CHAR_3-5`, `REC+CHAR` |
| „navesti koji višejezični model" | `xlm-roberta-base` primarni, `bert-base-multilingual-cased` drugi |
| „razmotriti i neki dekoderski model" | `04_dekoderski_modeli.py` |

## Predlog modela

**Osnovni:** većinski klasifikator (donji prag), MultinomialNB, logistička
regresija (L2, `class_weight="balanced"`), LinearSVC. Hiperparametar
regularizacije se bira ugnežđenom validacijom.

**Enkoderski:**

| Uloga | Model | Napomena |
|---|---|---|
| monolingvalni / BCMS | `classla/bcms-bertic` | BERTić, ELECTRA, ~110M par., staje na besplatni Colab |
| **višejezični (primarni)** | `xlm-roberta-base` | 278M par., standardni izbor za srpski |
| višejezični (drugi) | `bert-base-multilingual-cased` | slabiji, ali referentna tačka |
| opciono | `classla/xlm-r-bertic` | XLM-R-**large** dotreniran na južnoslovenskim jezicima; ~24 GB VRAM-a |

**Dekoderski:** GPT-4o-mini ili Gemini 2.0 Flash (jeftino, dobri na srpskom),
ili Ollama + Qwen2.5-7B / Llama-3.1-8B ako izbegavate API ključ.

## Instalacija

```bash
pip install scikit-learn pandas numpy matplotlib scipy statsmodels
pip install classla                 # lematizacija
python -c "import classla; classla.download('sr')"
pip install transformers torch      # enkoderski modeli (Colab GPU)
pip install openai                  # ili: pip install google-genai
```

**Stemer:** preuzmite `SerbianStemmer.py` sa
<https://github.com/nikolamilosevic86/SerbianStemmer> i stavite ga pored ovih
skripti. `preprocessing.py` ga automatski detektuje; ako fajl ne postoji,
koristi se ugrađeni rezervni stemer i skripta vas na to upozori. U
dokumentaciji obavezno navedite koji ste zapravo koristili.

Alternativa vredna pomena u izveštaju: **SCStemmers**
(<https://vukbatanovic.github.io/SCStemmers/>) — Java paket sa Kešelj–Šipka
greedy/optimal stemerom, Miloševićevim poboljšanjem i Ljubešić–Pandžić
stemerom. Autor je vaš predavač, pa je poređenje dva stemera lak dodatni poen.

## Detalji na koje treba obratiti pažnju

- **Metrika.** Sentiment je gotovo uvek neuravnotežen — glavna metrika je
  macro-F1, tačnost ide uz nju radi poređenja sa većinskim klasifikatorom.
- **Ćirilica/latinica.** Sve se prevodi u latinicu pre svega ostalog; i
  SerbianStemmer i BERTić rade nad latiničnim tekstom.
- **Curenje podataka.** Vektorizacija je unutar `Pipeline`-a, pa se rečnik i
  IDF računaju samo na trening foldu. Few-shot primeri se izbacuju iz
  evaluacionog skupa.
- **Statistička značajnost.** `statisticko_poredjenje.csv` sadrži Wilcoxonov
  upareni test po foldovima. Razlika od 0.01 macro-F1 između dve varijante
  pretprocesiranja bez tog testa nije nalaz.
- **Izbor broja epoha.** Kriva se dobija evaluacijom na test foldu — to je ono
  što postavka traži. Ako tvrdite da je „N epoha najbolje", napišite i da je
  vrednost izabrana post hoc.
- **Očekivan nalaz.** Kod bogate morfologije srpskog stemovanje i karakterski
  n-grami obično podignu linearne modele; lematizacija često daje sličan efekat
  uz mnogo veću cenu. Uredna priča za diskusiju — samo je potkrepite testom.
