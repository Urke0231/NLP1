"""
Automatizovano popunjavanje 'izvor' polja u JSON datasetu, poredjenjem preko
BROJA ZAJEDNICKIH RECI (Jaccard slicnost) umesto difflib-a na celom tekstu.
Brze je od difflib pristupa i dovoljno dobro hvata podudarnost teme/sadrzaja.

Linkovi vise ne dolaze iz Chrome istorije direktno, nego iz CSV fajla
(npr. onog koji pravi izvuci_jul_linkove.py), sa kolonama:
    datum,url,title,visit_count

Za svaku stavku u datasetu automatski se bira URL sa najvecim kombinovanim
skorom (podudaranje reci + vremenska blizina) i odmah upisuje.

Zavisnosti:
    pip install requests beautifulsoup4

Pokretanje:
    python izvori_batch_reci.py
"""

import csv
import json
import os
import re
import tempfile
import datetime
import concurrent.futures as cf

import requests

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

# ---------- PODESAVANJA ----------
LINKOVI_CSV = "./linkovi.csv"
DATASET_PATH = "./Uroseve_anotacije.json"
IZLAZ_PATH = "./dataset_popunjen.json"
IZVESTAJ_PATH = "./izvestaj_skorova.csv"
KES_FAJL = "./stranice_kes.json"

TEZINA_TEKST = 0.7      # koliko utice podudaranje reci
TEZINA_VREME = 0.3      # koliko utice vremenska blizina (redosled poseta)

SAMO_NEDOSTAJUCE = True   # True = popunjava samo stavke bez 'izvor'; False = prepisuje SVE
MIN_DUZINA_RECI = 3        # rec kraca od ovoga se ignorise (predlozi, veznici i sl.)
MAX_WORKERS = 8
TIMEOUT = 12
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"

STOP_RECI = {
    "i", "u", "na", "za", "je", "se", "da", "su", "sa", "od", "do", "iz",
    "ali", "ili", "koji", "koja", "koje", "kao", "što", "sto", "ne", "ni",
    "će", "ce", "bi", "bio", "bila", "bilo", "biti", "ovo", "ova", "ovaj",
    "taj", "ta", "to", "kod", "pre", "posle", "pod", "nad", "po", "kroz",
    "godine", "godini", "godina", "danas", "juce", "juče",
}
# -----------------------------------


def normalizuj(tekst):
    tekst = tekst.lower()
    tekst = re.sub(r"[^\w\sšđčćžŠĐČĆŽ]", " ", tekst)
    tekst = re.sub(r"\s+", " ", tekst).strip()
    return tekst


def reci_skup(tekst):
    reci = normalizuj(tekst).split()
    return {r for r in reci if len(r) >= MIN_DUZINA_RECI and r not in STOP_RECI}


def jaccard_slicnost(reci_a, reci_b):
    if not reci_a or not reci_b:
        return 0.0
    presek = reci_a & reci_b
    unija = reci_a | reci_b
    return len(presek) / len(unija) if unija else 0.0


# ---------- CITANJE LINKOVA IZ CSV-a ----------

def ucitaj_linkove_iz_csv(putanja):
    linkovi = []
    with open(putanja, encoding="utf-8") as f:
        citac = csv.DictReader(f)
        for red in citac:
            url = (red.get("url") or "").strip()
            if not url:
                continue
            datum = None
            datum_str = (red.get("datum") or "").strip()
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
                try:
                    datum = datetime.datetime.strptime(datum_str, fmt)
                    break
                except ValueError:
                    continue
            linkovi.append({
                "url": url,
                "title": (red.get("title") or "").strip(),
                "datum": datum,
            })

    # ako nema datuma, tretiramo ih kao da su na kraju - ali sortiranje po
    # onima koji datum imaju cuva redosled poseta
    linkovi.sort(key=lambda x: x["datum"] or datetime.datetime.max)
    return linkovi


# ---------- PREUZIMANJE CELOG TEKSTA STRANICE ----------

def html_u_tekst(html):
    if HAS_BS4:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
            tag.decompose()
        tekst = soup.get_text(separator=" ")
    else:
        tekst = re.sub(r"(?is)<(script|style).*?>.*?(</\1>)", " ", html)
        tekst = re.sub(r"(?s)<[^>]+>", " ", tekst)
    tekst = re.sub(r"\s+", " ", tekst).strip()
    return tekst


def preuzmi_tekst_stranice(session, url):
    try:
        resp = session.get(url, timeout=TIMEOUT, headers={"User-Agent": USER_AGENT})
        resp.raise_for_status()
        return html_u_tekst(resp.text)
    except Exception as e:
        print(f"  [GRESKA] {url} -> {e}")
        return None


def ucitaj_kes():
    if os.path.exists(KES_FAJL):
        with open(KES_FAJL, encoding="utf-8") as f:
            return json.load(f)
    return {}


def sacuvaj_kes(kes):
    with open(KES_FAJL, "w", encoding="utf-8") as f:
        json.dump(kes, f, ensure_ascii=False)


def preuzmi_sve_stranice(linkovi):
    kes = ucitaj_kes()
    urlovi_za_preuzimanje = [s["url"] for s in linkovi if s["url"] not in kes]

    if urlovi_za_preuzimanje:
        print(f"Preuzimam tekst za {len(urlovi_za_preuzimanje)} stranica "
              f"(vec u kesu: {len(linkovi) - len(urlovi_za_preuzimanje)})...")

        session = requests.Session()
        with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            buducnosti = {
                executor.submit(preuzmi_tekst_stranice, session, url): url
                for url in urlovi_za_preuzimanje
            }
            zavrseno = 0
            for buducnost in cf.as_completed(buducnosti):
                url = buducnosti[buducnost]
                tekst = buducnost.result()
                kes[url] = tekst or ""
                zavrseno += 1
                if zavrseno % 25 == 0 or zavrseno == len(urlovi_za_preuzimanje):
                    print(f"  ...{zavrseno}/{len(urlovi_za_preuzimanje)}")

        sacuvaj_kes(kes)
    else:
        print("Sve stranice su vec u kesu, preskacem preuzimanje.")

    return kes


# ---------- RACUNANJE SKOROVA ----------

def izracunaj_kandidate(reci_stavke, index_stavke, ukupno_stavki, linkovi, reci_stranica_kes):
    if not linkovi:
        return []

    n_lnk = len(linkovi)
    ocekivana_pozicija = index_stavke / max(ukupno_stavki - 1, 1)

    ocenjeno = []
    for i, stavka in enumerate(linkovi):
        reci_stranice = reci_stranica_kes.get(stavka["url"])
        if reci_stranice is None:
            # fallback na naslov ako preuzimanje nije uspelo
            reci_stranice = reci_skup(stavka["title"])

        skor_tekst = jaccard_slicnost(reci_stavke, reci_stranice)
        pozicija_u_listi = i / max(n_lnk - 1, 1)
        skor_vreme = 1 - abs(ocekivana_pozicija - pozicija_u_listi)

        kombinovano = TEZINA_TEKST * skor_tekst + TEZINA_VREME * skor_vreme
        ocenjeno.append({
            "url": stavka["url"],
            "title": stavka["title"],
            "skor_tekst": skor_tekst,
            "skor_vreme": skor_vreme,
            "skor": kombinovano,
        })

    ocenjeno.sort(key=lambda x: x["skor"], reverse=True)
    return ocenjeno


# ---------- GLAVNI TOK ----------

def main():
    if not os.path.exists(LINKOVI_CSV):
        print(f"GRESKA: ne postoji CSV sa linkovima:\n{LINKOVI_CSV}")
        return
    if not os.path.exists(DATASET_PATH):
        print(f"GRESKA: ne postoji dataset:\n{DATASET_PATH}")
        return

    print("Ucitavam linkove iz CSV-a...")
    linkovi = ucitaj_linkove_iz_csv(LINKOVI_CSV)
    print(f"Ucitano {len(linkovi)} linkova.")

    with open(DATASET_PATH, encoding="utf-8") as f:
        dataset = json.load(f)

    tekstovi_stranica = preuzmi_sve_stranice(linkovi)
    print("Racunam skupove reci za sve stranice...")
    reci_stranica_kes = {url: reci_skup(tekst) for url, tekst in tekstovi_stranica.items()}

    if SAMO_NEDOSTAJUCE:
        indeksi = [i for i, s in enumerate(dataset) if not s.get("izvor")]
    else:
        indeksi = list(range(len(dataset)))

    print(f"\nObradjujem {len(indeksi)} stavki...\n")

    izvestaj_redovi = []
    broj_dodeljenih = 0

    for idx in indeksi:
        stavka = dataset[idx]
        reci_stavke = reci_skup(stavka["tekst"])
        kandidati = izracunaj_kandidate(
            reci_stavke, idx, len(dataset), linkovi, reci_stranica_kes)

        if not kandidati:
            print(f"#{idx}: nema kandidata (prazna lista linkova)")
            continue

        najbolji = kandidati[0]
        dataset[idx]["izvor"] = najbolji["url"]
        broj_dodeljenih += 1

        print(f"#{idx}: skor={najbolji['skor']:.3f} "
              f"(tekst={najbolji['skor_tekst']:.3f}, vreme={najbolji['skor_vreme']:.3f}) "
              f"-> {najbolji['url']}")

        izvestaj_redovi.append({
            "index": idx,
            "tekst_pocetak": stavka["tekst"][:80],
            "dodeljeni_url": najbolji["url"],
            "skor": round(najbolji["skor"], 4),
            "skor_tekst": round(najbolji["skor_tekst"], 4),
            "skor_vreme": round(najbolji["skor_vreme"], 4),
            "drugi_kandidat_url": kandidati[1]["url"] if len(kandidati) > 1 else "",
            "drugi_kandidat_skor": round(kandidati[1]["skor"], 4) if len(kandidati) > 1 else "",
        })

    with open(IZLAZ_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    if izvestaj_redovi:
        with open(IZVESTAJ_PATH, "w", encoding="utf-8", newline="") as f:
            pisac = csv.DictWriter(f, fieldnames=list(izvestaj_redovi[0].keys()))
            pisac.writeheader()
            pisac.writerows(izvestaj_redovi)

    print(f"\nGotovo. Dodeljeno {broj_dodeljenih} izvora.")
    print(f"Novi dataset: {IZLAZ_PATH}")
    print(f"Izvestaj skorova (za proveru): {IZVESTAJ_PATH}")


if __name__ == "__main__":
    main()