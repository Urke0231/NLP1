import json
import os
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, messagebox, ttk

OUTPUT_FILE = "dataset.json"
COMPARISON_FILE = "poredjenje_annotacija.json"
ANNOTATOR_NAMES = ["Uros", "Manojlo", "Jelena", "Damjan"]
SENTIMENT_LABELS = {
    1: "Pozitivan (1)",
    -1: "Negativan (-1)",
    0: "Neutralan (0)",
}
SENTIMENT_VALUES = {
    "": None,
    "Pozitivan (1)": 1,
    "Negativan (-1)": -1,
    "Neutralan (0)": 0,
}


def ucitaj_dataset():
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            try:
                # rename corrupted file so user can inspect it
                corrupt_name = OUTPUT_FILE + ".corrupt"
                os.replace(OUTPUT_FILE, corrupt_name)
            except Exception:
                pass
            try:
                messagebox.showwarning(
                    "Upozorenje",
                    f"Fajl {OUTPUT_FILE} je korumpiran i biće resetovan. Sačuvan je kao {corrupt_name}.",
                )
            except Exception:
                pass
            return []
    return []


def sacuvaj_dataset(podaci):
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(podaci, f, ensure_ascii=False, indent=2)


def popravi_prazne_json_vrednosti(sadrzaj):
    rezultat = []
    u_stringu = False
    escape = False
    ocekuje_vrednost = False

    for znak in sadrzaj:
        if u_stringu:
            rezultat.append(znak)
            if escape:
                escape = False
            elif znak == "\\":
                escape = True
            elif znak == '"':
                u_stringu = False
            continue

        if znak == '"':
            u_stringu = True
            ocekuje_vrednost = False
            rezultat.append(znak)
            continue

        if ocekuje_vrednost:
            if znak.isspace():
                rezultat.append(znak)
                continue
            if znak in ",}]":
                rezultat.append('"x"')
                rezultat.append(znak)
                ocekuje_vrednost = False
                continue
            ocekuje_vrednost = False

        rezultat.append(znak)
        if znak == ":":
            ocekuje_vrednost = True

    if ocekuje_vrednost:
        rezultat.append('"x"')

    return "".join(rezultat)


def ucitaj_json_fajl_tolerantno(putanja):
    with open(putanja, "r", encoding="utf-8") as f:
        sadrzaj = f.read()

    try:
        return json.loads(sadrzaj)
    except json.JSONDecodeError as originalna_greska:
        popravljeno = popravi_prazne_json_vrednosti(sadrzaj)
        if popravljeno == sadrzaj:
            raise originalna_greska
        return json.loads(popravljeno)


def normalizuj_sentiment(vrednost):
    if isinstance(vrednost, bool):
        return None
    try:
        sentiment = int(vrednost)
    except (TypeError, ValueError):
        return None
    if sentiment in SENTIMENT_LABELS:
        return sentiment
    return None


def ucitaj_poredjenje():
    if not os.path.exists(COMPARISON_FILE):
        return {}

    try:
        sirovi_podaci = ucitaj_json_fajl_tolerantno(COMPARISON_FILE)
    except (json.JSONDecodeError, OSError):
        return {}

    poredjenje = {}
    if isinstance(sirovi_podaci, list):
        redovi = sirovi_podaci
    elif isinstance(sirovi_podaci, dict):
        redovi = [
            {"tekst": tekst, **(vrednost if isinstance(vrednost, dict) else {})}
            for tekst, vrednost in sirovi_podaci.items()
        ]
    else:
        return {}

    for red in redovi:
        if not isinstance(red, dict):
            continue
        tekst = red.get("tekst")
        if not isinstance(tekst, str) or not tekst.strip():
            continue

        sentimenti = {}
        sirovi_sentimenti = red.get("sentimenti", {})
        if isinstance(sirovi_sentimenti, dict):
            for ime in ANNOTATOR_NAMES:
                sentiment = normalizuj_sentiment(sirovi_sentimenti.get(ime))
                if sentiment is not None:
                    sentimenti[ime] = sentiment

        for ime in ANNOTATOR_NAMES:
            sentiment = normalizuj_sentiment(red.get(ime))
            if sentiment is not None:
                sentimenti[ime] = sentiment

        poredjenje[tekst] = {
            "duzina": red.get("duzina", ""),
            "sentimenti": sentimenti,
            "izmene": red.get("izmene", {}) if isinstance(red.get("izmene"), dict) else {},
            "notes": red.get("notes", "") if isinstance(red.get("notes"), str) else "",
        }

    return poredjenje


def sacuvaj_poredjenje(poredjenje):
    redovi = []
    for tekst in sorted(poredjenje, key=str.casefold):
        red = poredjenje[tekst]
        sentimenti = {}
        for ime in ANNOTATOR_NAMES:
            sentiment = normalizuj_sentiment(red.get("sentimenti", {}).get(ime))
            if sentiment is not None:
                sentimenti[ime] = sentiment

        izmene = {}
        sirove_izmene = red.get("izmene", {})
        if isinstance(sirove_izmene, dict):
            for ime in ANNOTATOR_NAMES:
                izmena = sirove_izmene.get(ime)
                if isinstance(izmena, dict):
                    izmene[ime] = izmena

        redovi.append(
            {
                "tekst": tekst,
                "duzina": red.get("duzina", ""),
                "sentimenti": sentimenti,
                "izmene": izmene,
                "notes": red.get("notes", "") if isinstance(red.get("notes"), str) else "",
            }
        )

    with open(COMPARISON_FILE, "w", encoding="utf-8") as f:
        json.dump(redovi, f, ensure_ascii=False, indent=2)


def duzina_kategorija(tekst, granica_kratak_srednji=30, granica_srednji_dugacak=100):
    broj_reci = len(tekst.split())
    if broj_reci < granica_kratak_srednji:
        return "kratak"
    elif broj_reci < granica_srednji_dugacak:
        return "srednji"
    else:
        return "dugacak"


class Aplikacija(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Prikupljanje podataka - Finansijski sentiment (NBS)")
        self.geometry("600x800")
        self.resizable(True, True)

        self.podaci = ucitaj_dataset()
        self.poredjenje = ucitaj_poredjenje()
        self._azuriranje_poredjenja = False
        # track file modification time to detect external changes
        try:
            if os.path.exists(OUTPUT_FILE):
                self._data_mtime = os.path.getmtime(OUTPUT_FILE)
            else:
                self._data_mtime = None
        except Exception:
            self._data_mtime = None
        self._comparison_mtime = self._mtime_fajla(COMPARISON_FILE)
        self.granica_kratak_var = tk.StringVar(value="30")
        self.granica_srednji_var = tk.StringVar(value="100")

        self._napravi_izgled()
        self._azuriraj_brojac()
        self._azuriraj_duzinu()

        # start monitoring dataset file for external changes
        self.after(1000, self._monitor_file_changes)

    def _napravi_izgled(self):
        self.geometry("1000x800")

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True)

        tab_unos = ttk.Frame(self.notebook)
        tab_poredjenje = ttk.Frame(self.notebook)
        self.notebook.add(tab_unos, text="Unos")
        self.notebook.add(tab_poredjenje, text="Poredjenje")

        okvir_glavni = ttk.Frame(tab_unos, padding=15)
        okvir_glavni.pack(fill="both", expand=True)

        self.labela_naslov = ttk.Label(
            okvir_glavni,
            text="Unos novog posta",
            font=("Segoe UI", 14, "bold"),
        )
        self.labela_naslov.pack(anchor="w")

        self.labela_brojac = ttk.Label(okvir_glavni, text="", font=("Segoe UI", 10))
        self.labela_brojac.pack(anchor="w", pady=(0, 10))

        ttk.Label(okvir_glavni, text="Granice za klasifikaciju duzine:").pack(anchor="w")
        okvir_granice = ttk.Frame(okvir_glavni)
        okvir_granice.pack(fill="x", pady=(0, 10))

        ttk.Label(okvir_granice, text="granicaKratakSrednji").pack(anchor="w")
        self.granica_kratak_entry = ttk.Entry(okvir_granice, textvariable=self.granica_kratak_var, width=10)
        self.granica_kratak_entry.pack(anchor="w", pady=(0, 5))

        ttk.Label(okvir_granice, text="granicaSrednjiDugacak").pack(anchor="w")
        self.granica_srednji_entry = ttk.Entry(okvir_granice, textvariable=self.granica_srednji_var, width=10)
        self.granica_srednji_entry.pack(anchor="w")

        self.granica_kratak_var.trace_add("write", lambda *_: self._azuriraj_duzinu())
        self.granica_srednji_var.trace_add("write", lambda *_: self._azuriraj_duzinu())

        ttk.Label(okvir_glavni, text="Tekst posta:").pack(anchor="w", pady=(10, 0))
        self.tekst_polje = tk.Text(okvir_glavni, height=12, wrap="word", font=("Segoe UI", 10))
        self.tekst_polje.pack(fill="both", expand=True, pady=(0, 10))
        self.tekst_polje.bind("<KeyRelease>", self._azuriraj_duzinu)

        self.labela_duzina = ttk.Label(okvir_glavni, text="Duzina: -", font=("Segoe UI", 10, "italic"))
        self.labela_duzina.pack(anchor="w", pady=(0, 10))

        self.labela_kombinacije = ttk.Label(
            okvir_glavni,
            text="",
            font=("Consolas", 10),
            justify="left",
            wraplength=650,
        )
        self.labela_kombinacije.pack(anchor="w", pady=(0, 10))

        ttk.Label(okvir_glavni, text="Sentiment:").pack(anchor="w")
        self.sentiment_var = tk.IntVar(value=-99)  # -99 = jos nije izabrano
        okvir_sentiment = ttk.Frame(okvir_glavni)
        okvir_sentiment.pack(anchor="w", pady=(0, 15))

        for tekst_dugme, vrednost in [
            ("Pozitivan", 1),
            ("Negativan", -1),
            ("Neutralan", 0),
        ]:
            ttk.Radiobutton(
                okvir_sentiment,
                text=tekst_dugme,
                variable=self.sentiment_var,
                value=vrednost,
            ).pack(side="left", padx=(0, 15))

        okvir_dugmici = ttk.Frame(okvir_glavni)
        okvir_dugmici.pack(fill="x")

        ttk.Button(okvir_dugmici, text="Sacuvaj unos", command=self._sacuvaj_unos).pack(
            side="left", padx=(0, 10)
        )
        ttk.Button(okvir_dugmici, text="Obrisi polja", command=self._obrisi_polja).pack(
            side="left"
        )

        self._napravi_tab_poredjenje(tab_poredjenje)

    def _napravi_tab_poredjenje(self, roditelj):
        okvir_glavni = ttk.Frame(roditelj, padding=15)
        okvir_glavni.pack(fill="both", expand=True)

        ttk.Label(
            okvir_glavni,
            text="Poredjenje anotacija",
            font=("Segoe UI", 14, "bold"),
        ).pack(anchor="w")

        self.labela_poredjenje_status = ttk.Label(okvir_glavni, text="")
        self.labela_poredjenje_status.pack(anchor="w", pady=(0, 10))

        okvir_import = ttk.Frame(okvir_glavni)
        okvir_import.pack(fill="x", pady=(0, 10))

        ttk.Label(okvir_import, text="Anotator:").pack(side="left", padx=(0, 6))
        self.import_annotator_var = tk.StringVar(value=ANNOTATOR_NAMES[0])
        self.import_annotator_combo = ttk.Combobox(
            okvir_import,
            textvariable=self.import_annotator_var,
            values=ANNOTATOR_NAMES,
            state="readonly",
            width=14,
        )
        self.import_annotator_combo.pack(side="left", padx=(0, 10))

        ttk.Button(
            okvir_import,
            text="Uvezi JSON",
            command=self._uvezi_poredjenje_fajl,
        ).pack(side="left", padx=(0, 10))

        ttk.Button(
            okvir_import,
            text="Sacuvaj poredjenje",
            command=self._sacuvaj_poredjenje_ui,
        ).pack(side="left")

        paned_poredjenje = tk.PanedWindow(
            okvir_glavni,
            orient=tk.VERTICAL,
            sashwidth=6,
            sashrelief="raised",
        )
        paned_poredjenje.pack(fill="both", expand=True)

        okvir_tabela = ttk.Frame(paned_poredjenje)
        paned_poredjenje.add(okvir_tabela, minsize=180)

        kolone = ["broj", "tekst", *ANNOTATOR_NAMES, "notes"]
        self.tabela_poredjenje = ttk.Treeview(
            okvir_tabela,
            columns=kolone,
            show="headings",
            selectmode="browse",
        )
        self.tabela_poredjenje.heading("broj", text="#")
        self.tabela_poredjenje.column("broj", width=50, minwidth=45, anchor="e", stretch=False)
        self.tabela_poredjenje.heading("tekst", text="Tekst")
        self.tabela_poredjenje.column("tekst", width=470, minwidth=250, stretch=True)
        for ime in ANNOTATOR_NAMES:
            self.tabela_poredjenje.heading(ime, text=ime)
            self.tabela_poredjenje.column(ime, width=115, minwidth=95, anchor="center", stretch=False)
        self.tabela_poredjenje.heading("notes", text="Notes")
        self.tabela_poredjenje.column("notes", width=220, minwidth=140, stretch=False)
        self.tabela_poredjenje.tag_configure("neutral_neparni_red", background="#ffffff")
        self.tabela_poredjenje.tag_configure("neutral_parni_red", background="#f2f2f2")
        self.tabela_poredjenje.tag_configure("zelen_neparni_red", background="#ebf7ed")
        self.tabela_poredjenje.tag_configure("zelen_parni_red", background="#dff0e3")
        self.tabela_poredjenje.tag_configure("crven_neparni_red", background="#fde8e8")
        self.tabela_poredjenje.tag_configure("crven_parni_red", background="#f8d8d8")

        scroll_y = ttk.Scrollbar(okvir_tabela, orient="vertical", command=self.tabela_poredjenje.yview)
        scroll_x = ttk.Scrollbar(okvir_tabela, orient="horizontal", command=self.tabela_poredjenje.xview)
        self.tabela_poredjenje.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)

        self.tabela_poredjenje.grid(row=0, column=0, sticky="nsew")
        scroll_y.grid(row=0, column=1, sticky="ns")
        scroll_x.grid(row=1, column=0, sticky="ew")
        okvir_tabela.columnconfigure(0, weight=1)
        okvir_tabela.rowconfigure(0, weight=1)

        self.tabela_poredjenje.bind("<<TreeviewSelect>>", self._na_izbor_poredjenje_reda)

        okvir_izmena = ttk.LabelFrame(paned_poredjenje, text="Izmena izabranog teksta", padding=10)
        paned_poredjenje.add(okvir_izmena, minsize=180)

        okvir_tekst_preview = ttk.Frame(okvir_izmena)
        okvir_tekst_preview.pack(fill="both", expand=True, pady=(0, 10))

        self.poredjenje_tekst_preview = tk.Text(okvir_tekst_preview, height=10, wrap="word", font=("Segoe UI", 11))
        scroll_preview = ttk.Scrollbar(okvir_tekst_preview, orient="vertical", command=self.poredjenje_tekst_preview.yview)
        self.poredjenje_tekst_preview.configure(yscrollcommand=scroll_preview.set)
        self.poredjenje_tekst_preview.grid(row=0, column=0, sticky="nsew")
        scroll_preview.grid(row=0, column=1, sticky="ns")
        okvir_tekst_preview.columnconfigure(0, weight=1)
        okvir_tekst_preview.rowconfigure(0, weight=1)
        self.poredjenje_tekst_preview.configure(state="disabled")

        okvir_sentimenti = ttk.Frame(okvir_izmena)
        okvir_sentimenti.pack(fill="x")

        self.poredjenje_sentiment_vars = {}
        sentiment_opcije = ["", SENTIMENT_LABELS[1], SENTIMENT_LABELS[-1], SENTIMENT_LABELS[0]]
        for indeks, ime in enumerate(ANNOTATOR_NAMES):
            okvir_osoba = ttk.Frame(okvir_sentimenti)
            okvir_osoba.grid(row=0, column=indeks, sticky="ew", padx=(0, 10))
            okvir_sentimenti.columnconfigure(indeks, weight=1)

            ttk.Label(okvir_osoba, text=ime).pack(anchor="w")
            var = tk.StringVar(value="")
            combo = ttk.Combobox(
                okvir_osoba,
                textvariable=var,
                values=sentiment_opcije,
                state="readonly",
                width=18,
            )
            combo.pack(fill="x")
            combo.bind("<<ComboboxSelected>>", lambda event, osoba=ime: self._promeni_poredjenje_sentiment(osoba))
            self.poredjenje_sentiment_vars[ime] = var

        okvir_notes = ttk.Frame(okvir_sentimenti)
        okvir_notes.grid(row=0, column=len(ANNOTATOR_NAMES), sticky="nsew")
        okvir_sentimenti.columnconfigure(len(ANNOTATOR_NAMES), weight=2)

        ttk.Label(okvir_notes, text="Notes").pack(anchor="w")
        self.poredjenje_notes_polje = tk.Text(okvir_notes, height=3, wrap="word", font=("Segoe UI", 10))
        self.poredjenje_notes_polje.pack(fill="both", expand=True)
        self.poredjenje_notes_polje.bind("<KeyRelease>", self._rasporedi_sacuvaj_notes_poredjenja)
        self.poredjenje_notes_polje.bind("<FocusOut>", self._sacuvaj_notes_poredjenja)
        self.poredjenje_notes_polje.bind("<Control-s>", self._sacuvaj_notes_poredjenja)

        self._azuriraj_tabelu_poredjenja()

    def _skracen_tekst(self, tekst, granica=120):
        jednolinijski = " ".join(tekst.split())
        if len(jednolinijski) <= granica:
            return jednolinijski
        return jednolinijski[: granica - 3] + "..."

    def _formatiraj_sentiment(self, sentiment):
        sentiment = normalizuj_sentiment(sentiment)
        if sentiment is None:
            return ""
        return str(sentiment)

    def _tag_za_poredjenje_red(self, sentimenti, indeks):
        vrednosti = [normalizuj_sentiment(sentimenti.get(ime)) for ime in ANNOTATOR_NAMES]
        svi_popunjeni = all(vrednost is not None for vrednost in vrednosti)
        red = "parni_red" if indeks % 2 else "neparni_red"
        if not svi_popunjeni:
            return f"neutral_{red}"
        boja = "crven" if len(set(vrednosti)) > 1 else "zelen"
        return f"{boja}_{red}"

    def _azuriraj_tabelu_poredjenja(self):
        if not hasattr(self, "tabela_poredjenje"):
            return

        izabrani_tekst = self._trenutni_poredjenje_tekst()
        self.tabela_poredjenje.delete(*self.tabela_poredjenje.get_children())
        self.poredjenje_iid_to_tekst = {}

        for indeks, tekst in enumerate(sorted(self.poredjenje, key=str.casefold)):
            iid = f"red_{indeks}"
            self.poredjenje_iid_to_tekst[iid] = tekst
            red = self.poredjenje[tekst]
            sentimenti = red.get("sentimenti", {})
            vrednosti = [str(indeks + 1), self._skracen_tekst(tekst)]
            for ime in ANNOTATOR_NAMES:
                vrednosti.append(self._formatiraj_sentiment(sentimenti.get(ime)))
            vrednosti.append(self._skracen_tekst(red.get("notes", ""), granica=80))
            tag = self._tag_za_poredjenje_red(sentimenti, indeks)
            self.tabela_poredjenje.insert("", "end", iid=iid, values=vrednosti, tags=(tag,))

        if izabrani_tekst in self.poredjenje:
            for iid, tekst in self.poredjenje_iid_to_tekst.items():
                if tekst == izabrani_tekst:
                    self.tabela_poredjenje.selection_set(iid)
                    self.tabela_poredjenje.see(iid)
                    break
        else:
            self._popuni_formu_poredjenja(None)

        self._azuriraj_poredjenje_status()

    def _azuriraj_poredjenje_status(self):
        ukupno = len(self.poredjenje)
        po_osobi = {ime: 0 for ime in ANNOTATOR_NAMES}
        neslaganja = 0

        for red in self.poredjenje.values():
            vrednosti = []
            for ime in ANNOTATOR_NAMES:
                sentiment = normalizuj_sentiment(red.get("sentimenti", {}).get(ime))
                if sentiment is not None:
                    po_osobi[ime] += 1
                    vrednosti.append(sentiment)
            if len(vrednosti) == len(ANNOTATOR_NAMES) and len(set(vrednosti)) > 1:
                neslaganja += 1

        detalji = "  |  ".join(f"{ime}: {po_osobi[ime]}" for ime in ANNOTATOR_NAMES)
        self.labela_poredjenje_status.config(
            text=f"Tekstova: {ukupno}  |  Neslaganja: {neslaganja}  |  {detalji}"
        )

    def _trenutni_poredjenje_tekst(self):
        if not hasattr(self, "tabela_poredjenje"):
            return None
        izbor = self.tabela_poredjenje.selection()
        if not izbor:
            return None
        return getattr(self, "poredjenje_iid_to_tekst", {}).get(izbor[0])

    def _na_izbor_poredjenje_reda(self, event=None):
        self._popuni_formu_poredjenja(self._trenutni_poredjenje_tekst())

    def _popuni_formu_poredjenja(self, tekst):
        self._azuriranje_poredjenja = True
        try:
            self.poredjenje_tekst_preview.configure(state="normal")
            self.poredjenje_tekst_preview.delete("1.0", "end")
            if tekst:
                self.poredjenje_tekst_preview.insert("1.0", tekst)
            self.poredjenje_tekst_preview.configure(state="disabled")

            red = self.poredjenje.get(tekst, {}) if tekst else {}
            sentimenti = red.get("sentimenti", {})
            for ime, var in self.poredjenje_sentiment_vars.items():
                sentiment = normalizuj_sentiment(sentimenti.get(ime))
                var.set(SENTIMENT_LABELS.get(sentiment, ""))
            self.poredjenje_notes_polje.delete("1.0", "end")
            if tekst:
                self.poredjenje_notes_polje.insert("1.0", red.get("notes", ""))
        finally:
            self._azuriranje_poredjenja = False

    def _promeni_poredjenje_sentiment(self, ime):
        if self._azuriranje_poredjenja:
            return

        tekst = self._trenutni_poredjenje_tekst()
        if not tekst:
            return

        vrednost = SENTIMENT_VALUES.get(self.poredjenje_sentiment_vars[ime].get())
        red = self.poredjenje.setdefault(tekst, {"duzina": "", "sentimenti": {}, "izmene": {}, "notes": ""})
        sentimenti = red.setdefault("sentimenti", {})
        if vrednost is None:
            sentimenti.pop(ime, None)
        else:
            sentimenti[ime] = vrednost
        red.setdefault("izmene", {})[ime] = {
            "sentiment": vrednost,
            "vreme": datetime.now().isoformat(timespec="seconds"),
        }

        sacuvaj_poredjenje(self.poredjenje)
        self._comparison_mtime = self._mtime_fajla(COMPARISON_FILE)
        self._azuriraj_tabelu_poredjenja()

    def _rasporedi_sacuvaj_notes_poredjenja(self, event=None):
        if self._azuriranje_poredjenja:
            return

        prethodni_id = getattr(self, "_notes_save_after_id", None)
        if prethodni_id is not None:
            try:
                self.after_cancel(prethodni_id)
            except tk.TclError:
                pass
        self._notes_save_after_id = self.after(600, self._sacuvaj_notes_poredjenja)

    def _sacuvaj_notes_poredjenja(self, event=None):
        self._notes_save_after_id = None
        prekini_precicu = event is not None and getattr(event, "keysym", "").lower() == "s"
        if self._azuriranje_poredjenja:
            return "break" if prekini_precicu else None

        tekst = self._trenutni_poredjenje_tekst()
        if not tekst:
            return "break" if prekini_precicu else None

        notes = self.poredjenje_notes_polje.get("1.0", "end-1c")
        red = self.poredjenje.setdefault(tekst, {"duzina": "", "sentimenti": {}, "izmene": {}, "notes": ""})
        if red.get("notes", "") == notes:
            return "break" if prekini_precicu else None

        red["notes"] = notes
        sacuvaj_poredjenje(self.poredjenje)
        self._comparison_mtime = self._mtime_fajla(COMPARISON_FILE)

        izbor = self.tabela_poredjenje.selection()
        if izbor:
            vrednosti = list(self.tabela_poredjenje.item(izbor[0], "values"))
            if vrednosti:
                vrednosti[-1] = self._skracen_tekst(notes, granica=80)
                self.tabela_poredjenje.item(izbor[0], values=vrednosti)

        return "break" if prekini_precicu else None

    def _sacuvaj_poredjenje_ui(self):
        if hasattr(self, "poredjenje_notes_polje"):
            self._sacuvaj_notes_poredjenja()
        sacuvaj_poredjenje(self.poredjenje)
        self._comparison_mtime = self._mtime_fajla(COMPARISON_FILE)
        messagebox.showinfo("Sacuvano", f"Poredjenje je sacuvano u {COMPARISON_FILE}.")

    def _uvezi_poredjenje_fajl(self):
        ime = self.import_annotator_var.get()
        if ime not in ANNOTATOR_NAMES:
            messagebox.showwarning("Greska", "Izaberite anotatora pre uvoza.")
            return

        putanja = filedialog.askopenfilename(
            title="Izaberite dataset JSON",
            filetypes=[("JSON fajlovi", "*.json"), ("Svi fajlovi", "*.*")],
        )
        if not putanja:
            return

        try:
            redovi = ucitaj_json_fajl_tolerantno(putanja)
        except (json.JSONDecodeError, OSError) as greska:
            messagebox.showerror("Greska", f"Ne mogu da ucitam fajl:\n{greska}")
            return

        if not isinstance(redovi, list):
            messagebox.showwarning("Greska", "JSON mora biti lista objekata kao dataset.json.")
            return

        uvezeno = 0
        bez_sentimenta = 0
        preskoceno = 0
        for red in redovi:
            if not isinstance(red, dict):
                preskoceno += 1
                continue

            tekst = red.get("tekst")
            if not isinstance(tekst, str) or not tekst.strip():
                preskoceno += 1
                continue
            tekst = tekst.strip()
            sentiment = normalizuj_sentiment(red.get("sentiment"))

            zapis = self.poredjenje.setdefault(
                tekst,
                {
                    "duzina": red.get("duzina", ""),
                    "sentimenti": {},
                    "izmene": {},
                    "notes": "",
                },
            )
            if not zapis.get("duzina") and red.get("duzina"):
                zapis["duzina"] = red.get("duzina")
            if sentiment is None:
                bez_sentimenta += 1
            else:
                zapis.setdefault("sentimenti", {})[ime] = sentiment
            uvezeno += 1

        sacuvaj_poredjenje(self.poredjenje)
        self._comparison_mtime = self._mtime_fajla(COMPARISON_FILE)
        self._azuriraj_tabelu_poredjenja()
        messagebox.showinfo(
            "Uvoz zavrsen",
            f"Uvezeno za {ime}: {uvezeno}\nBez sentimenta: {bez_sentimenta}\nPreskoceno: {preskoceno}",
        )

    def _parsiraj_granicu(self, vrednost, default):
        try:
            broj = int(vrednost)
        except (TypeError, ValueError):
            return None
        if broj < 1:
            return None
        return broj

    def _ucitaj_granice(self):
        granica_kratak = self._parsiraj_granicu(self.granica_kratak_var.get(), 30)
        granica_srednji = self._parsiraj_granicu(self.granica_srednji_var.get(), 100)
        if granica_kratak is None or granica_srednji is None:
            return None, None
        if granica_kratak >= granica_srednji:
            return None, None
        return granica_kratak, granica_srednji

    def _mtime_fajla(self, putanja):
        try:
            if os.path.exists(putanja):
                return os.path.getmtime(putanja)
        except Exception:
            pass
        return None

    def _reload_dataset_from_file(self):
        """Reload dataset from disk and update UI if it changed externally."""
        nova = ucitaj_dataset()
        if nova != self.podaci:
            self.podaci = nova
            self._azuriraj_brojac()

    def _reload_poredjenje_from_file(self):
        """Reload comparison data from disk and update UI if it changed externally."""
        novo_poredjenje = ucitaj_poredjenje()
        if novo_poredjenje != self.poredjenje:
            self.poredjenje = novo_poredjenje
            self._azuriraj_tabelu_poredjenja()

    def _monitor_file_changes(self):
        """Poll dataset and comparison files for external modifications or deletion."""
        mtime = self._mtime_fajla(OUTPUT_FILE)

        if mtime != getattr(self, "_data_mtime", None):
            # file was created/modified/deleted externally
            self._data_mtime = mtime
            self._reload_dataset_from_file()

        comparison_mtime = self._mtime_fajla(COMPARISON_FILE)
        if comparison_mtime != getattr(self, "_comparison_mtime", None):
            self._comparison_mtime = comparison_mtime
            self._reload_poredjenje_from_file()

        # reschedule
        self.after(1000, self._monitor_file_changes)

    def _azuriraj_duzinu(self, event=None):
        tekst = self.tekst_polje.get("1.0", "end").strip()
        granica_kratak, granica_srednji = self._ucitaj_granice()
        if tekst and granica_kratak is not None and granica_srednji is not None:
            kategorija = duzina_kategorija(tekst, granica_kratak, granica_srednji)
            broj_reci = len(tekst.split())
            self.labela_duzina.config(text=f"Duzina: {kategorija} ({broj_reci} reci)")
        else:
            self.labela_duzina.config(text="Duzina: -")

    def _azuriraj_brojac(self):
        ukupno = len(self.podaci)
        po_klasi = {1: 0, -1: 0, 0: 0}
        kombinacije = {}
        for duzina in ["kratak", "srednji", "dugacak"]:
            for sentiment in [1, -1, 0]:
                kombinacije[(duzina, sentiment)] = 0

        for unos in self.podaci:
            s = unos.get("sentiment")
            if s in po_klasi:
                po_klasi[s] += 1
            duzina = unos.get("duzina")
            if (duzina, s) in kombinacije:
                kombinacije[(duzina, s)] += 1

        self.labela_brojac.config(
            text=(
                f"Ukupno unosa: {ukupno}  |  "
                f"Pozitivan (1): {po_klasi[1]}  "
                f"Negativan (-1): {po_klasi[-1]}  "
                f"Neutralan (0): {po_klasi[0]}"
            )
        )

        sentimeti = [
            (1, "Pozitivan"),
            (-1, "Negativan"),
            (0, "Neutralan"),
        ]
        duzine = ["kratak", "srednji", "dugacak"]

        matrix_lines = [
            "Kombinacije (duzina x sentiment):",
            "",
            f"{'Duzina\\Sentiment':<18}{'Pozitivan':>12}{'Negativan':>12}{'Neutralan':>12}",
            f"{'-' * 18}{'-' * 12}{'-' * 12}{'-' * 12}",
        ]
        for duzina in duzine:
            row_values = [duzina.capitalize()]
            for sentiment, _ in sentimeti:
                row_values.append(str(kombinacije[(duzina, sentiment)]))
            matrix_lines.append(f"{row_values[0]:<18}{row_values[1]:>12}{row_values[2]:>12}{row_values[3]:>12}")

        self.labela_kombinacije.config(text="\n".join(matrix_lines))

    def _obrisi_polja(self):
        self.tekst_polje.delete("1.0", "end")
        self.sentiment_var.set(-99)
        self._azuriraj_duzinu()

    def _sacuvaj_unos(self):
        tekst = self.tekst_polje.get("1.0", "end").strip()
        sentiment = self.sentiment_var.get()
        granica_kratak, granica_srednji = self._ucitaj_granice()

        if not tekst:
            messagebox.showwarning("Greska", "Tekst posta ne sme biti prazan.")
            return
        if sentiment not in (1, -1, 0):
            messagebox.showwarning("Greska", "Morate izabrati sentiment.")
            return
        if granica_kratak is None or granica_srednji is None:
            messagebox.showwarning("Greska", "Unesite validne granice za klasifikaciju duzine.")
            return

        unos = {
            "tekst": tekst,
            "sentiment": sentiment,
            "duzina": duzina_kategorija(tekst, granica_kratak, granica_srednji),
        }
        self.podaci.append(unos)
        sacuvaj_dataset(self.podaci)
        try:
            # update tracked mtime after saving
            self._data_mtime = os.path.getmtime(OUTPUT_FILE)
        except Exception:
            self._data_mtime = None
        self._azuriraj_brojac()
        self._obrisi_polja()
        messagebox.showinfo("Sacuvano", "Unos je uspesno dodat u dataset.")


if __name__ == "__main__":
    app = Aplikacija()
    app.mainloop()
