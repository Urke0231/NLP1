#!/usr/bin/env python3
"""
Konvertuje listu JSON objekata (tekst, sentiment, duzina, uneo, izvor)
u tekstualni format sa blokovima:

    duzina: <duzina>
    sentiment: <sentiment>
    izvor: <izvor>
    <tekst>

    ---

Podesi ULAZ i IZLAZ ispod, pa pokreni:  python json_u_tekst.py
"""

import json

# ---- PODESI OVDE ----
ULAZ = "spojeni_dataset.json"   # putanja do ulaznog JSON fajla
IZLAZ = "spojeni_dataset.txt"              # putanja do izlaznog tekstualnog fajla
# ---------------------


def blok(obj: dict) -> str:
    duzina = obj.get("duzina", "") or ""
    sentiment = obj.get("sentiment", "") or ""
    izvor = obj.get("izvor")
    izvor = izvor if izvor else ""          # None / null / prazno -> prazan string
    tekst = obj.get("tekst", "") or ""

    return (
        f"duzina: {duzina}\n"
        f"sentiment: {sentiment}\n"
        f"izvor: {izvor}\n"
        f"{tekst}\n"
        f"\n"
        f"---\n"
    )


def konvertuj(ulaz: str, izlaz: str) -> int:
    with open(ulaz, encoding="utf-8") as f:
        podaci = json.load(f)

    # dozvoli i jedan objekat, ne samo listu
    if isinstance(podaci, dict):
        podaci = [podaci]

    blokovi = [blok(o) for o in podaci]
    with open(izlaz, "w", encoding="utf-8") as f:
        f.write("\n".join(blokovi))

    return len(blokovi)


if __name__ == "__main__":
    broj = konvertuj(ULAZ, IZLAZ)
    print(f"Upisano {broj} blokova u: {IZLAZ}")