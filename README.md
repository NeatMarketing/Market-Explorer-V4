Market Explorer : Neat

Application interne de data conçue pour aider les équipes Sales & Strategy à comprendre rapidement la taille, la structure et les priorités des marchés par zones et verticales.

L’outil fournit une vue claire, visuelle et actionnable des marchés pour faciliter les discussions commerciales et les décisions go-to-market.

---

## Fonctionnalités clés

- Zone & Vertical filtering
- Market KPIs (total market size, number of companies, etc.)
- Breakdown by sub-vertical (share of total market)
- Top countries & top companies views
- Automatic insights for sales prioritization
- CSV export for further analysis

---

## Structure du projet

```
Market_Sizing/
│
├── app.py                 # Bootstrap / redirect (not a UI page)
├── market_sizing.py       # Data loading, KPIs, business logic
│
├── Data/                  # Raw data
├── Data_Clean/            # Cleaned CSV datasets
│
├── pages/                 # Streamlit pages (MUST be lowercase)
│   ├── 0_Home.py
│   ├── 1_Market_Explorer.py
│   └── 2_Company_Business_Plan.py
│
├── .streamlit/
│   └── config.toml        # Hide default multipage navigation
│
├── README.md
└── requirements.txt

```

Comment faire fonctionner l'app ? : 
---

## Prérequis

- Python 3.10+
- pip

Se placer au niveau du dossier dans le terminal : 
---

cd "Neat/MARKETING SH - Documents/Market_Sizing"

## Installation

Depuis la racine du dépôt :

```bash
pip install -r requirements.txt
```

---

## Lancer l’application

```bash
streamlit run app.py
```
L’application s’ouvrira automatiquement dans votre navigateur.

---

## Accès distant (optionnel)

```bash
cloudflared tunnel --url http://localhost:8501
```

---

## Notes

- Les données brutes sont stockées dans `Data/` et les fichiers nettoyés dans `Data_Clean/`.
- Les pages Streamlit doivent rester en minuscules dans le dossier `pages/`.