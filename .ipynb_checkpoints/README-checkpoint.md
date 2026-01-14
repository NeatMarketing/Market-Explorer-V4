Market Explorer : Neat

Market Explorer is an internal data application designed to help Sales and Strategy teams quickly understand market size, structure, and priorities across zones and verticals.

The tool provides a clear, visual, and actionable view of markets, enabling better commercial discussions and go-to-market decisions.

---

Key features : 

- Zone & Vertical filtering
- Market KPIs (total market size, number of companies, etc.)
- Breakdown by sub-vertical (share of total market)
- Top countries & top companies views
- Automatic insights for sales prioritization
- CSV export for further analysis

---

Structure du Projet : 

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

Se placer au niveau du dossier dans le terminal : 

cd "Neat/MARKETING SH - Documents/Market_Sizing"

Télécharger les requirements : 

pip install -r requirements.txt


Lancer l'application : 

streamlit run app.py

The app will open automatically in your browser.

