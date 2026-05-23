# Fuzzy-Decision-Maker
This is a project developed for the "Modelarea si optimizarea deciziei economice" module at my university.

## Descriere
Aplicație interactivă pentru modelul fuzzy de decizie cu agregare prin **media aritmetică**.  
Implementează regulile R1, R3, R4 cu funcții de apartenența triunghiulare.

## Instalare și rulare

```bash
# 1. Instalează dependințele (o singură dată)
pip install streamlit matplotlib numpy pandas

# 2. Rulează aplicația
streamlit run app.py
```

Aplicația se deschide automat în browser la `http://localhost:8501`

## Problema test (date implicite)
- Mulțimea alternativelor: [12, 40]
- Cost de producție: 12  →  2×cost = 24
- Prețul concurenței: 22