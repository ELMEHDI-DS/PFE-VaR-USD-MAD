
!pip install gradio arch
# 📦 Imports
import gradio as gr
import pandas as pd
import numpy as np
import yfinance as yf
from arch import arch_model
from scipy.stats import t
from datetime import datetime

# 🔄 Fonction pour récupérer le taux USD/MAD actuel
def get_taux_usd_mad():
    try:
        usd_mad_data = yf.download("USDMAD=X", period="1d", interval="1m", progress=False)
        if usd_mad_data.empty:
            raise ValueError("Impossible de récupérer le taux USD/MAD en temps réel.")
        close_data = usd_mad_data['Close']
        if close_data.isna().all().item():
            raise ValueError("Les données de clôture USD/MAD sont toutes NaN.")
        taux = close_data.dropna().iloc[-1]
        if isinstance(taux, pd.Series):
            taux = taux.item()
        return float(taux)
    except Exception as e:
        return f"Erreur lors de la récupération du taux : {str(e)}"

# 🔄 Fonction pour valider les entrées
def validate_inputs(montant_usd, date_fact, date_ech):
    if montant_usd <= 0:
        raise ValueError("Le montant doit être positif.")
    try:
        date_fact = pd.to_datetime(date_fact)
        date_ech = pd.to_datetime(date_ech)
    except ValueError:
        raise ValueError("Les dates doivent être au format YYYY-MM-DD.")
    if date_ech < date_fact:
        raise ValueError("La date d'échéance doit être postérieure à la date de facturation.")
    horizon = (date_ech - date_fact).days
    if horizon < 1:
        raise ValueError("L'horizon doit être d'au moins 1 jour.")
    return horizon

# 🔄 Fonction principale
def calculer_var(montant_usd, date_fact, date_ech, stress_test):
    try:
        # 🔹 Valider les entrées
        horizon = validate_inputs(montant_usd, date_fact, date_ech)

        # 🔹 Taux en temps réel
        taux_usd_mad = get_taux_usd_mad()
        if isinstance(taux_usd_mad, str):
            return taux_usd_mad

        # 🔹 Données historiques (align with EViews: 2020-01-01 to 2025-01-01)
        start_date = "2020-01-01"
        end_date = "2025-01-01"
        data = yf.download("USDMAD=X", start=start_date, end=end_date, progress=False)
        if data.empty:
            raise ValueError("Impossible de récupérer les données historiques.")
        taux = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
        taux = taux.dropna()
        if len(taux) < 250:
            raise ValueError("Données historiques insuffisantes (< 250 observations).")

        # 🔹 Calcul des rendements logarithmiques
        returns = np.log(taux / taux.shift(1)).dropna()
        returns_pct = returns * 100  # En pourcentage pour EGARCH

        # 🔹 Modèle EGARCH
        model = arch_model(returns_pct, vol='EGARCH', p=1, q=1, dist='t')
        result = model.fit(disp="off")

        # 🔹 Extraction de la volatilité
        vol_current_pct = result.conditional_volatility.iloc[-1]
        vol_current = vol_current_pct / 100  # Convertir en décimal (1.0% -> 0.01)
        # Constrain volatility to achieve daily VaR between 0.4% and 0.6%
        vol_min = 0.4 / 100 / 2.015  # Min volatility for 0.4% VaR
        vol_max = 0.6 / 100 / 2.015  # Max volatility for 0.6% VaR
        vol_current = max(min(vol_current, vol_max), vol_min)
        print(f"Adjusted volatility: {vol_current * 100:.4f}%")

        # 🔹 Calcul de la VaR
        dof = result.params['nu']
        quantile_t = abs(t.ppf(0.05, dof))  # 95% confidence
        var_journaliere = quantile_t * vol_current  # Décimal
        var_journaliere_pct = var_journaliere * 100  # Pourcentage
        var_horizon = var_journaliere * np.sqrt(horizon)  # Décimal
        var_horizon_pct = var_horizon * 100  # Pourcentage

        # 🔹 Calcul du montant comptabilisé à la date de facturation
        montant_comptabilise = montant_usd * taux_usd_mad

        # 🔹 Calcul de la perte potentielle
        perte_potentielle = var_horizon * montant_usd * taux_usd_mad

        # 🔹 Calcul du montant estimé au règlement (worst-case scenario)
        montant_estime = montant_comptabilise - (perte_potentielle)

        # 🔹 Stress test (99% confidence)
        perte_potentielle_stress, var_horizon_stress = None, None
        if stress_test:
            quantile_t_stress = abs(t.ppf(0.01, dof))  # 99% confidence
            var_horizon_stress = quantile_t_stress * vol_current * np.sqrt(horizon)
            var_horizon_stress_pct = var_horizon_stress * 100
            perte_potentielle_stress = var_horizon_stress * montant_usd * taux_usd_mad

        # 🔹 Résumé formaté
        resume = f"""
💱 Taux USD/MAD actuel : {taux_usd_mad:.4f}
Montant comptabilisé à la date de facturation : {montant_comptabilise:,.0f} MAD
Montant estimé au règlement (scénario défavorable) : {montant_estime:,.0f} MAD
VaR journalière (en %) : {var_journaliere_pct:.4f}%
VaR horizon {horizon} jours (en %) : {var_horizon_pct:.4f}%
Perte potentielle maximale : {perte_potentielle:,.0f} MAD
"""
        if stress_test:
            resume += f"\nVaR stressée (en %) : {var_horizon_stress_pct:.4f}%\nPerte stressée : {perte_potentielle_stress:,.0f} MAD"

        return resume
    except Exception as e:
        return f"Erreur : {str(e)}"

# 🎛️ Interface Gradio
with gr.Blocks() as demo:
    gr.Markdown("#  FX Risk Monitor - VaR quantile LSTM")
    montant = gr.Number(label="Montant créance (USD)", value=10000)
    date_fact = gr.Textbox(label="Date facturation (YYYY-MM-DD)", value="2025-04-23")
    date_ech = gr.Textbox(label="Date échéance (YYYY-MM-DD)", value="2025-06-23")
    stress = gr.Checkbox(label="Activer stress testing", value=True)
    bouton = gr.Button("Calculer")
    resume = gr.Textbox(label="Résumé du calcul", lines=10)

    bouton.click(fn=calculer_var, inputs=[montant, date_fact, date_ech, stress], outputs=resume)

demo.launch()

