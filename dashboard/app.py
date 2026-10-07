"""Tableau de bord Streamlit (interface en français).

Il lit un instantané statique (dashboard/demo_data) : aucun AWS, aucune base de données et aucune API
ne sont nécessaires. Si API_URL est défini et joignable, le formulaire « Tester une mesure » appelle l'API en direct.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

DEMO = Path(os.getenv("DEMO_DIR", Path(__file__).parent / "demo_data"))
API_URL = os.getenv("API_URL", "").rstrip("/")

INK, MUTED, LINE, PAPER = "#16202C", "#5F6B7A", "#E3E7EC", "#F5F6F8"
ACCENT = "#1F5FBF"
RISK_COLORS = {"HIGH": "#C23B22", "MEDIUM": "#D98B0B", "LOW": "#6F8FAF"}
RISK_LABELS = {"HIGH": "Élevé", "MEDIUM": "Moyen", "LOW": "Faible"}
SENSORS = {"voltage": "Tension", "rotation": "Rotation", "pressure": "Pression", "vibration": "Vibration"}
SENSOR_COLORS = {"Pression": "#1F5FBF", "Rotation": "#6B7785", "Vibration": "#0E8A7B", "Tension": "#8E5BB5"}

FACTORS_FR = {
    "vibration (3 h average)": "vibration (moyenne 3 h)",
    "vibration (24 h average)": "vibration (moyenne 24 h)",
    "vibration (6 h peak)": "vibration (pic sur 6 h)",
    "vibration change over 3 h": "variation de la vibration sur 3 h",
    "vibration variability (24 h)": "variabilité de la vibration (24 h)",
    "pressure (3 h average)": "pression (moyenne 3 h)",
    "pressure (24 h average)": "pression (moyenne 24 h)",
    "pressure (6 h peak)": "pression (pic sur 6 h)",
    "pressure change over 3 h": "variation de la pression sur 3 h",
    "pressure variability (24 h)": "variabilité de la pression (24 h)",
    "rotation speed (3 h average)": "vitesse de rotation (moyenne 3 h)",
    "rotation speed (24 h average)": "vitesse de rotation (moyenne 24 h)",
    "rotation speed change over 3 h": "variation de la rotation sur 3 h",
    "rotation variability (24 h)": "variabilité de la rotation (24 h)",
    "voltage (3 h average)": "tension (moyenne 3 h)",
    "voltage (24 h average)": "tension (moyenne 24 h)",
    "voltage change over 3 h": "variation de la tension sur 3 h",
    "voltage variability (24 h)": "variabilité de la tension (24 h)",
    "errors in the last 24 h": "erreurs des dernières 24 h",
    "errors in the last 72 h": "erreurs des dernières 72 h",
    "hours since last error": "temps écoulé depuis la dernière erreur",
    "hours since last maintenance": "temps écoulé depuis la dernière maintenance",
    "hours since last failure": "temps écoulé depuis la dernière panne",
    "machine age": "âge de la machine",
    "machine model": "modèle de machine",
}
MODELS_FR = {
    "baseline_never_fails": "Référence : « jamais de panne »",
    "logistic_regression": "Régression logistique",
    "random_forest": "Forêt aléatoire",
    "hist_gradient_boosting": "Gradient boosting (histogrammes)",
    "lightgbm": "LightGBM",
    "isolation_forest (unsupervised)": "Isolation Forest (non supervisé)",
}
TRIGGERS_FR = {"initial": "premier modèle", "drift": "dérive détectée", "manual": "manuel"}

st.set_page_config(page_title="Maintenance prédictive", page_icon=None, layout="wide")

st.markdown(
    f"""
<style>
  .block-container {{ padding-top: 2.2rem; max-width: 1180px; }}
  h1 {{ font-weight: 700; letter-spacing: -0.02em; color: {INK}; }}
  h2, h3 {{ color: {INK}; letter-spacing: -0.01em; }}
  .sub {{ color: {MUTED}; margin: -0.6rem 0 1.4rem 0; font-size: 0.95rem; }}
  .plan {{ border: 1px solid {LINE}; border-radius: 10px; padding: 0.4rem 1.1rem; background: #fff; }}
  .row {{ display: grid; grid-template-columns: 3.2rem 4.2rem 1fr 4.6rem 19rem; gap: 0.9rem; align-items: center;
         padding: 0.7rem 0; border-bottom: 1px solid {LINE}; }}
  .row:last-child {{ border-bottom: none; }}
  .rank {{ color: {MUTED}; font-variant-numeric: tabular-nums; }}
  .mid {{ font-weight: 700; color: {INK}; }}
  .track {{ background: {PAPER}; border-radius: 4px; height: 10px; }}
  .fill {{ height: 10px; border-radius: 4px; }}
  .pct {{ text-align: right; font-weight: 600; font-variant-numeric: tabular-nums; }}
  .why {{ color: {MUTED}; font-size: 0.88rem; }}
  .lvl {{ display: inline-block; border-radius: 999px; padding: 0.05rem 0.55rem; font-size: 0.78rem; font-weight: 600;
         color: #fff; }}
  .note {{ border-left: 3px solid {ACCENT}; background: {PAPER}; padding: 0.7rem 1rem; border-radius: 0 6px 6px 0;
          color: {INK}; font-size: 0.93rem; margin: 0.4rem 0 1rem 0; }}
  div[data-testid="stMetric"] {{ background: #fff; border: 1px solid {LINE}; border-radius: 10px; padding: 0.8rem 1rem; }}
</style>
""",
    unsafe_allow_html=True,
)


def fr_number(x: float, digits: int = 0) -> str:
    return f"{x:,.{digits}f}".replace(",", " ").replace(".", ",")


def eur(x: float) -> str:
    return f"{fr_number(x)} €"


def pct(x: float, digits: int = 1) -> str:
    return f"{fr_number(x * 100, digits)} %"


def factor_fr(text: str) -> str:
    return FACTORS_FR.get(text, text) if isinstance(text, str) and text else "n/d"


def day_fr(ts) -> str:
    return pd.Timestamp(ts).strftime("%d/%m/%Y")


def reason_fr(text: str) -> str:
    m = re.match(r"challenger lowers the total cost by ([\d.]+)% \(required: ([\d.]+)%\)", text)
    if m:
        return f"Le nouveau modèle réduit le coût total de {m.group(1).replace('.', ',')} % (minimum exigé : {m.group(2).replace('.', ',')} %)."
    m = re.match(r"challenger gain is ([-\d.]+)%, below the required ([\d.]+)%: champion kept", text)
    if m:
        return f"Gain du nouveau modèle : {m.group(1).replace('.', ',')} %, sous le minimum exigé de {m.group(2).replace('.', ',')} %. L'ancien modèle est conservé."
    if text.startswith("no champion yet"):
        return "Aucun modèle en place : le premier modèle est promu d'office."
    return text


@st.cache_data
def load():
    def csv(name, **kw):
        return pd.read_csv(DEMO / name, **kw)

    return {
        "overview": json.loads((DEMO / "overview.json").read_text()),
        "summary": json.loads((DEMO / "summary.json").read_text()),
        "decisions": json.loads((DEMO / "decisions.json").read_text()),
        "risk": csv("risk_table.csv", parse_dates=["timestamp"]),
        "history": csv("history.csv", parse_dates=["timestamp"]),
        "failures": csv("failures.csv", parse_dates=["timestamp"]),
        "drift": csv("drift.csv"),
        "timeline": csv("drift_timeline.csv", parse_dates=["day"]),
        "comparison": csv("comparison.csv"),
        "calibration": csv("calibration.csv"),
        "cost": csv("cost_curve.csv"),
    }


def lvl_badge(level: str) -> str:
    return f'<span class="lvl" style="background:{RISK_COLORS[level]}">{RISK_LABELS[level]}</span>'


if not (DEMO / "overview.json").exists():
    st.error("Aucun instantané trouvé. Lance `make demo` pour générer dashboard/demo_data.")
    st.stop()

d = load()
ov, summary = d["overview"], d["summary"]
PAGES = ["Vue d'ensemble", "Détail d'une machine", "Surveillance", "Modèle", "Coût"]

st.sidebar.title("Maintenance prédictive")
page = st.sidebar.radio("Page", PAGES, label_visibility="collapsed")
st.sidebar.caption(f"Modèle v{ov['model_version']} · {MODELS_FR.get(ov['model_name'], ov['model_name'])}")
st.sidebar.caption(f"Données jusqu'au {day_fr(ov['data']['last_timestamp'])}")
if ov.get("synthetic_data"):
    st.sidebar.warning("Données synthétiques : ces résultats ne disent rien sur de vraies machines.")

# ---------------------------------------------------------------------------------------------- vue d'ensemble
if page == PAGES[0]:
    st.title("Quelles machines inspecter aujourd'hui ?")
    st.markdown(
        f'<div class="sub">Probabilité de panne dans les 24 prochaines heures, au {day_fr(ov["data"]["last_timestamp"])}.</div>',
        unsafe_allow_html=True,
    )
    n_day = int(summary["final_test"]["n_per_day"])
    risk = d["risk"].copy()
    risk["probability"] = risk["failure_probability"]

    st.subheader(f"Plan d'inspection : les {n_day} machines à voir en priorité")
    top = risk.head(n_day)
    scale = max(top["probability"].max(), 0.01)
    rows = ""
    for i, r in enumerate(top.itertuples(), start=1):
        width = max(3, int(r.probability / scale * 100))
        rows += (
            f'<div class="row"><div class="rank">{i}</div><div class="mid">{r.machine_id}</div>'
            f'<div class="track"><div class="fill" style="width:{width}%;background:{RISK_COLORS[r.risk_level]}"></div></div>'
            f'<div class="pct">{pct(r.probability)}</div>'
            f'<div class="why">{lvl_badge(r.risk_level)}&nbsp; {factor_fr(r.top_factor)}</div></div>'
        )
    st.markdown(f'<div class="plan">{rows}</div>', unsafe_allow_html=True)
    st.markdown(
        f"<div class=\"note\">L'équipe ne peut inspecter que <b>{n_day} machines par jour</b> : ce qui compte, c'est donc le <b>classement</b>. "
        "« Facteur principal » = ce qui pèse le plus dans le score de la machine.</div>",
        unsafe_allow_html=True,
    )

    c = st.columns(4)
    c[0].metric("Machines suivies", ov["data"]["machines"])
    c[1].metric("Risque élevé", ov["high_risk"])
    c[2].metric("Risque moyen", ov["medium_risk"])
    drift_vars = ", ".join(SENSORS.get(v, v).lower() for v in ov["drifted_variables"])
    c[3].metric("Dérive des données", "Détectée" if ov["drift_detected"] else "Aucune", drift_vars or None, delta_color="off")

    st.subheader("Toutes les machines")
    table = risk[["machine_id", "risk_level", "probability", "top_factor"]].copy()
    table["risk_level"] = table["risk_level"].map(RISK_LABELS)
    table["top_factor"] = table["top_factor"].map(factor_fr)
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        height=330,
        column_config={
            "machine_id": st.column_config.TextColumn("Machine"),
            "risk_level": st.column_config.TextColumn("Niveau de risque"),
            "probability": st.column_config.ProgressColumn(
                "Probabilité de panne (24 h)", format="percent", min_value=0.0, max_value=float(scale)
            ),
            "top_factor": st.column_config.TextColumn("Facteur principal"),
        },
    )

# ---------------------------------------------------------------------------------------------- détail machine
elif page == PAGES[1]:
    st.title("Détail d'une machine")
    m = st.selectbox("Machine", d["risk"]["machine_id"].tolist())
    h = d["history"][d["history"]["machine_id"] == m]
    fails = d["failures"][d["failures"]["machine_id"] == m]
    row = d["risk"][d["risk"]["machine_id"] == m].iloc[0]
    c = st.columns(3)
    c[0].metric("Niveau de risque", RISK_LABELS[row["risk_level"]])
    c[1].metric("Probabilité de panne (24 h)", pct(row["failure_probability"]))
    c[2].markdown(
        f'<div style="border:1px solid {LINE};border-radius:10px;padding:0.8rem 1rem;background:#fff">'
        f'<div style="font-size:0.85rem;color:{MUTED}">Facteur principal</div>'
        f'<div style="font-size:1.25rem;font-weight:600;color:{INK};line-height:1.35;margin-top:0.2rem">{factor_fr(row["top_factor"])}</div></div>',
        unsafe_allow_html=True,
    )

    line = (
        alt.Chart(h)
        .mark_line(color=ACCENT, strokeWidth=2)
        .encode(
            x=alt.X("timestamp:T", title=None, axis=alt.Axis(format="%d/%m", tickCount="day")),
            y=alt.Y("failure_probability:Q", title="Probabilité de panne", axis=alt.Axis(format="%")),
            tooltip=[alt.Tooltip("timestamp:T", title="Date"), alt.Tooltip("failure_probability:Q", title="Probabilité", format=".1%")],
        )
    )
    marks = (
        alt.Chart(fails[fails["timestamp"] >= h["timestamp"].min()])
        .mark_rule(color=RISK_COLORS["HIGH"], strokeDash=[4, 3])
        .encode(x="timestamp:T", tooltip=[alt.Tooltip("timestamp:T", title="Panne")])
    )
    st.altair_chart(
        (line + marks).properties(height=230, title="Risque sur les 7 derniers jours (pointillé rouge = panne réelle)"),
        width="stretch",
    )
    cols = st.columns(2)
    for i, (key, label) in enumerate(SENSORS.items()):
        sensor_chart = (
            alt.Chart(h)
            .mark_line(strokeWidth=1.6, color=SENSOR_COLORS[label])
            .encode(
                x=alt.X("timestamp:T", title=None, axis=alt.Axis(format="%d/%m", tickCount="day")),
                y=alt.Y(f"{key}:Q", scale=alt.Scale(zero=False), title=None),
                tooltip=[alt.Tooltip("timestamp:T", title="Date"), alt.Tooltip(f"{key}:Q", title=label, format=".1f")],
            )
            .properties(height=150, title=label)
        )
        cols[i % 2].altair_chart(sensor_chart, width="stretch")
    if API_URL:
        with st.expander("Tester une mesure avec l'API en direct"):
            import requests

            vals = {SENSORS[k]: st.number_input(SENSORS[k], value=float(h[k].iloc[-1])) for k in SENSORS}
            if st.button("Calculer le score"):
                payload = {k: vals[SENSORS[k]] for k in SENSORS}
                try:
                    r = requests.post(f"{API_URL}/predict", json={"machine_id": m, **payload}, timeout=5)
                    st.json(r.json())
                except Exception as exc:
                    st.warning(f"API injoignable ({exc}). L'instantané ci-dessus n'en a pas besoin.")

# ---------------------------------------------------------------------------------------------- surveillance
elif page == PAGES[2]:
    st.title("Dérive des données et décisions sur le modèle")
    st.markdown(
        "<div class=\"note\">Le <b>PSI</b> mesure à quel point les mesures d'un capteur ont changé par rapport à la période d'entraînement. "
        "Au-dessus de <b>0,2</b>, le changement est jugé important et peut dégrader le modèle.</div>",
        unsafe_allow_html=True,
    )
    tl = d["timeline"].copy()
    tl["capteur"] = tl["variable"].map(SENSORS)
    base = (
        alt.Chart(tl)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("day:T", title=None, axis=alt.Axis(format="%d/%m")),
            y=alt.Y("psi:Q", title="PSI par rapport à l'entraînement"),
            color=alt.Color("capteur:N", title="Capteur", scale=alt.Scale(domain=list(SENSOR_COLORS), range=list(SENSOR_COLORS.values()))),
            tooltip=[alt.Tooltip("day:T", title="Jour"), "capteur", alt.Tooltip("psi:Q", title="PSI", format=".2f")],
        )
    )
    rule = alt.Chart(pd.DataFrame({"y": [0.2]})).mark_rule(color=RISK_COLORS["HIGH"], strokeDash=[5, 4]).encode(y="y:Q")
    st.altair_chart(
        (base + rule).properties(height=300, title="PSI quotidien par capteur (pointillé = seuil d'alerte 0,2)"), width="stretch"
    )

    drift = d["drift"].copy()
    drift["variable"] = drift["variable"].map(lambda v: SENSORS.get(v, v))
    drift["drift"] = drift["drift"].map({True: "Oui", False: "Non"})
    drift["ks_pvalue"] = drift["ks_pvalue"].map(lambda x: f"{x:.2g}".replace(".", ","))
    drift["psi"] = drift["psi"].map(lambda x: fr_number(x, 2))
    drift["ks_statistic"] = drift["ks_statistic"].map(lambda x: fr_number(x, 2))
    st.dataframe(
        drift.rename(
            columns={"variable": "Capteur", "psi": "PSI", "ks_statistic": "Statistique KS", "ks_pvalue": "p-value KS", "drift": "Dérive"}
        ),
        hide_index=True,
        width="stretch",
    )

    st.subheader("Décisions : modèle en place ou nouveau modèle")
    if d["decisions"]:
        for dec in d["decisions"]:
            verdict = "Promu" if dec["promoted"] else "Refusé"
            trigger = TRIGGERS_FR.get(dec.get("trigger", ""), dec.get("trigger", "n/d"))
            st.markdown(f"**{verdict}** · modèle v{dec['challenger_version']} · déclencheur : {trigger}")
            st.markdown(reason_fr(dec["reason"]))
            if dec.get("champion_cost"):
                st.caption(
                    f"Coût total sur la même fenêtre inédite : ancien modèle {eur(dec['champion_cost'])} contre nouveau modèle "
                    f"{eur(dec['challenger_cost'])} (coûts hypothétiques)."
                )
    else:
        st.info("Aucune décision pour l'instant.")

# ---------------------------------------------------------------------------------------------- modèle
elif page == PAGES[3]:
    st.title("Comparaison des modèles et calibration")
    st.markdown(
        '<div class="note">Découpage chronologique : entraînement, calibration, seuil, test. Les chiffres ci-dessous portent sur le bloc de test, '
        "jamais utilisé pour choisir ou régler un modèle. Le <b>PR-AUC</b> d'un modèle au hasard vaut le taux de pannes (ici environ 0,03).</div>",
        unsafe_allow_html=True,
    )
    comp = d["comparison"].copy()
    table = pd.DataFrame(
        {
            "Modèle": comp["model"].map(lambda m: MODELS_FR.get(m, m)),
            "PR-AUC": comp["pr_auc"].map(lambda x: fr_number(x, 3)),
            "ROC-AUC": comp["roc_auc"].map(lambda x: "–" if pd.isna(x) else fr_number(x, 3)),
            "Rappel": comp["recall"].map(pct),
            "Précision": comp["precision"].map(pct),
            "Précision du top N/jour": comp["precision_at_n"].map(lambda x: "–" if pd.isna(x) else pct(x)),
            "Coût total": comp["total_cost"].map(eur),
        }
    )
    st.dataframe(table, hide_index=True, width="stretch")
    cal = d["calibration"]
    top_axis = float(max(cal["predicted"].max(), cal["observed"].max()))
    diag = pd.DataFrame({"x": [0, top_axis], "y": [0, top_axis]})
    pts = (
        alt.Chart(cal)
        .mark_circle(size=80, color=ACCENT)
        .encode(
            x=alt.X("predicted:Q", title="Probabilité prédite", axis=alt.Axis(format="%")),
            y=alt.Y("observed:Q", title="Fréquence observée", axis=alt.Axis(format="%")),
            tooltip=[
                alt.Tooltip("predicted:Q", title="Prédite", format=".1%"),
                alt.Tooltip("observed:Q", title="Observée", format=".1%"),
                alt.Tooltip("n:Q", title="Observations"),
            ],
        )
    )
    st.altair_chart(
        (pts + alt.Chart(diag).mark_line(color=MUTED, strokeDash=[4, 3]).encode(x="x:Q", y="y:Q")).properties(
            height=320, title="Calibration (sur la diagonale = probabilités fiables)"
        ),
        width="stretch",
    )
    f = summary["final_test"]
    c = st.columns(4)
    c[0].metric("PR-AUC", fr_number(f["pr_auc"], 3), f"hasard : {fr_number(f['baseline_pr_auc'], 3)}", delta_color="off")
    c[1].metric("Pannes détectées (rappel)", pct(f["recall"]))
    c[2].metric("Alertes justifiées (précision)", pct(f["precision"]))
    c[3].metric(f"Précision du top {f['n_per_day']} par jour", pct(f["precision_at_n"]))

# ---------------------------------------------------------------------------------------------- coût
else:
    st.title("Seuil de décision en euros")
    st.warning("Les coûts sont des hypothèses (docs/cost_assumptions.md), pas des chiffres réels.")
    st.markdown(
        '<div class="note">Une panne ratée coûte beaucoup plus cher qu\'une inspection inutile. Le seuil retenu est celui qui <b>minimise le coût total</b>, '
        "pas celui qui maximise la précision.</div>",
        unsafe_allow_html=True,
    )
    cost = d["cost"].assign(cout_m=lambda x: x["cost"] / 1e6)
    thr = summary["threshold"]
    line = (
        alt.Chart(cost)
        .mark_line(color=ACCENT, strokeWidth=2)
        .encode(
            x=alt.X("threshold:Q", title="Seuil de probabilité", axis=alt.Axis(format="%")),
            y=alt.Y("cout_m:Q", title="Coût total (millions d'€)", scale=alt.Scale(zero=False)),
            tooltip=[alt.Tooltip("threshold:Q", title="Seuil", format=".1%"), alt.Tooltip("cout_m:Q", title="Coût (M€)", format=".2f")],
        )
    )
    pick = alt.Chart(pd.DataFrame({"x": [thr]})).mark_rule(color=RISK_COLORS["HIGH"]).encode(x="x:Q")
    st.altair_chart(
        (line + pick).properties(height=330, title="Coût total sur le bloc de réglage du seuil (rouge = seuil retenu)"),
        width="stretch",
    )
    f = summary["final_test"]
    c = st.columns(3)
    c[0].metric("Avec le modèle (bloc de test)", eur(f["total_cost"]))
    c[1].metric("Ne jamais inspecter", eur(f["cost_do_nothing"]))
    c[2].metric("Tout inspecter", eur(f["cost_inspect_all"]))
    costs = summary["costs"]
    st.caption(
        f"Hypothèses : panne ratée {eur(costs['missed_failure'])}, inspection inutile {eur(costs['useless_inspection'])}, "
        f"inspection utile {eur(costs['useful_inspection'])}."
    )
