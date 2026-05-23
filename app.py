"""
Model Fuzzy de Decizie - Decizia de Pret pentru Produse Noi
Metoda mediei aritmetice | Reguli: R1 & R3 & R4

Rulare: streamlit run app.py
Dependinte: pip install streamlit matplotlib numpy pandas
"""

import io

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd

# ── Configurare pagina ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Model Fuzzy – Decizia de Pret",
    layout="wide",
)

st.markdown("""
<style>
    .block-container {
        padding-top: 1.2rem;
        padding-left: 4rem;
        padding-right: 4rem;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# FUNCTII MATEMATICE
# ══════════════════════════════════════════════════════════════════════════════

def trimf(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Functie de apartenenta triunghiulara."""
    mu = np.zeros_like(x, dtype=float)
    if b > a:
        mask = (x >= a) & (x <= b)
        mu[mask] = (x[mask] - a) / (b - a)
    mu[x == b] = 1.0
    if c > b:
        mask = (x > b) & (x <= c)
        mu[mask] = (c - x[mask]) / (c - b)
    return np.clip(mu, 0.0, 1.0)


def compute_decision(mu_list: list) -> np.ndarray:
    """Decizia fuzzy prin media aritmetica."""
    return np.mean(np.stack(mu_list, axis=0), axis=0)


def optimal_price(x: np.ndarray, mu_d: np.ndarray) -> tuple:
    idx = np.argmax(mu_d)
    return float(x[idx]), float(mu_d[idx])


# ══════════════════════════════════════════════════════════════════════════════
# TITLU
# ══════════════════════════════════════════════════════════════════════════════

st.title("Model Fuzzy de Decizie — Decizia de Pret pentru Produse Noi")
st.markdown(
    "**Metoda:** Media aritmetica &nbsp;|&nbsp; "
    "**Reguli active:** R1, R3, R4 &nbsp;|&nbsp; "
    "**Agregare:** D(x) = [μ₁(x) + μ₃(x) + μ₄(x)] / 3"
)
st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

def _clamp_u_max():
    if st.session_state.u_max <= st.session_state.u_min:
        st.session_state.u_max = round(st.session_state.u_min + 0.5, 1)


with st.sidebar:
    st.header("Parametrii problemei")

    st.subheader("Multimea alternativelor")
    c1, c2 = st.columns(2)
    u_min = c1.number_input("Pret min", value=12.0, step=1.0, format="%.1f", key="u_min", on_change=_clamp_u_max)
    u_max = c2.number_input("Pret max", value=40.0, step=1.0, format="%.1f", key="u_max")

    span = u_max - u_min

    st.subheader("Date economice")
    prod_cost = st.number_input("Cost de productie (c)", value=12.0, step=0.5, format="%.2f")
    comp_price = st.number_input("Pretul concurentei (p_c)", value=22.0, step=0.5, format="%.2f")

    double_cost = 2.0 * prod_cost
    st.caption(f"Dublul costului de productie: **{double_cost:.2f}**")

    st.markdown("---")
    st.subheader("Functii de apartenenta")
    st.caption(
        "Fiecare regula se defineste ca numar fuzzy triunghiular (a, b, c), "
        "unde b este varful (gradul 1). "
        "Valorile implicite se recalculeaza automat cand se modifica datele economice."
    )

    half_w = span * 0.22  # latimea implicita a triunghiurilor R3 si R4

    # ── R1: Pret scazut ───────────────────────────────────────────────────────
    # R1 nu depinde de parametrii economici, key simplu
    with st.expander("R1 — Pret scazut", expanded=True):
        st.caption("Varf la pretul minim, descrescand spre dreapta.")
        r1_a = st.number_input("a1 (stanga)", value=u_min, key=f"r1a_{u_min:.4f}_{u_max:.4f}", format="%.2f")
        r1_b = st.number_input("b1 (varf)", value=u_min, key=f"r1b_{u_min:.4f}_{u_max:.4f}", format="%.2f")
        r1_c = st.number_input("c1 (dreapta)", value=u_min + span / 2, key=f"r1c_{u_min:.4f}_{u_max:.4f}", format="%.2f")

    # ── R3: Pret ~ 2×cost ────────────────────────────────────────────────────
    # Key dinamic bazat pe double_cost → se reinitializeaza la schimbarea costului
    with st.expander(f"R3 — Pret aproape de 2 x cost ({double_cost:.2f})", expanded=True):
        st.caption("Triunghi simetric centrat pe dublul costului de productie.")
        r3_a = st.number_input("a3 (stanga)", value=double_cost - half_w, key=f"r3a_{double_cost:.4f}_{span:.4f}", format="%.2f")
        r3_b = st.number_input("b3 (varf)", value=double_cost, key=f"r3b_{double_cost:.4f}_{span:.4f}", format="%.2f")
        r3_c = st.number_input("c3 (dreapta)", value=double_cost + half_w, key=f"r3c_{double_cost:.4f}_{span:.4f}", format="%.2f")

    # ── R4: Pret ~ concurenta ─────────────────────────────────────────────────
    # Key dinamic bazat pe comp_price → se reinitializeaza la schimbarea concurentei
    with st.expander(f"R4 — Pret aproape de concurenta ({comp_price:.2f})", expanded=True):
        st.caption("Triunghi simetric centrat pe pretul concurentei.")
        r4_a = st.number_input("a4 (stanga)", value=comp_price - half_w, key=f"r4a_{comp_price:.4f}_{span:.4f}", format="%.2f")
        r4_b = st.number_input("b4 (varf)", value=comp_price, key=f"r4b_{comp_price:.4f}_{span:.4f}", format="%.2f")
        r4_c = st.number_input("c4 (dreapta)", value=comp_price + half_w, key=f"r4c_{comp_price:.4f}_{span:.4f}", format="%.2f")

    st.markdown("---")
    resolution = st.slider("Rezolutie calcul (puncte)", 200, 2000, 800, 100)


# ══════════════════════════════════════════════════════════════════════════════
# CALCULE
# ══════════════════════════════════════════════════════════════════════════════

x = np.linspace(u_min, u_max, resolution)

mu_r1 = trimf(x, r1_a, r1_b, r1_c)
mu_r3 = trimf(x, r3_a, r3_b, r3_c)
mu_r4 = trimf(x, r4_a, r4_b, r4_c)

mu_d = compute_decision([mu_r1, mu_r3, mu_r4])
x_opt, mu_opt = optimal_price(x, mu_d)


# ══════════════════════════════════════════════════════════════════════════════
# METRICI
# ══════════════════════════════════════════════════════════════════════════════

col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Pret optim recomandat", f"{x_opt:.2f} u.m.")
col_b.metric("Grad de decizie D(x*)", f"{mu_opt:.4f}")
col_c.metric("2 x Cost productie", f"{double_cost:.2f} u.m.")
col_d.metric("Pretul concurentei", f"{comp_price:.2f} u.m.")

st.markdown("")


# ══════════════════════════════════════════════════════════════════════════════
# GRAFICE
# ══════════════════════════════════════════════════════════════════════════════

COLORS = {
    "r1": "#1565c0",
    "r3": "#2e7d32",
    "r4": "#c62828",
    "d": "#6a1b9a",
    "opt": "#f57c00",
}

fig = plt.figure(figsize=(11, 5.5))
gs = gridspec.GridSpec(2, 1, hspace=0.45)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

# Grafic 1 — functiile individuale
ax1.fill_between(x, mu_r1, alpha=0.10, color=COLORS["r1"])
ax1.fill_between(x, mu_r3, alpha=0.10, color=COLORS["r3"])
ax1.fill_between(x, mu_r4, alpha=0.10, color=COLORS["r4"])
ax1.plot(x, mu_r1, color=COLORS["r1"], lw=2.2, label="R1 — Pret scazut")
ax1.plot(x, mu_r3, color=COLORS["r3"], lw=2.2, label=f"R3 — Pret ≈ 2×cost ({double_cost:.1f})")
ax1.plot(x, mu_r4, color=COLORS["r4"], lw=2.2, label=f"R4 — Pret ≈ concurenta ({comp_price:.1f})")
ax1.axvline(x_opt, color=COLORS["opt"], lw=1.4, ls="--", alpha=0.7)
ax1.set_xlim(u_min, u_max)
ax1.set_ylim(-0.04, 1.12)
ax1.set_xlabel("Pret (u.m.)", fontsize=10)
ax1.set_ylabel("Grad de apartenenta  μ(x)", fontsize=10)
ax1.set_title("Functiile de apartenenta ale regulilor fuzzy", fontsize=11, fontweight="bold")
ax1.legend(fontsize=9, loc="upper right")
ax1.grid(True, alpha=0.25, linestyle=":")
ax1.spines[["top", "right"]].set_visible(False)

# Grafic 2 — functia de decizie
ax2.fill_between(x, mu_d, alpha=0.15, color=COLORS["d"])
ax2.plot(x, mu_d, color=COLORS["d"], lw=2.5,
         label="D(x) = [μ₁ + μ₃ + μ₄] / 3")
ax2.axvline(x_opt, color=COLORS["opt"], lw=2, ls="--",
            label=f"Pret optim: {x_opt:.2f} u.m.  (D = {mu_opt:.4f})")
ax2.scatter([x_opt], [mu_opt], color=COLORS["opt"], zorder=6, s=90, ec="white", lw=1.5)
ax2.annotate(
    f"  x* = {x_opt:.2f}\n  D(x*) = {mu_opt:.4f}",
    xy=(x_opt, mu_opt),
    xytext=(x_opt + span * 0.05, mu_opt - 0.12),
    fontsize=9, color=COLORS["opt"],
    arrowprops=dict(arrowstyle="->", color=COLORS["opt"], lw=1.2),
)
ax2.set_xlim(u_min, u_max)
ax2.set_ylim(-0.04, 1.12)
ax2.set_xlabel("Pret (u.m.)", fontsize=10)
ax2.set_ylabel("Grad de decizie  D(x)", fontsize=10)
ax2.set_title("Functia de decizie fuzzy — metoda mediei aritmetice", fontsize=11, fontweight="bold")
ax2.legend(fontsize=9, loc="upper right")
ax2.grid(True, alpha=0.25, linestyle=":")
ax2.spines[["top", "right"]].set_visible(False)

# Salveaza figura in buffer pentru export inainte de afisare
png_buf = io.BytesIO()
fig.savefig(png_buf, format="png", dpi=150, bbox_inches="tight")
png_buf.seek(0)

st.pyplot(fig, use_container_width=True)
plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# EXPORT
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("#### Exporta rezultate")

df_full = pd.DataFrame({
    "Pret x": np.round(x, 4),
    "mu_R1(x)": np.round(mu_r1, 6),
    "mu_R3(x)": np.round(mu_r3, 6),
    "mu_R4(x)": np.round(mu_r4, 6),
    "D(x) = medie": np.round(mu_d, 6),
})
csv_buf = io.StringIO()
df_full.to_csv(csv_buf, index=False)

ex1, ex2 = st.columns(2)
ex1.download_button(
    label="⬇ Descarca date complete (CSV)",
    data=csv_buf.getvalue(),
    file_name="fuzzy_pret_date.csv",
    mime="text/csv",
    use_container_width=True,
)
ex2.download_button(
    label="⬇ Descarca grafice (PNG)",
    data=png_buf,
    file_name="fuzzy_pret_grafice.png",
    mime="image/png",
    use_container_width=True,
)

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# TABEL CU VALORI
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("#### Tabel valori calculate (esantion)")

idx_sample = np.linspace(0, len(x) - 1, 15, dtype=int)
df_sample = pd.DataFrame({
    "Pret x": np.round(x[idx_sample], 3),
    "mu_R1(x)": np.round(mu_r1[idx_sample], 4),
    "mu_R3(x)": np.round(mu_r3[idx_sample], 4),
    "mu_R4(x)": np.round(mu_r4[idx_sample], 4),
    "D(x) = medie": np.round(mu_d[idx_sample], 4),
})
st.dataframe(df_sample, use_container_width=True, hide_index=True)

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# DETALII METODOLOGICE
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("#### Detalii metodologice")

st.markdown(f"""
**Model fuzzy de decizie — metoda mediei aritmetice**

Fie X = [{u_min:.1f}, {u_max:.1f}] multimea alternativelor (intervalul univers al preturilor).

Cele trei reguli definesc multimi fuzzy pe X:

- **R1** — *"Produsul ar trebui sa aiba un pret scazut"*
  parametri: ({r1_a:.2f}, {r1_b:.2f}, {r1_c:.2f})

- **R3** — *"Produsul ar trebui sa aiba un pret apropiat de dublul costului de productie"*
  2 x cost = {double_cost:.2f} → parametri: ({r3_a:.2f}, {r3_b:.2f}, {r3_c:.2f})

- **R4** — *"Produsul ar trebui sa aiba un pret apropiat de pretul concurentei"*
  concurenta = {comp_price:.2f} → parametri: ({r4_a:.2f}, {r4_b:.2f}, {r4_c:.2f})

**Functia de decizie (media aritmetica):**
""")

st.latex(r"D(x) = \frac{\mu_{R1}(x) + \mu_{R3}(x) + \mu_{R4}(x)}{3}, \quad x \in X")

st.markdown("**Pretul optim:**")

st.latex(rf"x^* = \arg\max_{{x \in X}} D(x) = {x_opt:.2f} \text{{ u.m.}}, \quad D(x^*) = {mu_opt:.4f}")

st.markdown("**Functia de apartenenta triunghiulara** cu parametrii (a, b, c):")

st.latex(r"""
\mu(x) = \begin{cases}
0 & x < a \\
\dfrac{x - a}{b - a} & a \leq x \leq b \\
\dfrac{c - x}{c - b} & b < x \leq c \\
0 & x > c
\end{cases}
""")

st.markdown("---")
st.caption(
    "Schiller Vlad ; Parvan Eduard | Modelarea si Optimizarea Deciziei Economice"
)
