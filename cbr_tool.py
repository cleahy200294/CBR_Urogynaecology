# Urogynaecology CBR Tool
# Finds the K most similar past patients to a new query using weighted KNN.
# Data lives in CBR_database_working.csv — new urodynamic cases can be appended via the UI.

import csv
import datetime
import math
import streamlit as st
from pathlib import Path

# Path to the working database (same folder as this script)
CSV_PATH = Path(__file__).parent / "CBR_database_working.csv"

# Column names used in the CSV — every row must have these fields
CSV_FIELDS = [
    "id", "hx", "age", "sui", "urgency", "frequency", "nocturia",
    "leaking", "pessary", "caffeine", "qmax", "leukocytes", "protein",
    "blood", "oe", "cystocele", "rectocele", "uvd", "pfc", "sensations",
    "diag_sui", "diag_det", "void", "diag_source", "date_added",
]

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Urogynaecology CBR Tool",
    page_icon=str(Path(__file__).parent / "logo_ucc.jpg"),
    layout="wide",
)

st.markdown("""
<style>
    .main { background-color: #f8f9fb; }
    .block-container { padding-top: 2rem; }
    h1 { color: #293A84; font-family: 'Georgia', serif; }
    h2, h3 { color: #293A84; }
    .result-card {
        background: white;
        border-left: 4px solid #2992C8;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 12px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    }
</style>
""", unsafe_allow_html=True)


# ── Load cases from CSV ────────────────────────────────────────────────────────
# Not cached so newly saved cases appear immediately without restarting.
# sf() safely converts a CSV cell to float; returns None if blank (skipped in distance calc).
def load_cases():
    if not CSV_PATH.exists():
        st.error("Database file not found: CBR_database_working.csv")
        st.stop()

    cases = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            def sf(key, default=0.0):
                v = row.get(key, "").strip()
                if v == "":
                    return None if default is None else default
                try:
                    return float(v)
                except:
                    return None if default is None else default

            cases.append({
                "id":          row.get("id", ""),
                "hx":          row.get("hx", ""),
                "age":         sf("age"),
                "sui":         sf("sui"),
                "urgency":     sf("urgency"),
                "frequency":   sf("frequency"),
                "nocturia":    sf("nocturia"),
                "leaking":     sf("leaking"),
                "pessary":     sf("pessary"),
                "caffeine":    sf("caffeine"),
                "qmax":        sf("qmax", None),    # None = not recorded, skipped in comparison
                "leukocytes":  sf("leukocytes"),
                "protein":     sf("protein"),
                "blood":       sf("blood"),
                "oe":          row.get("oe", ""),
                "cystocele":   sf("cystocele"),
                "rectocele":   sf("rectocele"),
                "uvd":         sf("uvd"),
                "pfc":         sf("pfc", None),     # None = not recorded, skipped in comparison
                "sensations":  row.get("sensations", ""),
                "diag_sui":    row.get("diag_sui", ""),
                "diag_det":    row.get("diag_det", ""),
                "void":        row.get("void", ""),
                "diag_source": row.get("diag_source", "original"),
                "date_added":  row.get("date_added", ""),
            })
    return cases


# ── Append a new case to the CSV ───────────────────────────────────────────────
def append_case_to_csv(new_case: dict):
    write_header = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(new_case)


# ── Feature table ──────────────────────────────────────────────────────────────
# (feature, max_value, SUI weight, Detrusor weight)
# Two separate weight columns allow the KNN to be tuned independently for
# SUI diagnosis vs detrusor diagnosis — run twice, shown in separate tabs.
# SUI weights favour anatomical features; detrusor weights favour OAB symptoms.
FEATURES = [
    # feature       max   SUI   DET
    ("age",         100,  1.0,  1.0),
    ("sui",           1,  2.5,  0.5),   # strong SUI predictor
    ("urgency",       1,  0.5,  2.5),   # strong detrusor predictor
    ("frequency",     1,  0.5,  2.0),
    ("nocturia",     10,  0.5,  1.5),
    ("leaking",       1,  1.0,  2.0),
    ("pessary",       1,  0.5,  0.5),
    ("caffeine",      5,  0.3,  1.5),   # bladder irritant — more detrusor relevant
    ("qmax",          2,  1.5,  1.0),
    ("leukocytes",    4,  0.5,  1.0),
    ("protein",       4,  0.5,  1.0),
    ("blood",         4,  0.5,  1.0),
    ("cystocele",     4,  2.0,  0.5),   # anatomical — strong SUI association
    ("rectocele",     4,  0.5,  0.5),
    ("uvd",           3,  2.0,  0.5),   # urethro-vesical descent — key SUI marker
    ("pfc",           5,  1.5,  1.0),
]


# ── Convert a case dict to a normalised number list ────────────────────────────
def vectorise(case):
    return [
        (case[key] / max_val if case[key] is not None else None)
        for key, max_val, *_ in FEATURES
    ]


# ── Weighted Euclidean distance between two patient vectors ────────────────────
# weight_col: 2 = SUI weights, 3 = detrusor weights
def distance(query_vec, case_vec, weight_col):
    sq, dims = 0.0, 0
    for i, feat in enumerate(FEATURES):
        w = feat[weight_col]
        q, c = query_vec[i], case_vec[i]
        if q is None or c is None:
            continue
        sq += w * (q - c) ** 2
        dims += 1
    return math.sqrt(sq / dims) if dims else float("inf")


# ── Find K nearest neighbours ──────────────────────────────────────────────────
# weight_col: 2 = SUI weights, 3 = detrusor weights
def find_knn(query, cases, k, weight_col):
    q_vec = vectorise(query)
    scored = [(distance(q_vec, vectorise(case), weight_col), case) for case in cases]
    scored.sort(key=lambda x: x[0])
    return scored[:k]


# ── Page header ────────────────────────────────────────────────────────────────
logo_path = Path(__file__).parent / "logo_ucc.jpg"
if logo_path.exists():
    st.image(str(logo_path), width=200)
st.title("🏥 Urogynaecology CBR Tool")
st.markdown("Enter patient details below to find the most similar cases from the database.")

cases = load_cases()
original_count = sum(1 for c in cases if c["diag_source"] == "original")
added_count    = len(cases) - original_count
caption = f"Database: **{original_count} original cases**"
if added_count:
    caption += f" + **{added_count} added** = **{len(cases)} total**"
st.caption(caption)

st.divider()


# ── Input form ─────────────────────────────────────────────────────────────────
# Grouped in a form so the search only runs on submit, not on every widget change.
with st.form("cbr_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Patient & Symptoms")
        age       = st.number_input("Age", min_value=10, max_value=100, value=55, step=1)
        st.markdown("**Symptoms** (tick all that apply)")
        sui       = st.checkbox("Hx of SUI (leaks with cough / sneeze / exercise)")
        urgency   = st.checkbox("Urgency")
        frequency = st.checkbox("Frequency")
        nocturia  = st.number_input("Nocturia (episodes per night)", min_value=0, max_value=10, value=0, step=1)
        leaking   = st.checkbox("Unprovoked leaking / urge incontinence")
        pessary   = st.checkbox("Pessary in use")
        caffeine  = st.number_input("Caffeinated drinks per day", min_value=0, max_value=10, value=0, step=1)

    with col2:
        st.subheader("Uroflow & Urine")
        qmax     = st.selectbox("Qmax", options=["Normal", "Low", "High"], index=0)
        qmax_val = {"Normal": 1.0, "Low": 0.0, "High": 2.0}[qmax]  # text → number
        st.markdown("**Urine Dipstick**")
        leukocytes = st.select_slider("Leukocytes", options=[0, 1, 2, 3, 4], value=0)
        protein    = st.select_slider("Protein",    options=[0, 1, 2, 3, 4], value=0)
        blood      = st.select_slider("Blood",      options=[0, 1, 2, 3, 4], value=0)

    with col3:
        st.subheader("Prolapse Assessment")
        cystocele = st.select_slider("Cystocele grade", options=[0, 1, 2, 3, 4], value=0)
        rectocele = st.select_slider("Rectocele grade", options=[0, 1, 2, 3, 4], value=0)
        uvd       = st.select_slider("UV Descent (UVD)", options=[0, 1, 2, 3], value=0)
        pfc       = st.select_slider("Pelvic Floor Contraction (PFC, Oxford)", options=[0, 1, 2, 3, 4, 5], value=3)

    st.divider()
    k         = st.slider("Number of nearest cases (K)", min_value=1, max_value=15, value=5)
    submitted = st.form_submit_button("🔍 Find Similar Cases", use_container_width=True, type="primary")


# ── Results ────────────────────────────────────────────────────────────────────
if submitted:
    # Build query dict — checkboxes become 1.0/0.0 to match database format
    query = {
        "age":        float(age),
        "sui":        1.0 if sui else 0.0,
        "urgency":    1.0 if urgency else 0.0,
        "frequency":  1.0 if frequency else 0.0,
        "nocturia":   float(nocturia),
        "leaking":    1.0 if leaking else 0.0,
        "pessary":    1.0 if pessary else 0.0,
        "caffeine":   float(caffeine),
        "qmax":       qmax_val,
        "leukocytes": float(leukocytes),
        "protein":    float(protein),
        "blood":      float(blood),
        "cystocele":  float(cystocele),
        "rectocele":  float(rectocele),
        "uvd":        float(uvd),
        "pfc":        float(pfc),
    }

    # Store in session_state so the "Add case" section below can access it after rerender
    st.session_state["last_query"]      = query
    st.session_state["last_qmax_label"] = qmax
    st.session_state["has_results"]     = True

    # Run KNN twice — once per diagnosis type, each with its own weights
    results_sui = find_knn(query, cases, k, weight_col=2)
    results_det = find_knn(query, cases, k, weight_col=3)
    st.session_state["last_results"] = results_sui

    st.subheader(f"Top {k} Most Similar Cases")

    def render_results(results):
        for rank, (dist, case) in enumerate(results, 1):
            similarity = max(0, round((1 - dist) * 100, 1))
            src_tag = " *(added)*" if case["diag_source"] != "original" else ""

            with st.expander(
                f"#{rank}  •  Case {case['id']}{src_tag}  •  Age {int(case['age'])}  •  Similarity {similarity}%",
                expanded=(rank <= 3),
            ):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Urodynamic Diagnoses**")
                    if case["diag_sui"]:
                        st.info(f"**SUI:** {case['diag_sui']}")
                    if case["diag_det"]:
                        st.info(f"**Detrusor:** {case['diag_det']}")
                    if case["void"]:
                        st.info(f"**Voiding:** {case['void']}")
                    if case["diag_source"] != "original":
                        st.caption(f"Source: Urodynamics  ·  Added {case['date_added']}")

                with c2:
                    st.markdown("**Case Profile**")
                    flags = []
                    if case["sui"]:       flags.append("Hx SUI")
                    if case["urgency"]:   flags.append("Urgency")
                    if case["frequency"]: flags.append("Frequency")
                    if case["nocturia"]:  flags.append(f"Nocturia ×{int(case['nocturia'])}")
                    if case["leaking"]:   flags.append("Leaking")
                    if case["pessary"]:   flags.append("Pessary")
                    st.write("Symptoms: " + (", ".join(flags) if flags else "None recorded"))
                    qmax_str = {0.0: "Low", 1.0: "Normal", 2.0: "High"}.get(case["qmax"], "—")
                    st.write(f"Qmax: **{qmax_str}**  |  "
                             f"Cystocele: **{int(case['cystocele'])}**  |  "
                             f"Rectocele: **{int(case['rectocele'])}**  |  "
                             f"UVD: **{int(case['uvd'])}**")

                if case["hx"]:
                    st.markdown("**Clinical history**")
                    st.caption(case["hx"])

                st.markdown(f"<small>Distance score: {dist:.4f}</small>", unsafe_allow_html=True)

    tab_sui, tab_det = st.tabs(["🔵 Urodynamic SUI Diagnosis", "🟠 Detrusor Diagnosis"])
    with tab_sui:
        st.caption("Ranked using SUI-optimised weights — emphasises hx SUI, cystocele, UVD.")
        render_results(results_sui)
    with tab_det:
        st.caption("Ranked using detrusor-optimised weights — emphasises urgency, frequency, nocturia, leaking.")
        render_results(results_det)


# ── Add case to dataset ────────────────────────────────────────────────────────
# Only real urodynamic results can be saved — not CBR-inferred diagnoses.
# Input features are taken from the last search; diagnoses are typed in here.
if st.session_state.get("has_results"):
    st.divider()
    with st.expander("➕ Add this case to the dataset (urodynamics only)", expanded=False):
        st.markdown(
            "Only cases with confirmed urodynamic results can be added to the dataset. "
            "Enter the results from the urodynamics study below."
        )

        m1, m2 = st.columns(2)
        with m1:
            case_id = st.text_input("Case ID", value=f"ADD-{datetime.date.today().isoformat()}")
        with m2:
            hx = st.text_input("Clinical notes / history (optional)")

        st.markdown("---")
        st.caption("Enter the confirmed results from the urodynamics study.")
        d1, d2, d3 = st.columns(3)
        with d1:
            diag_sui = st.text_input("SUI diagnosis", placeholder="e.g. Urodynamic SUI, No SUI", key="uro_sui")
        with d2:
            diag_det = st.text_input("Detrusor diagnosis", placeholder="e.g. Overactive detrusor, Stable", key="uro_det")
        with d3:
            void_dx  = st.text_input("Voiding diagnosis", placeholder="e.g. Normal void, Incomplete emptying", key="uro_void")

        st.markdown("")
        if st.button("💾 Save case to dataset", type="primary", key="save_case_btn"):
            q = st.session_state.get("last_query", {})
            new_case = {
                "id":          case_id.strip() or f"ADD-{datetime.date.today().isoformat()}",
                "hx":          hx.strip(),
                "age":         q.get("age", ""),
                "sui":         q.get("sui", ""),
                "urgency":     q.get("urgency", ""),
                "frequency":   q.get("frequency", ""),
                "nocturia":    q.get("nocturia", ""),
                "leaking":     q.get("leaking", ""),
                "pessary":     q.get("pessary", ""),
                "caffeine":    q.get("caffeine", ""),
                "qmax":        q.get("qmax", ""),
                "leukocytes":  q.get("leukocytes", ""),
                "protein":     q.get("protein", ""),
                "blood":       q.get("blood", ""),
                "oe":          "",
                "cystocele":   q.get("cystocele", ""),
                "rectocele":   q.get("rectocele", ""),
                "uvd":         q.get("uvd", ""),
                "pfc":         q.get("pfc", ""),
                "sensations":  "",
                "diag_sui":    diag_sui,
                "diag_det":    diag_det,
                "void":        void_dx,
                "diag_source": "urodynamics",
                "date_added":  datetime.date.today().isoformat(),
            }
            append_case_to_csv(new_case)
            st.success(f"✅ Case **{new_case['id']}** saved (Urodynamics). It will appear in future searches.")
            st.rerun()
