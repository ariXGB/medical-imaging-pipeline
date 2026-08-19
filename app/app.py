import streamlit as st

from PIL import Image

import os
import io
import base64
from dotenv import load_dotenv

import requests

load_dotenv()

FAST_API_URL = os.getenv("FAST_API_URL")

st.set_page_config(
    page_title="Chest X-Ray AI Diagnoser",
    page_icon="🩻",
    layout="centered"
)

# ─────────────────────────────────────────────────────────────────
# DESIGN SYSTEM
# Subject: a diagnostic-imaging workstation reading a live chest X-ray.
# Palette: near-black instrument housing (#05080D), panel (#0D131B),
#          scanner-teal signal (#33E6C9), alert red (#FF5C5C).
# Type:    Space Grotesk (display) + JetBrains Mono (data / readouts),
#          the pairing a radiology console actually uses.
# Signature: the scan-frame viewport — corner reticles + an animated
#            sweep line that plays while the pipeline is running.
# ─────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

:root {
    --bg: #05080D;
    --panel: #0D131B;
    --panel-2: #10192480;
    --border: #1E2A38;
    --border-bright: #2C3F52;
    --text: #E7EDF3;
    --text-dim: #6E7F91;
    --accent: #33E6C9;
    --accent-dim: #1B5C52;
    --danger: #FF5C5C;
    --danger-dim: #5C2323;
}

html, body, [class*="css"] { font-family: 'JetBrains Mono', monospace; }

.stApp {
    background:
        repeating-linear-gradient(180deg, rgba(255,255,255,0.012) 0px, rgba(255,255,255,0.012) 1px, transparent 1px, transparent 3px),
        radial-gradient(circle at 50% 0%, #0A1017 0%, var(--bg) 55%);
    color: var(--text);
}

#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
.block-container { padding-top: 2.5rem; max-width: 720px; }

/* ---------- Header ---------- */
.dx-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.22em;
    color: var(--accent);
    text-transform: uppercase;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.6rem;
}
.dx-eyebrow .dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--accent);
    box-shadow: 0 0 8px 2px var(--accent);
    animation: pulse 2.2s ease-in-out infinite;
}
.dx-title {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 2.15rem;
    letter-spacing: -0.02em;
    line-height: 1.1;
    margin: 0 0 0.9rem 0;
    color: var(--text);
}
.dx-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    line-height: 1.65;
    color: var(--text-dim);
    border-left: 2px solid var(--border-bright);
    padding-left: 0.9rem;
    margin-bottom: 2.2rem;
}
.dx-sub b { color: var(--text); font-weight: 500; }

.dx-rule {
    height: 1px;
    background: linear-gradient(90deg, var(--border-bright), transparent);
    margin: 1.6rem 0;
}

/* ---------- File uploader ---------- */
[data-testid="stFileUploaderDropzone"], section[data-testid="stFileUploadDropzone"] {
    background: var(--panel) !important;
    border: 1px dashed var(--border-bright) !important;
    border-radius: 2px !important;
}
[data-testid="stFileUploader"] section { background: transparent !important; }
[data-testid="stFileUploaderDropzoneInstructions"] svg { display: none; }

/* ---------- Buttons ---------- */
.stButton > button {
    width: 100%;
    background: transparent;
    color: var(--accent);
    border: 1px solid var(--accent-dim);
    border-radius: 2px;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    font-size: 0.78rem;
    padding: 0.65rem 0;
    transition: all 0.15s ease;
}
.stButton > button:hover {
    background: var(--accent);
    color: #04100D;
    border-color: var(--accent);
    box-shadow: 0 0 22px -4px var(--accent);
}
.stButton > button:active { transform: scale(0.99); }

/* ---------- Scan viewport (signature element) ---------- */
.scan-frame {
    position: relative;
    background: var(--panel);
    border: 1px solid var(--border);
    padding: 18px;
    margin: 1.4rem 0 1.6rem 0;
    overflow: hidden;
}
.corner {
    position: absolute;
    width: 16px; height: 16px;
    border-color: var(--accent);
    opacity: 0.9;
    z-index: 3;
}
.corner.tl { top: 8px; left: 8px; border-top: 2px solid var(--accent); border-left: 2px solid var(--accent); }
.corner.tr { top: 8px; right: 8px; border-top: 2px solid var(--accent); border-right: 2px solid var(--accent); }
.corner.bl { bottom: 8px; left: 8px; border-bottom: 2px solid var(--accent); border-left: 2px solid var(--accent); }
.corner.br { bottom: 8px; right: 8px; border-bottom: 2px solid var(--accent); border-right: 2px solid var(--accent); }

.scan-img {
    display: block;
    width: 100%;
    max-height: 480px;
    object-fit: contain;
    filter: grayscale(15%) contrast(1.04);
    background: #000;
}

.scan-sweep {
    position: absolute;
    left: 0; right: 0;
    height: 90px;
    top: -90px;
    background: linear-gradient(180deg, transparent, rgba(51,230,201,0.22) 45%, rgba(51,230,201,0.55) 50%, rgba(51,230,201,0.22) 55%, transparent);
    animation: sweep 1.9s linear infinite;
    z-index: 2;
    pointer-events: none;
}
@keyframes sweep {
    0% { top: -90px; }
    100% { top: 100%; }
}

.status-badge {
    position: absolute;
    top: 10px; left: 50%;
    transform: translateX(-50%);
    z-index: 4;
    display: flex; align-items: center; gap: 0.4rem;
    background: rgba(5,8,13,0.82);
    border: 1px solid var(--border-bright);
    padding: 0.28rem 0.7rem;
    font-size: 0.68rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--text-dim);
}
.status-badge .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--text-dim); }
.status-badge.idle .dot { background: var(--text-dim); }
.status-badge.active .dot { background: var(--accent); box-shadow: 0 0 6px 2px var(--accent); animation: pulse 1s ease-in-out infinite; }
.status-badge.active { color: var(--accent); border-color: var(--accent-dim); }
.status-badge.success .dot { background: var(--accent); box-shadow: 0 0 6px 2px var(--accent); }
.status-badge.success { color: var(--accent); border-color: var(--accent-dim); }
.status-badge.danger .dot { background: var(--danger); box-shadow: 0 0 6px 2px var(--danger); }
.status-badge.danger { color: var(--danger); border-color: var(--danger-dim); }

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.35; }
}

/* ---------- Verdict stamp ---------- */
.dx-stamp {
    display: inline-block;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 1.15rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    padding: 0.4rem 1rem;
    border: 2px solid var(--danger);
    color: var(--danger);
    transform: rotate(-1.2deg);
    margin-bottom: 1rem;
}
.dx-stamp.ok { border-color: var(--accent); color: var(--accent); transform: rotate(0.8deg); }

/* ---------- Readout grid ---------- */
.dx-readouts { display: flex; gap: 1px; background: var(--border); margin: 1rem 0; border: 1px solid var(--border); }
.dx-readout { flex: 1; background: var(--panel); padding: 0.9rem 1rem; }
.dx-readout .label {
    font-size: 0.68rem; letter-spacing: 0.14em; text-transform: uppercase;
    color: var(--text-dim); margin-bottom: 0.35rem;
}
.dx-readout .value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.55rem; font-weight: 600; color: var(--text);
}
.dx-readout.hl .value { color: var(--accent); }
.dx-readout.hl-danger .value { color: var(--danger); }

/* ---------- Message line ---------- */
.dx-msg {
    font-size: 0.82rem; color: var(--text-dim);
    border-left: 2px solid var(--danger-dim);
    padding-left: 0.8rem; margin: 0.6rem 0 1.2rem 0; line-height: 1.6;
}

/* ---------- Section label ---------- */
.dx-section-label {
    font-size: 0.72rem; letter-spacing: 0.18em; text-transform: uppercase;
    color: var(--text-dim); margin: 1.6rem 0 0.8rem 0;
    display: flex; align-items: center; gap: 0.6rem;
}
.dx-section-label::after { content: ""; flex: 1; height: 1px; background: var(--border-bright); }

/* ---------- Probability bars ---------- */
.dx-prob-row { margin-bottom: 0.65rem; }
.dx-prob-row .top {
    display: flex; justify-content: space-between;
    font-size: 0.78rem; margin-bottom: 0.28rem;
}
.dx-prob-row .cls { color: var(--text); }
.dx-prob-row.top1 .cls { color: var(--accent); font-weight: 600; }
.dx-prob-row .pct { color: var(--text-dim); }
.dx-prob-row.top1 .pct { color: var(--accent); }
.dx-bar-track { height: 6px; background: var(--panel-2); border: 1px solid var(--border); position: relative; }
.dx-bar-fill { height: 100%; background: var(--border-bright); }
.dx-prob-row.top1 .dx-bar-fill { background: linear-gradient(90deg, var(--accent-dim), var(--accent)); box-shadow: 0 0 10px -1px var(--accent); }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def image_to_base64_and_format(img: Image.Image) -> tuple[str, str]:
    fmt = (img.format or "PNG").upper()
    if fmt not in ("PNG", "JPEG"):
        fmt = "PNG"
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode(), fmt.lower()


def render_scan_frame(container, img_b64: str, img_fmt: str, status_label: str, status_class: str, scanning: bool = False):
    sweep_html = '<div class="scan-sweep"></div>' if scanning else ""
    container.markdown(
        f"""
        <div class="scan-frame">
            <div class="corner tl"></div>
            <div class="corner tr"></div>
            <div class="corner bl"></div>
            <div class="corner br"></div>
            <div class="status-badge {status_class}"><span class="dot"></span>{status_label}</div>
            <img class="scan-img" src="data:image/{img_fmt};base64,{img_b64}" />
            {sweep_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────

st.markdown(
    """
    <div class="dx-eyebrow"><span class="dot"></span>DIAGNOSTIC IMAGING · AI PIPELINE</div>
    <div class="dx-title">🩻 Chest X-Ray AI Diagnoser</div>
    <div class="dx-sub">
        Upload a medical image. It is first screened by the <b>Gatekeeper</b> model —
        if confirmed as a chest X-ray, it is routed to the <b>Diagnoser</b> model
        for classification.
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["png", "jpg", "jpeg"],
    label_visibility="collapsed",
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    img_b64, img_fmt = image_to_base64_and_format(image)

    viewport = st.empty()
    render_scan_frame(viewport, img_b64, img_fmt, "STANDBY", "idle")

    analyze_clicked = st.button("Run Analysis")

    if analyze_clicked:

        render_scan_frame(viewport, img_b64, img_fmt, "ANALYZING", "active", scanning=True)

        try:

            response = requests.post(
                url=f"{FAST_API_URL}/predict/",
                files={
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type
                    )
                },
                timeout=60
            )

            response.raise_for_status()

            result = response.json()

        except requests.exceptions.ConnectionError as e:
            render_scan_frame(viewport, img_b64, img_fmt, "LINK ERROR", "danger")
            st.error(f"Unable to connect to FastAPI:\n{e}")
            st.stop()

        except requests.exceptions.Timeout:
            render_scan_frame(viewport, img_b64, img_fmt, "TIMEOUT", "danger")
            st.error("Request timed out.")
            st.stop()

        except requests.exceptions.RequestException as e:
            render_scan_frame(viewport, img_b64, img_fmt, "API ERROR", "danger")
            st.error(f"API Error:\n{e}")
            st.stop()

        #  Gatekeeper rejected 
        if not result["accepted"]:

            render_scan_frame(viewport, img_b64, img_fmt, "REJECTED", "danger")

            st.markdown('<div class="dx-stamp">❌ Image Rejected</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="dx-msg">{result["message"]}</div>', unsafe_allow_html=True)

            st.markdown(
                f"""
                <div class="dx-readouts">
                    <div class="dx-readout hl-danger">
                        <div class="label">Gatekeeper Confidence</div>
                        <div class="value">{result['gatekeeper_confidence']*100:.2f}%</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Gatekeeper accepted
        else:

            render_scan_frame(viewport, img_b64, img_fmt, "VALID SCAN", "success")

            st.markdown('<div class="dx-stamp ok">✅ Valid Chest X-Ray</div>', unsafe_allow_html=True)

            st.markdown(
                f"""
                <div class="dx-readouts">
                    <div class="dx-readout hl">
                        <div class="label">Gatekeeper Confidence</div>
                        <div class="value">{result['gatekeeper_confidence']*100:.2f}%</div>
                    </div>
                    <div class="dx-readout hl">
                        <div class="label">Prediction</div>
                        <div class="value">{result['prediction']}</div>
                    </div>
                    <div class="dx-readout hl">
                        <div class="label">Confidence</div>
                        <div class="value">{result['confidence']*100:.2f}%</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown('<div class="dx-section-label">Class Probabilities</div>', unsafe_allow_html=True)

            sorted_probs = sorted(result["probabilities"].items(), key=lambda kv: kv[1], reverse=True)

            for i, (cls, prob) in enumerate(sorted_probs):
                row_class = "dx-prob-row top1" if i == 0 else "dx-prob-row"
                pct = prob * 100
                st.markdown(
                    f"""
                    <div class="{row_class}">
                        <div class="top">
                            <span class="cls">{cls}</span>
                            <span class="pct">{pct:.2f}%</span>
                        </div>
                        <div class="dx-bar-track">
                            <div class="dx-bar-fill" style="width:{pct:.2f}%;"></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )