import streamlit as st
import requests
import json
import time
import re
import threading
from datetime import datetime
from typing import Optional, Dict, List, Any
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Model Detector Pro",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS (More Premium) ─────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0a0c12 0%, #11151f 100%);
    }
    .hero-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #38bdf8, #a855f7, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -1.5px;
        margin-bottom: 0;
    }
    .hero-sub { color: #64748b; font-size: 1.05rem; }
    .card {
        background: rgba(255,255,255,0.045);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
    }
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .badge-real { background: rgba(74,222,128,0.15); color: #4ade80; border: 1px solid #4ade80; }
    .badge-proxy { background: rgba(251,146,60,0.15); color: #fb923c; border: 1px solid #fb923c; }
    .verdict-real { color: #4ade80; font-weight: 700; }
    .verdict-proxy { color: #fb923c; font-weight: 700; }
    .metric-value { font-family: 'JetBrains Mono', monospace; font-weight: 700; }
    .probe-log {
        font-family: 'JetBrains Mono', monospace;
        background: #0f172a;
        padding: 12px;
        border-radius: 8px;
        max-height: 320px;
        overflow-y: auto;
        font-size: 0.82rem;
    }
</style>
""", unsafe_allow_html=True)

# ─── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_BASE_URL = "https://ai.botclaw.top/v1"

PROVIDER_SIGNATURES = {
    "openai":     {"keywords": ["gpt", "o1", "o3", "davinci", "curie", "babbage", "whisper", "dall-e", "text-embedding"], "color": "#10a37f"},
    "anthropic":  {"keywords": ["claude"], "color": "#d97706"},
    "google":     {"keywords": ["gemini", "palm", "bison", "gecko"], "color": "#4285f4"},
    "meta":       {"keywords": ["llama", "codellama", "meta"], "color": "#0866ff"},
    "mistral":    {"keywords": ["mistral", "mixtral", "codestral"], "color": "#ef4444"},
    "deepseek":   {"keywords": ["deepseek"], "color": "#8b5cf6"},
    "qwen":       {"keywords": ["qwen", "qwq"], "color": "#06b6d4"},
    "cohere":     {"keywords": ["command", "embed"], "color": "#6366f1"},
    "01.ai":      {"keywords": ["yi-"], "color": "#f59e0b"},
    "perplexity": {"keywords": ["pplx", "sonar"], "color": "#22c55e"},
    "nous":       {"keywords": ["nous", "hermes", "capybara"], "color": "#a855f7"},
    "microsoft":  {"keywords": ["phi-", "wizardlm"], "color": "#0078d4"},
    "groq":       {"keywords": ["groq", "llama3-groq"], "color": "#f97316"},
}

PROXY_CLUES = [
    "openrouter", "together", "replicate", "fireworks", "anyscale", "groq",
    "lepton", "octo", "deepinfra", "modal", "novita", "botclaw", "aiml",
    "corcel", "siliconflow", "nebius", "hyperbolic", "massedcompute", "runpod"
]

# ─── Helper Functions ─────────────────────────────────────────────────────────
def make_headers(api_key: str) -> dict:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

def detect_provider(model_id: str) -> tuple[str, str]:
    m = model_id.lower()
    for provider, meta in PROVIDER_SIGNATURES.items():
        if any(kw in m for kw in meta["keywords"]):
            return provider, meta["color"]
    return "unknown", "#64748b"

def is_likely_proxied(model_id: str, owned_by: str) -> bool:
    combined = (model_id + " " + (owned_by or "")).lower()
    return any(clue in combined for clue in PROXY_CLUES)

def advanced_probe(api_key: str, base_url: str, model_id: str) -> dict:
    """Advanced probing with multiple signals"""
    url = base_url.rstrip("/") + "/chat/completions"
    payloads = [
        {"model": model_id, "messages": [{"role": "user", "content": "Reply only with your exact model name and version."}], "max_tokens": 60, "temperature": 0},
        {"model": model_id, "messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Who are you?"}], "max_tokens": 80, "temperature": 0},
    ]
    
    results = {"ok": True, "probes": [], "latency_ms": [], "model_mismatch": False}
    t_start = time.time()

    for i, payload in enumerate(payloads):
        try:
            t0 = time.time()
            resp = requests.post(url, headers=make_headers(api_key), json=payload, timeout=25)
            latency = round((time.time() - t0) * 1000)
            results["latency_ms"].append(latency)
            
            resp.raise_for_status()
            data = resp.json()
            
            reported = data.get("model", model_id)
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            results["probes"].append({"content": content, "reported": reported})
            
            if reported.lower() != model_id.lower():
                results["model_mismatch"] = True
                
        except Exception as e:
            results["probes"].append({"error": str(e)})
    
    results["total_latency"] = round(time.time() - t_start * 1000)
    return results

def build_analysis(model_id: str, owned_by: str, probe: Optional[dict]) -> dict:
    provider, color = detect_provider(model_id)
    proxied_hint = is_likely_proxied(model_id, owned_by)
    
    signals = []
    confidence = 92
    verdict = "ASLI"
    
    if proxied_hint:
        signals.append("⚠️ Nama model mengandung indikasi aggregator/proxy")
        confidence -= 25
        verdict = "PROXY"
    
    if probe and probe.get("ok"):
        if probe.get("model_mismatch"):
            signals.append("⚠️ Model mismatch terdeteksi (proxy/aliasing)")
            confidence -= 20
            verdict = "PROXY"
        
        avg_lat = sum(probe.get("latency_ms", [0])) / max(len(probe.get("latency_ms", [])), 1)
        if avg_lat > 7000:
            signals.append(f"🐢 Latensi tinggi ({int(avg_lat)}ms) — kemungkinan routing")
        elif avg_lat < 600:
            signals.append(f"⚡ Latensi sangat rendah ({int(avg_lat)}ms)")
    
    if not signals:
        signals.append("✅ Tidak ditemukan indikasi proxy")
    
    if confidence >= 75:
        verdict_display = "ASLI"
    elif confidence >= 50:
        verdict_display = "MUNGKIN PROXY"
    else:
        verdict_display = "KEMUNGKINAN PROXY"
    
    return {
        "provider": provider,
        "color": color,
        "proxied_hint": proxied_hint,
        "signals": signals,
        "verdict": verdict_display,
        "confidence": max(0, min(100, confidence)),
    }

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p style="font-family:\'JetBrains Mono\';color:#38bdf8;font-weight:700;font-size:1.2rem;">⚙️ AI Model Detector Pro</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    api_key = st.text_input("API Key", type="password", placeholder="sk-...")
    base_url = st.text_input("Base URL", value=DEFAULT_BASE_URL)
    
    st.markdown("---")
    do_probe = st.toggle("Aktifkan Advanced Probe", value=True, help="Gunakan token untuk fingerprinting mendalam")
    batch_size = st.slider("Batch Size (Probe paralel)", 1, 8, 3)
    
    st.caption("v2.0 Advanced • OpenAI Compatible")

# ─── Main UI ───────────────────────────────────────────────────────────────────
st.markdown('<p class="hero-title">🔍 AI Model Detector <span style="font-size:1.1rem;">PRO</span></p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Deteksi model asli vs proxy dari third-party dengan akurasi tinggi</p>', unsafe_allow_html=True)

if not api_key:
    st.info("👈 Masukkan **API Key** di sidebar untuk memulai", icon="🔑")
    st.stop()

# Fetch Models
if "models_data" not in st.session_state:
    st.session_state.models_data = None
if "probe_results" not in st.session_state:
    st.session_state.probe_results = {}

col1, col2 = st.columns([1, 3])
with col1:
    if st.button("🚀 Ambil Daftar Model", type="primary", use_container_width=True):
        with st.spinner("Mengambil daftar model..."):
            try:
                url = base_url.rstrip("/") + "/models"
                resp = requests.get(url, headers=make_headers(api_key), timeout=15)
                resp.raise_for_status()
                st.session_state.models_data = resp.json()
                st.success("✅ Berhasil mengambil daftar model")
            except Exception as e:
                st.error(f"Gagal: {e}")

if st.session_state.models_data:
    raw = st.session_state.models_data
    models = raw.get("data", []) if isinstance(raw, dict) else raw
    if not isinstance(models, list):
        models = [models]

    st.markdown(f"### 📊 Ditemukan **{len(models)}** Model")

    # Filters
    col_s, col_f, col_a = st.columns([3, 2, 2])
    with col_s:
        search = st.text_input("🔎 Cari model", placeholder="gpt-4, claude...")
    with col_f:
        providers = sorted({detect_provider(m.get("id",""))[0] for m in models})
        provider_filter = st.selectbox("Provider", ["Semua"] + providers)
    with col_a:
        if st.button("🔬 Probe Semua Model", type="secondary"):
            st.info("Batch probing dimulai...")

    # Filtered models
    filtered = [m for m in models 
                if (not search or search.lower() in m.get("id","").lower()) and
                   (provider_filter == "Semua" or detect_provider(m.get("id",""))[0] == provider_filter)]

    # Display
    for m in filtered:
        mid = m.get("id", "N/A")
        owned_by = m.get("owned_by", "")
        provider, color = detect_provider(mid)
        probe = st.session_state.probe_results.get(mid)
        
        analysis = build_analysis(mid, owned_by, probe)
        
        with st.expander(f"**{mid}** — {provider.upper()}", expanded=False):
            c1, c2 = st.columns([3, 2])
            with c1:
                st.markdown(f"""
                <div class="card">
                    <span style="font-family:'JetBrains Mono';font-size:1.1rem;">{mid}</span><br>
                    <span class="badge {'badge-proxy' if analysis['proxied_hint'] else 'badge-real'}">
                        {'PROXY' if analysis['proxied_hint'] else 'DIRECT'}
                    </span>
                    <br><br>
                    <b>Provider:</b> <span style="color:{color}">{provider}</span><br>
                    <b>Owned by:</b> <code>{owned_by or '—'}</code>
                </div>
                """, unsafe_allow_html=True)
            
            with c2:
                verdict_class = "verdict-real" if "ASLI" in analysis["verdict"] else "verdict-proxy"
                st.markdown(f"""
                <div class="card">
                    <div style="font-size:0.8rem;color:#64748b;">VERDICT</div>
                    <div class="metric-value {verdict_class}" style="font-size:1.6rem;">{analysis["verdict"]}</div>
                    <div style="color:#38bdf8;font-weight:600;">{analysis["confidence"]}% Confidence</div>
                </div>
                """, unsafe_allow_html=True)
            
            for sig in analysis["signals"]:
                st.write(sig)
            
            if do_probe:
                if st.button("🔬 Probe Sekarang", key=f"probe_{mid}"):
                    with st.spinner(f"Probing {mid}..."):
                        result = advanced_probe(api_key, base_url, mid)
                        st.session_state.probe_results[mid] = result
                        st.rerun()

            if probe:
                st.markdown("**📡 Probe Result**")
                st.json(probe, expanded=False)

    # Export
    st.markdown("---")
    if st.button("📥 Export Full Report"):
        summary = []
        for m in models:
            mid = m.get("id")
            analysis = build_analysis(mid, m.get("owned_by"), st.session_state.probe_results.get(mid))
            summary.append({
                "model": mid,
                "provider": analysis["provider"],
                "verdict": analysis["verdict"],
                "confidence": analysis["confidence"],
                "proxied_hint": analysis["proxied_hint"]
            })
        
        df = pd.DataFrame(summary)
        csv = df.to_csv(index=False).encode()
        st.download_button("Download CSV", csv, f"model_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", "text/csv")

# Footer
st.caption("AI Model Detector Pro v2.0 — Dibuat untuk mendeteksi proxy dengan akurat")
