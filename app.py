import streamlit as st
import requests
import json
import time
import re
from datetime import datetime
from typing import Optional

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Model Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
  }

  /* Dark theme background */
  .stApp {
    background: linear-gradient(135deg, #0d0f14 0%, #111520 50%, #0d0f14 100%);
  }

  /* Main title */
  .hero-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.2rem;
    font-weight: 600;
    color: #e2e8f0;
    letter-spacing: -0.5px;
    margin-bottom: 0;
  }
  .hero-sub {
    font-size: 0.95rem;
    color: #64748b;
    margin-top: 4px;
    font-weight: 400;
  }
  .accent { color: #38bdf8; }

  /* Card containers */
  .card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
  }

  /* Provider badge */
  .badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  .badge-real    { background: rgba(34,197,94,0.15);  color: #4ade80; border: 1px solid rgba(34,197,94,0.3); }
  .badge-proxy   { background: rgba(251,146,60,0.15); color: #fb923c; border: 1px solid rgba(251,146,60,0.3); }
  .badge-unknown { background: rgba(148,163,184,0.15);color: #94a3b8; border: 1px solid rgba(148,163,184,0.3); }

  /* Result metric */
  .metric-box {
    background: rgba(56,189,248,0.05);
    border: 1px solid rgba(56,189,248,0.15);
    border-radius: 8px;
    padding: 14px 18px;
    text-align: center;
  }
  .metric-label { font-size: 0.72rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.8px; }
  .metric-value { font-size: 1.5rem; font-weight: 700; color: #38bdf8; font-family: 'JetBrains Mono', monospace; }

  /* Code block style */
  .code-block {
    background: rgba(0,0,0,0.4);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 8px;
    padding: 14px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    color: #94a3b8;
    overflow-x: auto;
    white-space: pre-wrap;
  }

  /* Verdict colors */
  .verdict-real    { color: #4ade80; font-weight: 700; }
  .verdict-proxy   { color: #fb923c; font-weight: 700; }
  .verdict-unknown { color: #94a3b8; font-weight: 700; }

  /* Sidebar styling */
  section[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.02);
    border-right: 1px solid rgba(255,255,255,0.06);
  }

  /* Remove default streamlit padding */
  .block-container { padding-top: 2rem; }

  /* Progress / spinner override */
  .stSpinner > div { border-top-color: #38bdf8 !important; }

  /* Table */
  .result-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
  .result-table th {
    color: #64748b; text-transform: uppercase; font-size: 0.7rem;
    letter-spacing: 0.8px; padding: 8px 12px;
    border-bottom: 1px solid rgba(255,255,255,0.06); text-align: left;
  }
  .result-table td {
    padding: 10px 12px; color: #e2e8f0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
  }
  .result-table tr:last-child td { border-bottom: none; }
</style>
""", unsafe_allow_html=True)


# ─── Constants ───────────────────────────────────────────────────────────────────
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
}

PROXY_CLUES = [
    "openrouter", "together", "replicate", "fireworks", "anyscale",
    "perplexity", "groq", "lepton", "octo", "deepinfra", "modal",
    "novita", "botclaw", "aiml", "corcel",
]


# ─── Helper Functions ─────────────────────────────────────────────────────────────
def make_headers(api_key: str) -> dict:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def detect_provider(model_id: str) -> tuple[str, str]:
    """Return (provider_name, hex_color) based on model ID keywords."""
    m = model_id.lower()
    for provider, meta in PROVIDER_SIGNATURES.items():
        if any(kw in m for kw in meta["keywords"]):
            return provider, meta["color"]
    return "unknown", "#64748b"


def is_likely_proxied(model_id: str, owned_by: str) -> bool:
    """Heuristic: check if model is likely proxied through aggregator."""
    combined = (model_id + " " + (owned_by or "")).lower()
    return any(clue in combined for clue in PROXY_CLUES)


def fetch_models(api_key: str, base_url: str) -> dict:
    """Fetch /v1/models from the endpoint."""
    url = base_url.rstrip("/") + "/models"
    try:
        resp = requests.get(url, headers=make_headers(api_key), timeout=15)
        resp.raise_for_status()
        return {"ok": True, "data": resp.json(), "status": resp.status_code}
    except requests.exceptions.ConnectionError:
        return {"ok": False, "error": "❌ Tidak bisa terhubung ke server. Periksa URL."}
    except requests.exceptions.Timeout:
        return {"ok": False, "error": "⏳ Request timeout setelah 15 detik."}
    except requests.exceptions.HTTPError as e:
        code = e.response.status_code if e.response else "?"
        body = ""
        try:
            body = e.response.json()
        except Exception:
            pass
        if code == 401:
            return {"ok": False, "error": "🔐 API key tidak valid atau tidak memiliki akses."}
        return {"ok": False, "error": f"HTTP {code}: {body or str(e)}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def probe_model(api_key: str, base_url: str, model_id: str) -> dict:
    """
    Send a minimal chat completion to detect actual model behavior.
    Returns latency, response snippet, reported model ID, and fingerprint clues.
    """
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": "Reply only: what is your model name and version?"}],
        "max_tokens": 80,
        "temperature": 0,
    }
    t0 = time.time()
    try:
        resp = requests.post(url, headers=make_headers(api_key), json=payload, timeout=30)
        latency = round((time.time() - t0) * 1000)
        resp.raise_for_status()
        data = resp.json()

        # Extract fields
        choice      = data.get("choices", [{}])[0]
        message     = choice.get("message", {})
        content     = message.get("content", "")
        finish      = choice.get("finish_reason", "")
        reported_id = data.get("model", model_id)
        usage       = data.get("usage", {})

        # Fingerprint: does reported model differ from requested?
        model_mismatch = reported_id.lower() != model_id.lower()

        # Look for self-identification in response
        self_id = None
        patterns = [
            r"(?:i am|i'm|my (?:name|model|version) is|i(?:'m| am) called)\s+([A-Za-z0-9.\-_ ]+)",
            r"(?:model|version):\s*([A-Za-z0-9.\-_]+)",
        ]
        for p in patterns:
            m = re.search(p, content, re.IGNORECASE)
            if m:
                self_id = m.group(1).strip()
                break

        return {
            "ok": True,
            "latency_ms": latency,
            "content": content,
            "finish_reason": finish,
            "reported_model": reported_id,
            "model_mismatch": model_mismatch,
            "self_identified": self_id,
            "usage": usage,
            "raw": data,
        }
    except requests.exceptions.Timeout:
        return {"ok": False, "error": f"Timeout setelah 30 detik", "latency_ms": 30000}
    except requests.exceptions.HTTPError as e:
        code = e.response.status_code if e.response else "?"
        return {"ok": False, "error": f"HTTP {code}", "latency_ms": round((time.time() - t0) * 1000)}
    except Exception as e:
        return {"ok": False, "error": str(e), "latency_ms": round((time.time() - t0) * 1000)}


def build_analysis(model_id: str, owned_by: str, probe: Optional[dict]) -> dict:
    """Aggregate all signals into a final analysis report."""
    provider, color = detect_provider(model_id)
    proxied_hint    = is_likely_proxied(model_id, owned_by)

    signals = []
    verdict = "ASLI"
    confidence = 85

    if proxied_hint:
        signals.append("⚠️ Nama model / owned_by mengandung clue aggregator/proxy")
        verdict = "PROXY"
        confidence -= 20

    if probe and probe.get("ok"):
        if probe.get("model_mismatch"):
            signals.append(f"⚠️ Model ID yang dilaporkan berbeda: `{probe['reported_model']}`")
            verdict = "PROXY"
            confidence -= 15

        if probe.get("self_identified"):
            sid = probe["self_identified"].lower()
            req = model_id.lower()
            if not any(word in req for word in sid.split()[:2]):
                signals.append(f"⚠️ Model mengidentifikasi dirinya sebagai **{probe['self_identified']}** (tidak cocok)")
                verdict = "MUNGKIN PROXY"
                confidence -= 10
            else:
                signals.append(f"✅ Model mengkonfirmasi identitasnya: **{probe['self_identified']}**")
                confidence = min(confidence + 10, 99)

        lat = probe.get("latency_ms", 0)
        if lat > 8000:
            signals.append(f"🐌 Latensi tinggi ({lat}ms) – bisa jadi routing melalui beberapa hop")
        elif lat < 500:
            signals.append(f"⚡ Latensi sangat rendah ({lat}ms)")

    if not signals:
        signals.append("✅ Tidak ada tanda proxy yang terdeteksi")

    if confidence >= 80:
        verdict_display = "ASLI"
    elif confidence >= 55:
        verdict_display = "MUNGKIN PROXY"
    else:
        verdict_display = "KEMUNGKINAN PROXY"

    return {
        "provider":     provider,
        "color":        color,
        "proxied_hint": proxied_hint,
        "signals":      signals,
        "verdict":      verdict_display,
        "confidence":   confidence,
    }


# ─── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p style="font-family:\'JetBrains Mono\',monospace;color:#38bdf8;font-weight:600;font-size:1.1rem;">⚙️ Konfigurasi</p>', unsafe_allow_html=True)
    st.markdown("---")

    api_key = st.text_input(
        "API Key",
        type="password",
        placeholder="sk-...",
        help="Masukkan API key dari provider",
    )

    base_url = st.text_input(
        "Base URL",
        value=DEFAULT_BASE_URL,
        help="Endpoint API (OpenAI-compatible)",
    )

    st.markdown("---")
    st.markdown("**Mode Analisis**")
    do_probe = st.toggle(
        "Probe Model (Chat Completion)",
        value=False,
        help="Kirim pesan test ke model untuk fingerprinting. Menggunakan token/kuota.",
    )

    if do_probe:
        st.info("⚡ Mode probe aktif. Akan menggunakan token dari API key Anda untuk setiap model yang dipilih.", icon="ℹ️")

    st.markdown("---")
    st.caption("v1.0 · AI Model Detector")
    st.caption("Support: OpenAI-compatible APIs")


# ─── Main UI ─────────────────────────────────────────────────────────────────────
st.markdown('<p class="hero-title">🔍 AI <span class="accent">Model</span> Detector</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Deteksi model AI asli vs proxy dari third-party provider — <code>ai.botclaw.top</code></p>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

if not api_key:
    st.markdown("""
    <div class="card">
      <p style="color:#94a3b8;margin:0;">
        👈 Masukkan <strong>API Key</strong> di sidebar untuk memulai.<br>
        Kemudian klik <strong>Ambil Daftar Model</strong> untuk melihat semua model yang tersedia.
      </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─── Fetch Models ─────────────────────────────────────────────────────────────────
col_btn, col_info = st.columns([2, 5])
with col_btn:
    fetch_btn = st.button("🚀 Ambil Daftar Model", type="primary", use_container_width=True)

if "models_data" not in st.session_state:
    st.session_state.models_data = None
if "probe_results" not in st.session_state:
    st.session_state.probe_results = {}

if fetch_btn:
    with st.spinner("Menghubungi server..."):
        result = fetch_models(api_key, base_url)
    if result["ok"]:
        st.session_state.models_data = result["data"]
        st.session_state.probe_results = {}
        st.success(f"✅ Berhasil! Status HTTP {result['status']}")
    else:
        st.error(result["error"])

# ─── Display Models ───────────────────────────────────────────────────────────────
if st.session_state.models_data:
    raw = st.session_state.models_data
    models = raw.get("data", raw) if isinstance(raw, dict) else raw
    if not isinstance(models, list):
        models = [models]

    st.markdown(f"### 📋 Ditemukan **{len(models)}** Model")
    st.markdown("---")

    # Summary stats
    providers_found = {}
    for m in models:
        mid = m.get("id", "")
        p, _ = detect_provider(mid)
        providers_found[p] = providers_found.get(p, 0) + 1

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f'<div class="metric-box"><div class="metric-label">Total Model</div><div class="metric-value">{len(models)}</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-box"><div class="metric-label">Provider Terdeteksi</div><div class="metric-value">{len(providers_found)}</div></div>', unsafe_allow_html=True)
    with m3:
        top_provider = max(providers_found, key=providers_found.get) if providers_found else "-"
        st.markdown(f'<div class="metric-box"><div class="metric-label">Provider Terbanyak</div><div class="metric-value" style="font-size:1rem;padding-top:6px">{top_provider}</div></div>', unsafe_allow_html=True)
    with m4:
        proxied_count = sum(1 for m in models if is_likely_proxied(m.get("id",""), m.get("owned_by","")))
        pct = round(proxied_count / len(models) * 100) if models else 0
        st.markdown(f'<div class="metric-box"><div class="metric-label">Indikasi Proxy</div><div class="metric-value">{proxied_count} <span style="font-size:0.9rem;color:#64748b">({pct}%)</span></div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Filter / search
    col_s, col_f = st.columns([3, 2])
    with col_s:
        search_q = st.text_input("🔎 Cari model", placeholder="gpt, claude, llama...", label_visibility="collapsed")
    with col_f:
        provider_filter = st.selectbox(
            "Filter provider",
            options=["Semua"] + sorted(providers_found.keys()),
            label_visibility="collapsed",
        )

    filtered = []
    for m in models:
        mid = m.get("id", "")
        p, _ = detect_provider(mid)
        if search_q and search_q.lower() not in mid.lower():
            continue
        if provider_filter != "Semua" and p != provider_filter:
            continue
        filtered.append(m)

    st.caption(f"Menampilkan {len(filtered)} dari {len(models)} model")
    st.markdown("---")

    # ── Per-model cards ──────────────────────────────────────────────────────────
    for m in filtered:
        mid        = m.get("id", "N/A")
        owned_by   = m.get("owned_by", "")
        created_ts = m.get("created")
        created_dt = datetime.utcfromtimestamp(created_ts).strftime("%Y-%m-%d") if created_ts else "N/A"

        provider, color = detect_provider(mid)
        proxied_hint    = is_likely_proxied(mid, owned_by)

        badge_class = "badge-proxy" if proxied_hint else "badge-real"
        badge_text  = "PROXY HINT" if proxied_hint else "DIRECT"

        with st.expander(f"**{mid}**  —  `{provider}`", expanded=False):
            c1, c2 = st.columns([3, 2])

            with c1:
                st.markdown(f"""
                <div class="card" style="margin-bottom:10px">
                  <span style="font-family:'JetBrains Mono',monospace;font-size:1.05rem;color:#e2e8f0;">{mid}</span>
                  &nbsp;&nbsp;<span class="badge {badge_class}">{badge_text}</span>
                  <br><br>
                  <table class="result-table">
                    <tr><th>Field</th><th>Value</th></tr>
                    <tr><td>owned_by</td><td><code>{owned_by or "—"}</code></td></tr>
                    <tr><td>created</td><td>{created_dt}</td></tr>
                    <tr><td>Provider terdeteksi</td><td><span style="color:{color};font-weight:600">{provider}</span></td></tr>
                  </table>
                </div>
                """, unsafe_allow_html=True)

            with c2:
                # Quick static analysis
                analysis = build_analysis(mid, owned_by, st.session_state.probe_results.get(mid))
                verdict_class = {
                    "ASLI": "verdict-real",
                    "MUNGKIN PROXY": "verdict-proxy",
                    "KEMUNGKINAN PROXY": "verdict-proxy",
                }.get(analysis["verdict"], "verdict-unknown")

                st.markdown(f"""
                <div class="card">
                  <div class="metric-label">Verdict</div>
                  <div class="metric-value {verdict_class}" style="font-size:1.3rem;margin:6px 0">{analysis["verdict"]}</div>
                  <div class="metric-label">Confidence</div>
                  <div style="font-size:1rem;color:#38bdf8;font-weight:600;font-family:'JetBrains Mono',monospace;">{analysis["confidence"]}%</div>
                </div>
                """, unsafe_allow_html=True)

            # Signals
            st.markdown("**🧪 Sinyal Deteksi:**")
            for sig in analysis["signals"]:
                st.markdown(f"- {sig}")

            # Probe button
            if do_probe:
                probe_key = f"probe_{mid}"
                if st.button(f"⚡ Probe Model", key=probe_key):
                    with st.spinner(f"Mengirim permintaan ke `{mid}`..."):
                        probe_res = probe_model(api_key, base_url, mid)
                    st.session_state.probe_results[mid] = probe_res

            # Show probe result if available
            if mid in st.session_state.probe_results:
                pr = st.session_state.probe_results[mid]
                st.markdown("---")
                st.markdown("**📡 Hasil Probe:**")
                if pr.get("ok"):
                    pc1, pc2, pc3 = st.columns(3)
                    with pc1:
                        st.metric("Latensi", f"{pr['latency_ms']} ms")
                    with pc2:
                        st.metric("Model Dilaporkan", pr.get("reported_model", "—"))
                    with pc3:
                        st.metric("Finish Reason", pr.get("finish_reason", "—"))

                    if pr.get("content"):
                        st.markdown("**Respons Model:**")
                        st.markdown(f'<div class="code-block">{pr["content"]}</div>', unsafe_allow_html=True)

                    if pr.get("model_mismatch"):
                        st.warning(f"⚠️ Model ID yang dikembalikan API (`{pr['reported_model']}`) berbeda dengan yang diminta (`{mid}`). Kemungkinan proxy atau aliasing.")

                    if pr.get("self_identified"):
                        st.info(f"💬 Model mengidentifikasi diri sebagai: **{pr['self_identified']}**")

                    if pr.get("usage"):
                        u = pr["usage"]
                        st.caption(f"Token: prompt={u.get('prompt_tokens','?')}, completion={u.get('completion_tokens','?')}, total={u.get('total_tokens','?')}")
                else:
                    st.error(f"Probe gagal: {pr.get('error','Unknown error')}")

    # ── Raw JSON View ──────────────────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("📄 Raw JSON Response"):
        st.json(st.session_state.models_data)

    # ── Export Summary ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📥 Export Hasil Analisis")

    summary_rows = []
    for m in models:
        mid      = m.get("id", "")
        owned_by = m.get("owned_by", "")
        provider, _ = detect_provider(mid)
        proxied  = is_likely_proxied(mid, owned_by)
        probe    = st.session_state.probe_results.get(mid, {})
        analysis = build_analysis(mid, owned_by, probe if probe.get("ok") else None)
        summary_rows.append({
            "model_id":  mid,
            "owned_by":  owned_by,
            "provider":  provider,
            "verdict":   analysis["verdict"],
            "confidence": analysis["confidence"],
            "proxy_hint": proxied,
            "probed":    bool(probe.get("ok")),
            "latency_ms": probe.get("latency_ms", "—") if probe.get("ok") else "—",
            "reported_model": probe.get("reported_model", "—") if probe.get("ok") else "—",
        })

    export_json = json.dumps(summary_rows, indent=2, ensure_ascii=False)
    st.download_button(
        label="⬇️ Download JSON Ringkasan",
        data=export_json,
        file_name=f"ai_model_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
    )

    # Show as table
    if st.checkbox("Tampilkan tabel ringkasan"):
        import pandas as pd
        df = pd.DataFrame(summary_rows)
        st.dataframe(df, use_container_width=True)
