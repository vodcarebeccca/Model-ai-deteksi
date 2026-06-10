# 🔍 AI Model Detector — Panduan Lengkap

Aplikasi ini mendeteksi apakah model AI yang ditawarkan oleh provider adalah **asli langsung dari vendor** atau **cracked/resold** melalui proxy/aggregator.

---

## 🚀 Setup & Run

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Jalankan Aplikasi
```bash
streamlit run app.py
```

Aplikasi akan buka di `http://localhost:8501`

---

## 📋 Cara Kerja Detection

### 1️⃣ Static Analysis (Tanpa Probe)
Dijalankan otomatis untuk **setiap model** berdasarkan:

| Indikator | Apa yang dicek | Skor |
|---|---|---|
| **Model ID Pattern** | Apakah nama model cocok dengan vendor resmi (OpenAI, Anthropic, Google, dll) | ±10 |
| **Owned By Field** | Apakah field `owned_by` mengandung clue proxy (openrouter, together, deepinfra, etc) | ±15 |
| **Known Proxy Names** | Apakah ada kata kunci agregator terkenal dalam model ID atau owned_by | ±15 |

**Cracked Score Update**: Menambah 15-20 poin jika ada tanda proxy

---

### 2️⃣ Dynamic Analysis (Dengan Probe)
Dijalankan **jika Anda enable mode Probe** di sidebar.

#### Test 1: Identity Test
```
Prompt: "What is your model name and version?"
Detects: 
  - Self-identification accuracy
  - Model ID mismatch between request & response
  - Latency (routing overhead indicator)
```

**🚨 Critical Signal**: Jika Anda minta `claude-opus-4-8` tapi model respond "I am Claude 3.5 Sonnet" → **MISMATCH = Resold**

#### Test 2: Training Cutoff Test
```
Prompt: "What is your training cutoff date?"
Detects:
  - Knowledge accuracy
  - If model response matches expected cutoff
```

#### Metrics Collected
- **Latency**: Dalam milliseconds
  - `< 500ms` → Likely direct endpoint
  - `500ms - 2500ms` → Normal with some routing
  - `> 5000ms` → Heavy proxy routing / far away server

- **Model ID Mismatch**
  - Request ≠ Response → **CRITICAL INDICATOR** (+35 cracked_score)

- **Headers Fingerprinting**
  - Presence of `x-ratelimit-limit` → Legit API gateway
  - Missing rate limit headers → Minimal proxy wrapper

---

## 🎯 Understanding the Verdict

### Cracked Score Ranges

| Score | Verdict | Meaning |
|---|---|---|
| **0-24** | ✅ KEMUNGKINAN ASLI | Model likely genuine from vendor |
| **25-44** | ❓ MUNGKIN PROXY | Possible proxy, but unclear |
| **45-69** | ⚠️ KEMUNGKINAN CRACKED | Likely resold/proxied model |
| **70-100** | 🚨 CRACKED / RESOLD | Very high confidence model is stolen/proxied |

### Confidence Score
Orthogonal metric from 0-100 showing how sure the analysis is.
- High confidence + high cracked score = **Very likely cracked**
- Low confidence + high cracked score = **Needs more probe tests**

---

## 🔎 Red Flags for "Cracked" Models

### 🚨 CRITICAL (High Impact)
1. **Model ID Mismatch**
   - Requested: `claude-opus-4-8`
   - Received: `claude-opus-4-6` or different model entirely
   - **Score Impact**: +35

2. **Self-Identification Mismatch**
   - Model says it's "GPT-4" but you requested "Claude"
   - **Score Impact**: +30

### 🟡 STRONG (Medium Impact)
3. **High Latency** (> 5000ms)
   - Indicates request routing through proxy/far endpoint
   - **Score Impact**: +20

4. **Missing Rate Limit Headers**
   - Legit API gateways expose rate limit metadata
   - **Score Impact**: +5-10

5. **Proxy Keywords in Metadata**
   - `owned_by: "openrouter"`, `model_id: "openrouter/..."`, etc
   - **Score Impact**: +15-20

### 🟠 SUSPICIOUS (Low-Medium Impact)
6. **Very Low Latency** (< 300ms)
   - Unusually fast for distant endpoint, might indicate local fake
   - **Score Impact**: -5

---

## 💰 Why This Matters: The "Cracked Model" Problem

### How It Works (Example: Opus 4.8)

```
Scenario: Illegitimate Provider ("ai.botclaw.top")

1. They somehow access Anthropic's Opus 4.8 endpoint
   (via stolen API key, leaked credentials, insider access, etc)

2. They wrap it in their own gateway:
   ai.botclaw.top/v1/chat/completions
     ↓
   Their proxy server
     ↓
   api.anthropic.com/v1/messages (using their stolen key)

3. They charge YOU 50% less than official Anthropic price

4. Problem:
   - They're reselling stolen compute
   - Your usage might hit THEIR rate limits
   - API key could be revoked anytime
   - Your data flows through unauthorized servers
   - Terms of service violated
```

### Why Detection Matters
- **For Users**: Avoid paying for unreliable models that could die anytime
- **For Vendors**: Protect their IP and ensure service quality
- **For Security**: Identify supply chain vulnerabilities

---

## 📊 Interpreting the JSON Export

When you export results, you get JSON like:

```json
{
  "model_id": "claude-opus-4-8",
  "owned_by": "Anthropic",
  "provider": "anthropic",
  "verdict": "⚠️ KEMUNGKINAN CRACKED",
  "confidence": 72,
  "cracked_score": 65,
  "proxy_hint": false,
  "probed": true,
  "latency_ms": 3200,
  "reported_model": "claude-opus-4-6",
  "model_id_mismatch": true,
  "self_identified": "Claude 3.5 Sonnet"
}
```

**Reading the Signals:**
- `model_id_mismatch: true` → 🚨 Request/response IDs don't match
- `latency_ms: 3200` → ⏱️ Moderate routing overhead
- `reported_model != model_id` → ⚠️ Different model returned
- `cracked_score: 65` → ⚠️ High suspicion of proxy/resale

---

## 🛠️ Advanced Usage

### Custom Provider Detection
Edit `PROVIDER_SIGNATURES` in `app.py`:

```python
PROVIDER_SIGNATURES = {
    "your_provider": {
        "keywords": ["model-prefix-", "internal-"],
        "color": "#hexcolor"
    }
}
```

### Custom Proxy Clues
Edit `PROXY_CLUES` in `app.py`:

```python
PROXY_CLUES = [
    "openrouter", "together", "your_gateway_name", ...
]
```

### Extend Probing Tests
Add more test payloads in the `probe_model()` function:

```python
payload_3 = {
    "model": model_id,
    "messages": [{"role": "user", "content": "Your custom test..."}],
    "max_tokens": 50,
}
```

---

## ⚠️ Limitations

- **OpenAI-compatible APIs only** — Standard `/v1/models` and `/v1/chat/completions` endpoints
- **Requires active API key** — Can't probe without valid credentials
- **Token usage** — Each probe consumes tokens from your API quota
- **Rate limiting** — Provider might throttle multiple rapid probes
- **Proxy hiding** — Very sophisticated proxies might spoof headers/latency

---

## 📝 Sample Workflow

### Step 1: Load Models
1. Paste API key in sidebar
2. Set base URL (default: `https://ai.botclaw.top/v1`)
3. Click "🚀 Ambil Daftar Model"
4. View summary metrics

### Step 2: Static Analysis
- Browse through models
- Check verdicts from name patterns and owned_by field
- Look for obvious proxy keywords

### Step 3: Enable Probe Mode (Optional)
1. Toggle "Probe Model" in sidebar
2. Click ⚡ **Probe Model** on models you want to investigate
3. Wait for identity & cutoff tests to complete

### Step 4: Review Results
- High cracked_score + model_id_mismatch = **🚨 Likely cracked**
- Download JSON for bulk analysis
- Check latency and headers for additional signals

### Step 5: Export & Report
- Download JSON summary
- Share findings or use for internal audit
- Track changes over time

---

## 🤔 FAQ

**Q: Bagaimana cara mereka "crack" model Opus?**
A: Bisa dari: stolen API keys, insider leaks, compromised accounts, atau unauthorized API access. Sekali mereka punya akses, mereka bisa proxy requests.

**Q: Apakah model hasil proxy ini masih bekerja dengan baik?**
A: Ya, tapi dengan risiko:
- Kecepatan lebih lambat (routing overhead)
- Bisa dimatikan kapan saja jika Anthropic deteksi
- Bisa ada downtime atau rate limit
- Data privacy tidak terjamin

**Q: Bagaimana Anthropic mendeteksi dan melindungi?**
A: Mereka monitor:
- Unusual usage patterns
- API key anomalies
- Cross-checking model behavior
- Sending honeypot/test prompts

**Q: Apakah menggunakan cracked model illegal?**
A: Ya, itu:
- Intellectual property theft
- Terms of service violation
- Potentially wire fraud/unauthorized access
- Could expose you to legal liability

**Q: Tool ini bisa digunakan untuk attack?**
A: Tool ini untuk **detection & transparency** saja. Tujuannya membantu identify masalah, bukan membuat masalah.

---

## 📞 Support & Feedback

- Issues atau feature request → Open issue di repo
- Questions → Check the FAQ section
- Contribute → Pull requests welcome!

---

## 📄 License

This tool is provided as-is for educational and security research purposes.

---

**Last Updated**: 2026-06-10  
**Version**: 1.0
