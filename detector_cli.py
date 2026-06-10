#!/usr/bin/env python3
"""
Standalone CLI tool for detecting cracked/resold AI models.
Tidak perlu Streamlit - bisa run dari command line.

Usage:
    python detector_cli.py --api-key sk-xxx --base-url https://ai.botclaw.top/v1 --probe
"""

import argparse
import json
import sys
import time
import re
from typing import Optional, List
from datetime import datetime
import requests

# ─── Constants ──────────────────────────────────────────────────────────────

DEFAULT_BASE_URL = "https://ai.botclaw.top/v1"

PROVIDER_SIGNATURES = {
    "openai":     ["gpt", "o1", "o3", "davinci", "curie", "whisper", "dall-e", "text-embedding"],
    "anthropic":  ["claude"],
    "google":     ["gemini", "palm", "bison", "gecko"],
    "meta":       ["llama", "codellama"],
    "mistral":    ["mistral", "mixtral"],
    "deepseek":   ["deepseek"],
    "qwen":       ["qwen"],
    "cohere":     ["command"],
    "perplexity": ["pplx", "sonar"],
}

PROXY_CLUES = [
    "openrouter", "together", "replicate", "fireworks", "anyscale",
    "groq", "lepton", "deepinfra", "modal", "botclaw", "aiml",
]

# ─── Helpers ────────────────────────────────────────────────────────────────

def make_headers(api_key: str) -> dict:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

def detect_provider(model_id: str) -> str:
    m = model_id.lower()
    for provider, keywords in PROVIDER_SIGNATURES.items():
        if any(kw in m for kw in keywords):
            return provider
    return "unknown"

def is_likely_proxied(model_id: str, owned_by: str) -> bool:
    combined = (model_id + " " + (owned_by or "")).lower()
    return any(clue in combined for clue in PROXY_CLUES)

def fetch_models(api_key: str, base_url: str) -> dict:
    """Fetch /v1/models from endpoint."""
    url = base_url.rstrip("/") + "/models"
    try:
        resp = requests.get(url, headers=make_headers(api_key), timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return {"ok": True, "data": data, "status": resp.status_code}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def probe_model(api_key: str, base_url: str, model_id: str) -> dict:
    """Run identity and cutoff tests on model."""
    url = base_url.rstrip("/") + "/chat/completions"
    
    payload_identity = {
        "model": model_id,
        "messages": [{"role": "user", "content": "What is your model name and version? One line."}],
        "max_tokens": 80,
        "temperature": 0,
    }
    
    payload_cutoff = {
        "model": model_id,
        "messages": [{"role": "user", "content": "What is your training cutoff date? YYYY-MM format."}],
        "max_tokens": 20,
        "temperature": 0,
    }
    
    t0 = time.time()
    results = {"ok": False, "tests": []}
    
    try:
        # Test 1: Identity
        resp1 = requests.post(url, headers=make_headers(api_key), json=payload_identity, timeout=30)
        lat1 = round((time.time() - t0) * 1000)
        resp1.raise_for_status()
        data1 = resp1.json()
        
        choice1 = data1.get("choices", [{}])[0]
        content1 = choice1.get("message", {}).get("content", "")
        reported_id = data1.get("model", model_id)
        
        results["tests"].append({
            "name": "Identity",
            "latency_ms": lat1,
            "content": content1,
            "reported_model": reported_id,
        })
        
        # Test 2: Cutoff
        t_test2 = time.time()
        resp2 = requests.post(url, headers=make_headers(api_key), json=payload_cutoff, timeout=30)
        lat2 = round((time.time() - t_test2) * 1000)
        resp2.raise_for_status()
        data2 = resp2.json()
        
        choice2 = data2.get("choices", [{}])[0]
        content2 = choice2.get("message", {}).get("content", "")
        
        results["tests"].append({
            "name": "Cutoff",
            "latency_ms": lat2,
            "content": content2,
        })
        
        # Analysis
        model_mismatch = reported_id.lower() != model_id.lower()
        avg_latency = (lat1 + lat2) // 2
        
        # Self-ID pattern
        self_id = None
        for p in [r"(?:i am|i'm|my (?:name|model) is)\s+([A-Za-z0-9.\-_\s]+?)(?:\.|,|$)",
                  r"(?:model|version):\s*([A-Za-z0-9.\-_]+)"]:
            m = re.search(p, content1, re.IGNORECASE)
            if m:
                self_id = m.group(1).strip()
                break
        
        results.update({
            "ok": True,
            "avg_latency_ms": avg_latency,
            "model_mismatch": model_mismatch,
            "self_identified": self_id,
            "reported_model": reported_id,
        })
        
        return results
        
    except Exception as e:
        return {"ok": False, "error": str(e), "tests": results.get("tests", [])}

def analyze(model_id: str, owned_by: str, probe: Optional[dict]) -> dict:
    """Calculate cracked_score and verdict."""
    provider = detect_provider(model_id)
    proxied = is_likely_proxied(model_id, owned_by)
    
    cracked_score = 0
    signals = []
    confidence = 85
    
    # Static
    if proxied:
        signals.append("⚠️  Proxy keywords in metadata")
        cracked_score += 15
        confidence -= 20
    
    # Dynamic
    if probe and probe.get("ok"):
        avg_lat = probe.get("avg_latency_ms", 0)
        
        if probe.get("model_mismatch"):
            signals.append(f"🚨 Model ID mismatch: requested {model_id}, got {probe['reported_model']}")
            cracked_score += 35
            confidence -= 30
        
        if probe.get("self_identified"):
            sid = probe["self_identified"].lower()
            req = model_id.lower()
            req_tokens = set(req.replace("-", " ").split())
            sid_tokens = set(sid.replace("-", " ").split()[:2])
            
            if sid_tokens.isdisjoint(req_tokens):
                signals.append(f"⚠️  Self-ID mismatch: model claims to be {probe['self_identified']}")
                cracked_score += 30
                confidence -= 20
            else:
                signals.append(f"✅ Model confirms identity: {probe['self_identified']}")
                confidence += 5
        
        if avg_lat > 5000:
            signals.append(f"🐌 Very high latency ({avg_lat}ms) - routing overhead")
            cracked_score += 20
        elif avg_lat < 400:
            signals.append(f"⚡ Very low latency ({avg_lat}ms)")
            cracked_score -= 5
    else:
        signals.append("⚠️  No probe data")
    
    if not signals:
        signals.append("✅ No anomalies detected")
    
    # Clamp
    cracked_score = max(0, min(100, cracked_score))
    confidence = max(10, min(99, confidence))
    
    # Verdict
    if cracked_score >= 70:
        verdict = "🚨 CRACKED/RESOLD"
    elif cracked_score >= 45:
        verdict = "⚠️  LIKELY PROXY"
    elif cracked_score >= 25:
        verdict = "❓ MAYBE PROXY"
    else:
        verdict = "✅ LIKELY GENUINE"
    
    return {
        "provider": provider,
        "verdict": verdict,
        "cracked_score": cracked_score,
        "confidence": confidence,
        "signals": signals,
    }

# ─── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Detect cracked/resold AI models from third-party providers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python detector_cli.py --api-key sk-xxx --base-url https://ai.botclaw.top/v1
  python detector_cli.py --api-key sk-xxx --probe --output results.json
  python detector_cli.py --api-key sk-xxx --filter claude --probe
        """,
    )
    
    parser.add_argument(
        "--api-key", "-k",
        required=True,
        help="API key for authentication (Bearer token)"
    )
    parser.add_argument(
        "--base-url", "-b",
        default=DEFAULT_BASE_URL,
        help=f"Base URL endpoint (default: {DEFAULT_BASE_URL})"
    )
    parser.add_argument(
        "--probe", "-p",
        action="store_true",
        help="Enable deep probing tests (uses tokens)"
    )
    parser.add_argument(
        "--filter", "-f",
        help="Filter models by name substring (e.g., 'claude', 'gpt')"
    )
    parser.add_argument(
        "--output", "-o",
        help="Save JSON results to file"
    )
    parser.add_argument(
        "--csv",
        help="Save CSV results to file"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output"
    )
    
    args = parser.parse_args()
    
    # Fetch models
    print(f"📡 Fetching models from {args.base_url}...")
    result = fetch_models(args.api_key, args.base_url)
    
    if not result["ok"]:
        print(f"❌ Error: {result['error']}")
        sys.exit(1)
    
    raw = result["data"]
    models = raw.get("data", raw) if isinstance(raw, dict) else raw
    if not isinstance(models, list):
        models = [models]
    
    print(f"✅ Found {len(models)} models\n")
    
    # Filter
    if args.filter:
        models = [m for m in models if args.filter.lower() in m.get("id", "").lower()]
        print(f"📌 Filtered to {len(models)} models\n")
    
    # Analyze each
    results = []
    for i, m in enumerate(models, 1):
        mid = m.get("id", "N/A")
        owned_by = m.get("owned_by", "")
        
        print(f"[{i}/{len(models)}] {mid}...", end="", flush=True)
        
        probe = None
        if args.probe:
            print(" [probing]", end="", flush=True)
            probe = probe_model(args.api_key, args.base_url, mid)
            time.sleep(0.5)  # Rate limit
        
        analysis = analyze(mid, owned_by, probe if probe and probe.get("ok") else None)
        
        record = {
            "model_id": mid,
            "owned_by": owned_by,
            "provider": analysis["provider"],
            "verdict": analysis["verdict"],
            "cracked_score": analysis["cracked_score"],
            "confidence": analysis["confidence"],
            "signals": analysis["signals"],
        }
        
        if probe and probe.get("ok"):
            record.update({
                "latency_ms": probe.get("avg_latency_ms"),
                "model_mismatch": probe.get("model_mismatch"),
                "reported_model": probe.get("reported_model"),
            })
        
        results.append(record)
        
        # Status emoji
        emoji = "🚨" if analysis["cracked_score"] >= 70 else "⚠️" if analysis["cracked_score"] >= 45 else "✅"
        print(f" {emoji} {analysis['verdict']} ({analysis['cracked_score']}/100)")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    by_verdict = {}
    for r in results:
        v = r["verdict"]
        by_verdict[v] = by_verdict.get(v, 0) + 1
    
    for v, count in sorted(by_verdict.items(), key=lambda x: -x[1]):
        print(f"{v}: {count}")
    
    cracked_count = sum(1 for r in results if r["cracked_score"] >= 70)
    suspicious_count = sum(1 for r in results if 45 <= r["cracked_score"] < 70)
    genuine_count = sum(1 for r in results if r["cracked_score"] < 25)
    
    print(f"\n📊 Risk Analysis:")
    print(f"   🚨 High Risk (>=70):  {cracked_count}")
    print(f"   ⚠️  Medium Risk (45-69): {suspicious_count}")
    print(f"   ✅ Low Risk (<25):     {genuine_count}")
    
    # Export
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Results saved to {args.output}")
    
    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["model_id", "provider", "verdict", "cracked_score", "confidence"]
            )
            writer.writeheader()
            for r in results:
                writer.writerow({
                    "model_id": r["model_id"],
                    "provider": r["provider"],
                    "verdict": r["verdict"],
                    "cracked_score": r["cracked_score"],
                    "confidence": r["confidence"],
                })
        print(f"📊 CSV saved to {args.csv}")
    
    # Exit code based on risk
    if cracked_count > 0:
        sys.exit(2)  # Risk detected
    elif suspicious_count > 0:
        sys.exit(1)  # Warnings
    else:
        sys.exit(0)  # All good

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️  Cancelled")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
