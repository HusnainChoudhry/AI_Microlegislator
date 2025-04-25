# utils.py

import os
import re
import requests
import backoff
from typing import Dict
from html import unescape
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

# smaller model? keep same if you wish
SIMILARITY_MODEL = SentenceTransformer('all-MiniLM-L6-v2')

CONGRESS_API_KEY = os.getenv("CONGRESS_GOV_API_KEY")

def extract_legal_precedents(text: str) -> str:
    matches = re.findall(
        r'(?:see also|cf\.|accord)\s+([A-Z].*?\d{4})',
        text, flags=re.IGNORECASE
    )
    return ", ".join(dict.fromkeys(matches))[:500]

@backoff.on_exception(backoff.expo, requests.RequestException, max_tries=3)
def get_full_bill_details(title: str, congress: int, session: int) -> Dict:
    bill_type = 'hr' if 'H.R.' in title else 's'
    m = re.search(r'\d+', title)
    if not m:
        return {
            "chamber": "Unknown",
            "committees": [],
            "sponsor": "Unknown",
            "sponsor_party": "",
            "summary": "Summary unavailable.",
            "precedents": ""
        }
    num = m.group()
    url = (
        f"https://api.congress.gov/v3/bill/{congress}/{bill_type}/{num}"
        f"?api_key={CONGRESS_API_KEY}&format=json"
    )
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json().get("bill", {})

        # Summary
        summary = "Summary unavailable."
        sums_url = data.get("summaries", {}).get("url", "")
        if sums_url:
            if "format=json" not in sums_url:
                sums_url += "&format=json" if "?" in sums_url else "?format=json"
            s2 = requests.get(sums_url, timeout=15)
            if s2.ok:
                items = s2.json().get("summaries", [])
                if items:
                    raw = unescape(items[0].get("text", ""))
                    summary = re.sub(r'<.*?>', '', raw)

        latest = data.get("latestAction", {}).get("text", "")
        committee = latest.replace("Referred to the ", "").rstrip(".")
        sp = data.get("sponsors", [{}])[0]
        sponsor = sp.get("fullName", "Unknown")
        party   = sp.get("party", "")
        precedents = extract_legal_precedents(data.get("text", ""))

        return {
            "chamber": data.get("originChamber", "Unknown"),
            "committees": [committee] if committee else [],
            "sponsor": sponsor,
            "sponsor_party": party,
            "summary": summary[:2000],
            "precedents": precedents
        }
    except:
        return {
            "chamber": "Unknown",
            "committees": [],
            "sponsor": "Unknown",
            "sponsor_party": "",
            "summary": "Summary unavailable.",
            "precedents": ""
        }

def extract_confidence(text: str) -> int:
    m = re.search(r'Confidence:\s*(\d{1,3})(?:\s*%|\s*percent)?', text, re.IGNORECASE)
    return max(0, min(int(m.group(1)), 100)) if m else -1

def split_response_components(text: str) -> Dict:
    conf = extract_confidence(text)
    body = re.split(r'Confidence\s*:', text, flags=re.IGNORECASE)[0]
    parts = re.split(r'Rationale\s*:\s*', body, flags=re.IGNORECASE)
    return {
        "amendment": parts[0].strip(),
        "analysis": parts[1].strip() if len(parts) > 1 else "",
        "confidence": conf
    }

# simple in‐process cache to avoid repeat Cornell LII hits
_CITATION_CACHE: Dict[str, bool] = {}

def validate_legal_citations(text: str) -> float:
    """
    0.0 if no citations or none valid, up to 1.0 if all extracted citations
    return HTTP 200 from Cornell LII.
    """
    pat = re.compile(r'(\d+)\s+U\.S\.C\.\s*§?\s*([\w\-\(\)]+)', flags=re.IGNORECASE)
    matches = pat.findall(text)
    if not matches:
        return 0.0

    valid = 0
    for title, sec in matches:
        key = f"{title}-{sec}"
        if key in _CITATION_CACHE:
            if _CITATION_CACHE[key]:
                valid += 1
            continue

        url = f"https://api.law.cornell.edu/v3/usc/{title}/{sec}"
        try:
            resp = requests.head(url, timeout=2)
            ok = resp.status_code == 200
        except:
            ok = False

        _CITATION_CACHE[key] = ok
        if ok:
            valid += 1

    return valid / len(matches)
