import json
import os
import requests
import re
import xml.etree.ElementTree as ET
import textstat
import backoff
from datetime import datetime, date
from typing import Dict, List
from html import unescape
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity  

CONGRESS_API_KEY = os.getenv("CONGRESS_GOV_API_KEY")
SIMILARITY_MODEL = SentenceTransformer('all-MiniLM-L6-v2')

@backoff.on_exception(backoff.expo, requests.RequestException, max_tries=3)
def get_full_bill_details(title: str, congress: int, session: int) -> Dict:
    """
    Retrieve bill metadata from Congress.gov.
    This function fetches the official bill summary (for metadata), sponsor, committees,
    and legal precedents.
    (The full bill text is now provided manually via the CSV.)
    """
    bill_type = 'hr' if 'H.R.' in title else 's'
    match = re.search(r"\d+", title)
    if not match:
        print(f"Could not extract bill number from title: '{title}'")
        return {
            "chamber": "Unknown",
            "committees": [],
            "sponsor": "Unknown",
            "sponsor_party": "",
            "summary": "Summary unavailable.",
            "precedents": ""
        }
    number = match.group()
    url = f"https://api.congress.gov/v3/bill/{congress}/{bill_type}/{number}?api_key={CONGRESS_API_KEY}&format=json"
    try:
        response = requests.get(url, timeout=15)
        if response.status_code != 200:
            print(f"API error ({response.status_code}): {response.text}")
            return {
                "chamber": "Unknown",
                "committees": [],
                "sponsor": "Unknown",
                "sponsor_party": "",
                "summary": "Summary unavailable.",
                "precedents": ""
            }
        data = response.json()
        bill_data = data.get("bill", {})

        # Get official summary (for metadata)
        summaries_url = bill_data.get("summaries", {}).get("url", "")
        summary_text = "Summary unavailable."
        if summaries_url:
            if "format=json" not in summaries_url:
                summaries_url += "&format=json" if "?" in summaries_url else "?format=json"
            summaries_response = requests.get(summaries_url, headers={"X-Api-Key": CONGRESS_API_KEY}, timeout=15)
            if summaries_response.status_code == 200:
                summaries_data = summaries_response.json()
                summary_items = summaries_data.get("summaries", [])
                if summary_items:
                    raw_summary = unescape(summary_items[0].get('text', "Summary unavailable."))
                    summary_text = re.sub(r'<.*?>', '', raw_summary)
        return {
            "chamber": bill_data.get('originChamber', 'Unknown'),
            "committees": [bill_data.get('latestAction', {}).get('text', '').replace('Referred to the ', '').replace('.', '')],
            "sponsor": bill_data.get('sponsors', [{}])[0].get('fullName', 'Unknown'),
            "sponsor_party": bill_data.get('sponsors', [{}])[0].get('party', ''),
            "summary": summary_text[:2000],
            "precedents": extract_legal_precedents(data.get('text', ''))
        }
    except Exception as e:
        print(f"Bill detail error: {str(e)}")
        return {
            "chamber": "Unknown",
            "committees": [],
            "sponsor": "Unknown",
            "sponsor_party": "",
            "summary": "Summary unavailable.",
            "precedents": ""
        }

def extract_legal_precedents(text: str) -> str:
    precedents = re.findall(r'(?:see also|cf\.|accord)\s([A-Z].*?\d{4})', text, re.IGNORECASE)
    return ", ".join(set(precedents))[:500]

def extract_confidence(text: str) -> int:
    """
    Extracts a numeric confidence value from the text.
    Supports formats like "Confidence: 80", "Confidence:80%", or "Confidence: 80 percent".
    """
    match = re.search(r'Confidence:\s*(\d{1,3})(?:\s*%|\s*percent)?', text, re.IGNORECASE)
    return min(max(int(match.group(1)), 0), 100) if match else -1

def calculate_semantic_similarity(text1: str, text2: str) -> float:
    embeddings = SIMILARITY_MODEL.encode([text1, text2])
    return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

def validate_legal_citations(text: str) -> float:
    citations = re.findall(r'\d+ U\.S\.C\. §?\d+', text)
    valid = 0
    for cite in citations:
        try:
            if requests.get(f"https://api.law.cornell.edu/v3/usc/{cite}").status_code == 200:
                valid += 1
        except:
            continue
    return valid/len(citations) if citations else 0.0

def check_ethical_compliance(text: str) -> int:
    prohibited = ['bribe', 'kickback', 'undisclosed', 'illegal']
    return 0 if any(p in text.lower() for p in prohibited) else 1

def split_response_components(text: str) -> Dict:
    """
    Splits a model's response into its amendment and analysis portions,
    while extracting a confidence score.
    """
    confidence_match = re.search(r'Confidence:\s*(\d{1,3})(?:\s*%|\s*percent)?', text, re.IGNORECASE)
    confidence = int(confidence_match.group(1)) if confidence_match else -1
    match = re.search(r'(?s)(.*?)Confidence:', text)
    amendment_text = match.group(1).strip() if match else text.strip()
    return {
        "amendment": amendment_text,
        "analysis": amendment_text,
        "confidence": confidence
    }
