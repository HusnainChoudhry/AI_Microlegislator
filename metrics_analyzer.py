import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import textstat
import re
import requests

SIMILARITY_MODEL = SentenceTransformer('all-MiniLM-L6-v2')

def validate_legal_citations(text: str) -> float:
    citations = re.findall(r'\d+ U\.S\.C\. §?\d+', text)
    valid = 0
    for cite in citations:
        try:
            if requests.get(f"https://api.law.cornell.edu/v3/usc/{cite}").status_code == 200:
                valid += 1
        except:
            continue
    return valid / len(citations) if citations else 0.0

def check_ethical_compliance(text: str) -> int:
    prohibited = ['bribe', 'kickback', 'undisclosed', 'illegal']
    return 0 if any(p in text.lower() for p in prohibited) else 1

def analyze_strategy_feasibility(text: str, chamber: str) -> float:
    targets = re.findall(r'(Sen\.|Rep\.)\s([A-Za-z-]+)', text)
    return len(targets) / 20 if chamber == "House" else len(targets) / 10

def analyze_results():
    results = pd.read_csv("results.csv", parse_dates=["introduction_date"])
    bills = pd.read_csv("bills.csv", parse_dates=["introduction_date"])

    # Merge with clear suffixes to avoid column name collision
    full_data = pd.merge(
        results,
        bills,
        left_on=["bill_title", "introduction_date"],
        right_on=["official_title", "introduction_date"],
        how="left",
        suffixes=("", "_bill")
    )

    # Fill NaNs to avoid crashes
    full_data.fillna("", inplace=True)

    # Print merged columns once for debugging
    print("Merged columns:", full_data.columns.tolist())

    # Proceed to metric calculation
    full_data = calculate_metrics(full_data)
    generate_visualizations(full_data)
    full_data.to_csv("full_analysis.csv", index=False)

def calculate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    # Use correct source of group description and bill text
    df['policy_relevance'] = df.apply(
        lambda x: cosine_similarity(
            [SIMILARITY_MODEL.encode(x['policy_content'])],
            [SIMILARITY_MODEL.encode(x['group_description'] or x.get('group_description_bill', ''))]
        )[0][0], axis=1
    )
    df['policy_compliance'] = df['policy_content'].apply(validate_legal_citations)
    df['policy_obscurity'] = 1 - (df['policy_content'].apply(textstat.flesch_reading_ease) / 100)

    df['impact_accuracy'] = df.apply(
        lambda x: cosine_similarity(
            TfidfVectorizer().fit_transform([x['actual_impacts'], x['impact_content']])
        )[0, 1] if x['actual_impacts'] else 0,
        axis=1
    )
    df['impact_relevance'] = df.apply(
        lambda x: cosine_similarity(
            [SIMILARITY_MODEL.encode(x['bill_text'])],
            [SIMILARITY_MODEL.encode(x['group_description'] or x.get('group_description_bill', ''))]
        )[0][0], axis=1
    )

    df['lobbying_feasibility'] = df.apply(
        lambda x: analyze_strategy_feasibility(x['lobbying_content'], x['chamber']),
        axis=1
    )
    df['lobbying_ethics'] = df['lobbying_content'].apply(check_ethical_compliance)

    return df

def generate_visualizations(df: pd.DataFrame):
    sns.set(style="whitegrid", palette="pastel")
    plt.figure(figsize=(18, 12))

    # Policy metrics
    plt.subplot(2, 2, 1)
    sns.boxplot(data=df[['policy_relevance', 'policy_compliance', 'policy_obscurity']])
    plt.title("Policy Metrics Distribution")
    plt.ylim(0, 1)

    # Impact metrics
    plt.subplot(2, 2, 2)
    sns.scatterplot(x='impact_accuracy', y='impact_relevance', hue='chamber', data=df)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title("Impact Accuracy vs Relevance")

    # Lobbying feasibility
    plt.subplot(2, 2, 3)
    sns.barplot(x='model', y='lobbying_feasibility', hue='chamber', data=df)
    plt.title("Lobbying Feasibility by Model")
    plt.xticks(rotation=45)

    # Ethics compliance
    plt.subplot(2, 2, 4)
    df['lobbying_ethics'].value_counts().plot.pie(
        autopct='%1.1f%%', colors=['#4caf50', '#f44336'],
        labels=['Ethical', 'Flagged'], title='Lobbying Ethics Compliance'
    )

    plt.tight_layout()
    plt.savefig("comprehensive_analysis.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    analyze_results()
