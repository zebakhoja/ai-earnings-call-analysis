"""
Exploratory topic model over the AI-flagged snippet corpus: what specific
AI topics/use-cases are actually being discussed (customer service
chatbots, supply chain optimization, coding assistants, fraud detection,
etc.), as opposed to the exploratory-vs-operational SPECIFICITY axis
(Code/specificity_pipeline/) or the coarse general/infrastructure/
predictive/generative/autonomous CATEGORY taxonomy (ai_terms_dictionary.py)
already in place. This is a genuinely different cut on the same corpus:
topical content, not language register or technology-type.

Method: BERTopic (sentence embeddings -> UMAP dimensionality reduction ->
HDBSCAN clustering -> class-based TF-IDF for topic keywords), using the
same all-MiniLM-L6-v2 embedding model as the specificity classifier for
consistency. Reuses ai_documents.csv's wide (±1-sentence) snippet_text,
since topic modeling benefits from more surrounding context, same
rationale as the specificity classifier's window choice.

Outputs:
    - topic_model_topics.csv: one row per discovered topic, its top
      keywords, document count, and a representative example snippet.
    - ai_documents_with_topic.csv: every snippet with its assigned topic_id.
    - topic_trend.png: prevalence over time of the top N topics.

Run:
    python run_topic_model.py --sample 30000   # optional subsample for speed
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

EXTRA_STOPWORDS = [
    "ai", "artificial", "intelligence", "using", "use", "used", "uses",
    "think", "know", "going", "just", "like", "really", "actually",
    "quarter", "year", "question", "yes", "thanks", "thank",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="../Data/ai_documents.csv")
    ap.add_argument("--output-topics", default="../Data/topic_model_topics.csv")
    ap.add_argument("--output-docs", default="../Data/ai_documents_with_topic.csv")
    ap.add_argument("--plot", default="../Data/topic_trend.png")
    ap.add_argument("--sample", type=int, default=None, help="optional random subsample for speed")
    ap.add_argument("--top-n-plot", type=int, default=8)
    ap.add_argument("--min-topic-size", type=int, default=150)
    args = ap.parse_args()

    df = pd.read_csv(args.input, parse_dates=["date"])
    if args.sample:
        df = df.sample(n=min(args.sample, len(df)), random_state=42).reset_index(drop=True)
    print(f"Running topic model on {len(df)} snippets")

    docs = df["snippet_text"].fillna("").tolist()

    print(f"Embedding with {EMBED_MODEL_NAME}...")
    embedder = SentenceTransformer(EMBED_MODEL_NAME)
    embeddings = embedder.encode(docs, show_progress_bar=True, batch_size=256)

    vectorizer = CountVectorizer(
        stop_words=list(CountVectorizer(stop_words="english").get_stop_words()) + EXTRA_STOPWORDS,
        ngram_range=(1, 2), min_df=1,
    )

    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0,
                       metric="cosine", random_state=42, n_jobs=1)
    hdbscan_model = HDBSCAN(min_cluster_size=args.min_topic_size,
                            metric="euclidean", cluster_selection_method="eom",
                            prediction_data=True)

    topic_model = BERTopic(
        embedding_model=embedder,
        vectorizer_model=vectorizer,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        min_topic_size=args.min_topic_size,
        calculate_probabilities=False,
        verbose=True,
    )
    topics, _ = topic_model.fit_transform(docs, embeddings)

    df["topic_id"] = topics
    df.to_csv(args.output_docs, index=False)

    topic_info = topic_model.get_topic_info()
    print(f"\nDiscovered {len(topic_info) - 1} topics (plus outlier/-1 bucket)")
    print(topic_info.head(30).to_string(index=False))

    rows = []
    for _, row in topic_info.iterrows():
        tid = row["Topic"]
        keywords = ", ".join([w for w, _ in topic_model.get_topic(tid)][:10]) if tid != -1 else "(outlier/no topic)"
        example = df.loc[df["topic_id"] == tid, "snippet_text"].iloc[0] if (df["topic_id"] == tid).any() else ""
        rows.append({
            "topic_id": tid, "count": row["Count"], "keywords": keywords,
            "example_snippet": example[:300],
        })
    pd.DataFrame(rows).to_csv(args.output_topics, index=False)
    print(f"\nWrote topic summary to {args.output_topics}")

    df["calendar_quarter"] = df["date"].dt.to_period("Q")
    top_topics = topic_info[topic_info["Topic"] != -1].nlargest(args.top_n_plot, "Count")["Topic"].tolist()

    fig, ax = plt.subplots(figsize=(11, 6))
    for tid in top_topics:
        sub = df[df["topic_id"] == tid]
        counts = sub.groupby("calendar_quarter").size()
        totals = df.groupby("calendar_quarter").size()
        share = (counts / totals).reindex(totals.index).fillna(0)
        label = ", ".join([w for w, _ in topic_model.get_topic(tid)][:3])
        ax.plot(share.index.to_timestamp(), share.values, marker="o", markersize=3, label=f"#{tid}: {label}")

    ax.set_xlabel("Calendar quarter")
    ax.set_ylabel("Share of all AI-flagged snippets")
    ax.set_title(f"Top {args.top_n_plot} AI discussion topics over time")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(args.plot, dpi=150)
    print(f"Saved plot to {args.plot}")


if __name__ == "__main__":
    main()
