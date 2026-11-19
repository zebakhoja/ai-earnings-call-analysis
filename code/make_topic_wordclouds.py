"""
Word clouds for the 6 genuinely cross-sector AI use-case topics identified
in AI Topic Model.docx (not the single-vendor self-description topics --
see that doc for why those were excluded).

Word importance per topic is recomputed directly from that topic's own
snippets via TF-IDF (comparing each topic's vocabulary against the OTHER
topics' combined vocabulary as the background corpus) -- this reuses
ai_documents_with_topic.csv rather than needing the original BERTopic model
object reloaded/retrained, and gives a topic-specific term weighting
(distinctive words for THIS topic, not just frequent words overall) matching
the same design principle as BERTopic's own class-based TF-IDF.

Run:
    python make_topic_wordclouds.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud

CROSS_SECTOR_TOPICS = {
    38: "Customer service chatbots",
    39: "Supply chain",
    6: "Power demand for AI infrastructure",
    50: "ChatGPT",
    56: "DeepSeek",
    15: "Data types and structure",
}

EXTRA_STOPWORDS = [
    "ai", "artificial", "intelligence", "using", "use", "used", "uses",
    "think", "know", "going", "just", "like", "really", "actually",
    "quarter", "year", "question", "yes", "thanks", "thank", "ve", "don",
    "ll", "re", "way", "lot", "right", "things", "thing",
]


def build_topic_wordclouds(df: pd.DataFrame, topic_ids: dict):
    docs = []
    labels = []
    for tid in topic_ids:
        text = " ".join(df.loc[df["topic_id"] == tid, "snippet_text"].astype(str))
        docs.append(text)
        labels.append(tid)

    vectorizer = TfidfVectorizer(
        stop_words=list(TfidfVectorizer(stop_words="english").get_stop_words()) + EXTRA_STOPWORDS,
        ngram_range=(1, 2), min_df=1, max_df=0.9,
    )
    tfidf = vectorizer.fit_transform(docs)
    vocab = vectorizer.get_feature_names_out()

    clouds = {}
    for i, tid in enumerate(labels):
        row = tfidf[i].toarray().flatten()
        freqs = {vocab[j]: row[j] for j in row.nonzero()[0]}
        clouds[tid] = freqs
    return clouds


def main():
    df = pd.read_csv("../Data/ai_documents_with_topic.csv")
    clouds = build_topic_wordclouds(df, CROSS_SECTOR_TOPICS)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    for ax, (tid, label) in zip(axes.flat, CROSS_SECTOR_TOPICS.items()):
        wc = WordCloud(
            width=800, height=500, background_color="white",
            colormap="viridis", max_words=60, prefer_horizontal=0.9,
        ).generate_from_frequencies(clouds[tid])
        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(f"#{tid}: {label}", fontsize=13)
        ax.axis("off")

    fig.suptitle("Cross-sector AI use-case topics: word clouds", fontsize=16)
    fig.tight_layout()
    fig.savefig("../Data/topic_wordclouds_cross_sector.png", dpi=150)
    print("Saved grid to ../Data/topic_wordclouds_cross_sector.png")

    for tid, label in CROSS_SECTOR_TOPICS.items():
        wc = WordCloud(
            width=1000, height=600, background_color="white",
            colormap="viridis", max_words=60, prefer_horizontal=0.9,
        ).generate_from_frequencies(clouds[tid])
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.imshow(wc, interpolation="bilinear")
        ax2.set_title(f"#{tid}: {label}", fontsize=14)
        ax2.axis("off")
        fig2.tight_layout()
        safe_name = label.lower().replace(" ", "_").replace("/", "_")
        fig2.savefig(f"../Data/topic_wordcloud_{tid}_{safe_name}.png", dpi=150)
        plt.close(fig2)
    print("Saved individual word clouds to ../Data/topic_wordcloud_<id>_<name>.png")


if __name__ == "__main__":
    main()
