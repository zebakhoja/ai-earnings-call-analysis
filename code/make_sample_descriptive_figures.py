"""
Three exploratory/descriptive figures characterising the corpus, for the
Data chapter: geographic spread of covered firms, transcript volume by
year, and sector composition. Purely descriptive -- no hypothesis
testing -- so kept separate from the analyze_*.py scripts.

Run:
    python make_sample_descriptive_figures.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

SECTOR_MAPPING = "../Data/ticker_sector_mapping.csv"
VOLUME_PANEL = "../Data/ai_volume_panel.csv"

SECTOR_COLOR = "#2c6e9e"
BAR_COLOR = "#3d7ea6"


def fig_world_map():
    """Dot map: one bubble per country at its centroid, sized by sqrt(firm
    count) (linear size would make the US swamp everything else) and
    labelled with the raw count."""
    import numpy as np

    sm = pd.read_csv(SECTOR_MAPPING)
    counts = sm.groupby("country").size().reset_index(name="n_firms")
    counts = counts.sort_values("n_firms", ascending=False)
    counts["bubble_size"] = np.sqrt(counts["n_firms"]) * 4.5

    fig = go.Figure(
        data=go.Scattergeo(
            locations=counts["country"],
            locationmode="country names",
            text=counts["n_firms"].astype(str),
            mode="markers+text",
            marker=dict(
                size=counts["bubble_size"],
                color="#2c6e9e",
                opacity=0.75,
                line=dict(width=1, color="white"),
            ),
            textposition="middle center",
            textfont=dict(size=9, color="white"),
            hovertext=[f"{c}: {n} firms" for c, n in zip(counts["country"], counts["n_firms"])],
            hoverinfo="text",
        )
    )

    fig.update_layout(
        title="Geographic coverage: firms per country (n=852 total, 22 countries)",
        geo=dict(
            showframe=False,
            showcoastlines=True,
            showcountries=True,
            countrycolor="#cccccc",
            landcolor="#f2f2f2",
            projection_type="natural earth",
            bgcolor="white",
        ),
        margin=dict(l=10, r=10, t=50, b=10),
        width=1100, height=650,
    )
    fig.write_image("../Data/fig_country_coverage_map.png", scale=2)
    print("Saved ../Data/fig_country_coverage_map.png")
    print(counts.to_string(index=False))


def fig_transcript_density_by_year():
    """Bar/histogram of transcript counts by year."""
    vol = pd.read_csv(VOLUME_PANEL)
    transcripts = vol.drop_duplicates(["symbol", "year", "period"])
    by_year = transcripts.groupby("year").size()

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(by_year.index.astype(str), by_year.values, color=BAR_COLOR, width=0.65)
    for bar, val in zip(bars, by_year.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 30, str(val),
                 ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Year")
    ax.set_ylabel("Number of quarterly transcripts")
    ax.set_title(f"Transcript coverage by year (n={len(transcripts):,} total, 2018-2026)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.5, -0.02,
              "2026 is partial (through the most recent quarter available at extraction).",
              ha="center", fontsize=8, style="italic", color="#555")
    fig.tight_layout()
    fig.savefig("../Data/fig_transcript_density_by_year.png", dpi=150, bbox_inches="tight")
    print("Saved ../Data/fig_transcript_density_by_year.png")
    print(by_year)


def fig_sector_spread():
    """Horizontal bar chart of firm counts by GICS sector."""
    sm = pd.read_csv(SECTOR_MAPPING)
    by_sector = sm.groupby("gics_sector").size().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    bars = ax.barh(by_sector.index, by_sector.values, color=SECTOR_COLOR)
    for bar, val in zip(bars, by_sector.values):
        ax.text(val + 2, bar.get_y() + bar.get_height() / 2, str(val),
                 va="center", fontsize=9)

    ax.set_xlabel("Number of firms")
    ax.set_title(f"Sector composition of the corpus (n={len(sm):,} firms, GICS sectors)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig("../Data/fig_sector_spread.png", dpi=150, bbox_inches="tight")
    print("Saved ../Data/fig_sector_spread.png")
    print(by_sector.sort_values(ascending=False))


def main():
    fig_world_map()
    print()
    fig_transcript_density_by_year()
    print()
    fig_sector_spread()


if __name__ == "__main__":
    main()
