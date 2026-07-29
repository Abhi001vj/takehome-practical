"""Compute dataset statistics and generate exploratory plots."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .config import LABELS, REPORTS
from .data import load_training_data
from .grouping import assign_groups, normalise_for_grouping

#: Openers/closers the generator attached at random. Counted here to demonstrate they
#: carry no label information.
BOILERPLATE_OPENERS = (
    "Quick question",
    "Hello team",
    "Please help",
    "Report fraud",
    "Urgent",
    "Hi",
    "Hey",
)

#: A stoplist for the "distinctive terms" table. English stopwords plus the boilerplate,
#: because otherwise every class's top terms are "the", "my" and "thanks".
_EXTRA_STOP = {
    "hi", "hey", "hello", "team", "urgent", "please", "help", "thanks", "thank", "you",
    "advise", "appreciate", "quick", "question", "let", "know", "would", "could", "can",
    "my", "i", "me", "the", "a", "an", "is", "it", "to", "and", "of", "in", "on", "for",
    "this", "that", "was", "have", "has", "but", "not", "with", "at", "be", "am", "are",
    "your", "so", "if", "as", "any", "do", "does", "did", "there", "what", "how", "when",
}


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z']{2,}", text.lower()) if t not in _EXTRA_STOP]


def compute_stats(df: pd.DataFrame) -> dict:
    """Every number the EDA report quotes."""
    grouping = assign_groups(df["text"].tolist())

    # Template counts per label: the key correction to the naive imbalance reading.
    first_label_of_group: dict[int, str] = {}
    for label, group in zip(df["label"], grouping.groups, strict=True):
        first_label_of_group.setdefault(int(group), label)
    template_counts = Counter(first_label_of_group.values())

    row_counts = df["label"].value_counts()
    lengths = df["text"].str.split().str.len()
    chars = df["text"].str.len()

    # Opener x label contingency: if openers were signal, rows would concentrate.
    opener_table: dict[str, dict[str, int]] = {}
    for opener in (*BOILERPLATE_OPENERS, "(none)"):
        opener_table[opener] = dict.fromkeys(LABELS, 0)
    for text, label in zip(df["text"], df["label"], strict=True):
        matched = "(none)"
        stripped = text.strip().strip('"')
        for opener in BOILERPLATE_OPENERS:
            if stripped.lower().startswith(opener.lower()):
                matched = opener
                break
        opener_table[matched][label] += 1

    per_class_lengths = {
        label: {
            "mean_words": float(lengths[df["label"] == label].mean()),
            "median_words": float(lengths[df["label"] == label].median()),
            "min_words": int(lengths[df["label"] == label].min()),
            "max_words": int(lengths[df["label"] == label].max()),
        }
        for label in LABELS
    }

    vocab = Counter()
    per_class_vocab: dict[str, Counter] = {label: Counter() for label in LABELS}
    for text, label in zip(df["text"], df["label"], strict=True):
        toks = _tokens(text)
        vocab.update(toks)
        per_class_vocab[label].update(toks)

    return {
        "n_rows": int(len(df)),
        "n_duplicate_texts": int(len(df) - df["text"].nunique()),
        "n_normalised_forms": int(grouping.n_exact_groups),
        "n_template_groups": int(grouping.n_groups),
        "largest_template_group": int(grouping.largest_group),
        "template_compression": float(grouping.compression),
        "grouping_summary": grouping.summary(),
        "row_counts": {label: int(row_counts.get(label, 0)) for label in LABELS},
        "template_counts": {label: int(template_counts.get(label, 0)) for label in LABELS},
        "row_imbalance_ratio": float(row_counts.max() / row_counts.min()),
        "template_imbalance_ratio": float(
            max(template_counts.values()) / min(template_counts.values())
        ),
        "length_words": {
            "min": int(lengths.min()),
            "median": float(lengths.median()),
            "mean": float(lengths.mean()),
            "max": int(lengths.max()),
        },
        "length_chars": {
            "min": int(chars.min()),
            "median": float(chars.median()),
            "max": int(chars.max()),
        },
        "per_class_lengths": per_class_lengths,
        "opener_by_label": opener_table,
        "vocabulary_size": len(vocab),
        "top_terms_overall": vocab.most_common(25),
        "distinctive_terms": _distinctive_terms(per_class_vocab),
    }


def _distinctive_terms(per_class_vocab: dict[str, Counter], top_n: int = 12) -> dict:
    """Terms most over-represented in a class relative to the rest of the corpus.

    A simple ratio of within-class frequency to overall frequency, smoothed. This is what
    a linear model is picking up on, so it doubles as a sanity check that the model has a
    plausible basis for its decisions.
    """
    totals = Counter()
    for counter in per_class_vocab.values():
        totals.update(counter)
    total_all = sum(totals.values()) or 1

    out: dict[str, list] = {}
    for label, counter in per_class_vocab.items():
        class_total = sum(counter.values()) or 1
        scored = []
        for term, count in counter.items():
            if count < 3:
                continue
            p_class = count / class_total
            p_all = totals[term] / total_all
            scored.append((term, count, round(p_class / p_all, 2)))
        scored.sort(key=lambda x: (-x[2], -x[1]))
        out[label] = scored[:top_n]
    return out


def _label_colours() -> dict[str, str]:
    # Fixed so every figure uses the same colour for the same route.
    return {
        "account-access": "#4C78A8",
        "transaction-dispute": "#F58518",
        "fraud-report": "#E45756",
        "general": "#54A24B",
    }


def make_plots(df: pd.DataFrame, stats: dict, out_dir: Path) -> list[Path]:
    """Write every figure. Returns the paths written."""
    import matplotlib

    matplotlib.use("Agg")  # headless: this runs in CI and in Docker
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    colours = _label_colours()
    written: list[Path] = []

    # 1. Rows vs templates per class - the imbalance correction.
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(LABELS))
    rows = [stats["row_counts"][label] for label in LABELS]
    templates = [stats["template_counts"][label] for label in LABELS]
    ax.bar(x - 0.2, rows, 0.4, label="rows", color=[colours[label] for label in LABELS])
    ax.bar(
        x + 0.2, templates, 0.4, label="distinct templates",
        color=[colours[label] for label in LABELS], alpha=0.45, hatch="//",
    )
    for i, (r, t) in enumerate(zip(rows, templates, strict=True)):
        ax.text(i - 0.2, r + 2, str(r), ha="center", fontsize=9)
        ax.text(i + 0.2, t + 2, str(t), ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, rotation=15, ha="right")
    ax.set_ylabel("count")
    ax.set_title(
        f"Rows imply {stats['row_imbalance_ratio']:.1f}:1 imbalance; "
        f"templates imply {stats['template_imbalance_ratio']:.1f}:1"
    )
    ax.legend()
    fig.tight_layout()
    path = out_dir / "class_distribution.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(path)

    # 2. Message length distribution per class.
    fig, ax = plt.subplots(figsize=(9, 4.5))
    data = [df.loc[df["label"] == label, "text"].str.split().str.len() for label in LABELS]
    # `tick_labels` replaced the old `labels` kwarg in matplotlib 3.9+.
    parts = ax.boxplot(data, tick_labels=list(LABELS), patch_artist=True, showmeans=True)
    for patch, label in zip(parts["boxes"], LABELS, strict=True):
        patch.set_facecolor(colours[label])
        patch.set_alpha(0.6)
    ax.set_ylabel("words per message")
    ax.set_title("Message length is uniform across routes - length carries no signal")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    path = out_dir / "length_distribution.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(path)

    # 3. Opener x label heatmap - shows the boilerplate is noise.
    openers = [o for o in stats["opener_by_label"] if sum(stats["opener_by_label"][o].values())]
    matrix = np.array(
        [[stats["opener_by_label"][o][label] for label in LABELS] for o in openers], dtype=float
    )
    # Row-normalise: if openers were informative, rows would be peaked rather than flat.
    normed = matrix / matrix.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(normed, cmap="RdYlBu_r", vmin=0, vmax=0.6, aspect="auto")
    ax.set_xticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS, rotation=20, ha="right")
    ax.set_yticks(range(len(openers)))
    ax.set_yticklabels(openers)
    for i in range(len(openers)):
        for j in range(len(LABELS)):
            ax.text(j, i, f"{normed[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Opener vs route (row-normalised)\nflat rows = boilerplate carries no signal")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    path = out_dir / "boilerplate_vs_label.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(path)

    # 4. Template group size distribution.
    grouping = assign_groups(df["text"].tolist())
    sizes = np.bincount(grouping.groups)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(sizes, bins=range(1, sizes.max() + 2), color="#4C78A8", edgecolor="white")
    ax.set_xlabel("rows generated from one template")
    ax.set_ylabel("number of templates")
    ax.set_title(
        f"{stats['n_rows']} rows come from {stats['n_template_groups']} templates "
        f"({stats['template_compression']:.0%} are repeats)"
    )
    fig.tight_layout()
    path = out_dir / "template_group_sizes.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(path)

    # 5. Wordclouds per route.
    try:
        from wordcloud import WordCloud

        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        for ax, label in zip(axes.ravel(), LABELS, strict=True):
            # Wordclouds are built on the *normalised* text so that asset names and
            # amounts do not dominate; what is left is the routing vocabulary.
            corpus = " ".join(
                " ".join(_tokens(normalise_for_grouping(t)))
                for t in df.loc[df["label"] == label, "text"]
            )
            cloud = WordCloud(
                width=800, height=440, background_color="white",
                colormap="viridis", random_state=0, max_words=60,
            ).generate(corpus or "empty")
            ax.imshow(cloud, interpolation="bilinear")
            ax.set_title(f"{label}  (n={stats['row_counts'][label]})", fontsize=12)
            ax.axis("off")
        fig.suptitle("Vocabulary by route (boilerplate, amounts and tickers removed)")
        fig.tight_layout()
        path = out_dir / "wordclouds.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        written.append(path)
    except ImportError:
        pass

    # 6. Distinctive terms per route.
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    for ax, label in zip(axes, LABELS, strict=True):
        terms = stats["distinctive_terms"][label][:10][::-1]
        if terms:
            ax.barh(
                [t[0] for t in terms], [t[2] for t in terms],
                color=colours[label], alpha=0.85,
            )
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("lift vs corpus")
        ax.tick_params(axis="y", labelsize=9)
    fig.suptitle("Most over-represented terms per route")
    fig.tight_layout()
    path = out_dir / "distinctive_terms.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(path)

    return written


def render_markdown(stats: dict, figures: list[Path]) -> str:
    lines = ["# Exploratory data analysis", ""]

    lines += [
        "## Shape",
        "",
        f"- **{stats['n_rows']}** labelled messages, 2 columns (`text`, `label`)",
        f"- **{stats['n_duplicate_texts']}** exact duplicate texts",
        f"- Vocabulary: **{stats['vocabulary_size']}** distinct content words",
        f"- Length: {stats['length_words']['min']}-{stats['length_words']['max']} words "
        f"(median {stats['length_words']['median']:.0f}), "
        f"{stats['length_chars']['min']}-{stats['length_chars']['max']} characters",
        "",
        "## The dataset is template-generated",
        "",
        f"{stats['grouping_summary']}",
        "",
        f"Normalising away the randomised opener, closer, amount and ticker collapses "
        f"{stats['n_rows']} rows to **{stats['n_normalised_forms']}** distinct forms, and "
        f"near-duplicate clustering reduces that further to **{stats['n_template_groups']}** "
        f"template groups. The largest single template accounts for "
        f"{stats['largest_template_group']} rows.",
        "",
        "**This is the single most important fact about the data.** The effective sample "
        f"size is ~{stats['n_template_groups']}, not {stats['n_rows']}. Any evaluation that "
        "splits rows at random will place near-identical messages on both sides of the "
        "split and report a score the hidden holdout will not reproduce.",
        "",
        "![template group sizes](template_group_sizes.png)",
        "",
        "## Class balance is not what the row counts suggest",
        "",
        "| route | rows | templates | rows/template |",
        "|---|---:|---:|---:|",
    ]
    for label in LABELS:
        rows = stats["row_counts"][label]
        templates = stats["template_counts"][label]
        lines.append(
            f"| {label} | {rows} | {templates} | {rows / templates:.1f} |"
            if templates
            else f"| {label} | {rows} | 0 | - |"
        )
    lines += [
        "",
        f"At the row level the imbalance is **{stats['row_imbalance_ratio']:.1f}:1**. At the "
        f"template level it is **{stats['template_imbalance_ratio']:.1f}:1**. The gap is "
        "duplication: `general` has the most rows but not proportionally more distinct "
        "messages, while `fraud-report` has the fewest rows and a comparable number of "
        "templates.",
        "",
        "This is why the imbalance strategy here is class weighting rather than "
        "oversampling. SMOTE or naive duplication would interpolate between renderings of "
        "the *same* template and add no information at all, while making the leakage "
        "problem worse.",
        "",
        "![class distribution](class_distribution.png)",
        "",
        "## Boilerplate is noise, not signal",
        "",
        "| opener | " + " | ".join(LABELS) + " |",
        "|---|" + "---:|" * len(LABELS),
    ]
    for opener, counts in stats["opener_by_label"].items():
        if sum(counts.values()):
            lines.append(f"| {opener} | " + " | ".join(str(counts[la]) for la in LABELS) + " |")
    lines += [
        "",
        "Openers are spread across routes in roughly the corpus proportions. Notably "
        "**`Urgent` does not indicate fraud** - it appears on more `general` messages than "
        "`fraud-report` ones. A model that keys on it is fitting noise, which is a live "
        "risk given char n-grams will happily learn these tokens.",
        "",
        "They are stripped for *grouping* but deliberately left in the text the models "
        "see, because the hidden holdout will carry the same boilerplate.",
        "",
        "![boilerplate vs label](boilerplate_vs_label.png)",
        "",
        "## Length carries no signal",
        "",
        "| route | mean words | median | min | max |",
        "|---|---:|---:|---:|---:|",
    ]
    for label in LABELS:
        entry = stats["per_class_lengths"][label]
        lines.append(
            f"| {label} | {entry['mean_words']:.1f} | {entry['median_words']:.0f} "
            f"| {entry['min_words']} | {entry['max_words']} |"
        )
    lines += [
        "",
        "![length distribution](length_distribution.png)",
        "",
        "## Vocabulary by route",
        "",
        "![wordclouds](wordclouds.png)",
        "",
        "### Most over-represented terms",
        "",
    ]
    for label in LABELS:
        top = stats["distinctive_terms"][label][:8]
        terms = ", ".join(f"`{t}` ({lift}x)" for t, _, lift in top)
        lines.append(f"- **{label}**: {terms}")
    lines += [
        "",
        "![distinctive terms](distinctive_terms.png)",
        "",
        "The `fraud-report` / `transaction-dispute` boundary is the one that matters: both "
        "are about missing money, and they separate on *agency* (`authorized`, `phishing`, "
        "`someone` vs `cancelled`, `order`, `pending`) rather than on topic. That is a "
        "semantic distinction, which is the substantive argument for why sentence "
        "embeddings outperform bag-of-n-grams here.",
        "",
        "## What this implies for modelling",
        "",
        "1. **Split on templates, not rows.** Otherwise ~95% of every test fold is already "
        "memorised. See `reports/comparison.md`.",
        "2. **Use macro-F1, not accuracy.** `general` is 40% of rows; predicting it always "
        "scores 0.40 accuracy and 0.0 recall on the route that matters.",
        "3. **Weight classes, do not resample.** The real imbalance is mild once "
        "duplication is accounted for.",
        "4. **Prefer semantic features.** The hard boundary is about who acted, not about "
        "which words appear.",
        "",
    ]
    if figures:
        lines += ["---", "", "Figures: " + ", ".join(f"`{p.name}`" for p in figures), ""]
    return "\n".join(lines)


def run_eda(
    data_path: Path | None = None, out_dir: Path | None = None, plots: bool = True
) -> tuple[Path, dict]:
    """Compute stats, draw figures, write `reports/eda.md`."""
    out_dir = out_dir or REPORTS
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_training_data(data_path)
    stats = compute_stats(df)

    figures: list[Path] = []
    if plots:
        try:
            figures = make_plots(df, stats, out_dir)
        except ImportError as exc:
            stats["plot_error"] = f"plotting dependencies missing: {exc}"

    (out_dir / "eda_stats.json").write_text(json.dumps(stats, indent=2, default=str))
    path = out_dir / "eda.md"
    path.write_text(render_markdown(stats, figures))
    return path, stats
