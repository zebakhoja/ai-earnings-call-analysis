"""
02_annotate.py

Lightweight terminal annotation tool. Resumable -- if you stop partway through,
re-running picks up where you left off (skips rows that already have a label).

Reads the annotation guide (00_annotation_guide.md) into memory so you can
reprint it with 'g' at any point during labeling.

Usage:
    /Users/zebakhoja/miniconda3/envs/tf-env/bin/python 02_annotate.py annotation_train.csv
    /Users/zebakhoja/miniconda3/envs/tf-env/bin/python 02_annotate.py annotation_holdout.csv
"""

import sys
import pandas as pd

TEXT_COL = "text"
LABEL_MAP = {"0": 0, "1": 1, "2": 2}

GUIDE_SUMMARY = """
  0 = exploratory/aspirational   (no deployment detail, forward-looking intent only)
  1 = mixed                      (names an initiative, no quantified outcome/deployment)
  2 = operationally specific     (named tool/system + deployment status or quantified outcome)

  Commands: 0/1/2 = label, s = skip, b = back one, g = show full guide, q = save & quit
"""


def load_guide():
    try:
        with open("00_annotation_guide.md") as f:
            return f.read()
    except FileNotFoundError:
        return "(guide file not found -- see conversation notes)"


def main():
    if len(sys.argv) != 2:
        print("Usage: python 02_annotate.py <path_to_csv>")
        sys.exit(1)

    path = sys.argv[1]
    df = pd.read_csv(path)
    guide_text = load_guide()

    unlabeled_idx = df.index[df["label"].isna() | (df["label"] == "")].tolist()
    if not unlabeled_idx:
        print("All rows already labeled.")
        return

    print(f"{len(unlabeled_idx)} snippets remaining out of {len(df)}.")
    print(GUIDE_SUMMARY)

    i = 0
    history = []
    while i < len(unlabeled_idx):
        idx = unlabeled_idx[i]
        row = df.loc[idx]
        print("\n" + "=" * 80)
        print(f"[{i+1}/{len(unlabeled_idx)}]  ticker={row.get('ticker','?')}  "
              f"quarter={row.get('quarter','?')}  section={row.get('section','?')}")
        print("-" * 80)
        print(row[TEXT_COL])
        print("-" * 80)

        choice = input("Label (0/1/2/s/b/g/q): ").strip().lower()

        if choice == "q":
            df.to_csv(path, index=False)
            print(f"Saved progress to {path}. {len(unlabeled_idx) - i} remaining.")
            return
        elif choice == "g":
            print(guide_text)
            continue
        elif choice == "s":
            i += 1
            continue
        elif choice == "b":
            if history:
                i = history.pop()
            continue
        elif choice in LABEL_MAP:
            df.loc[idx, "label"] = LABEL_MAP[choice]
            note = input("  Note (optional, enter to skip): ").strip()
            if note:
                df.loc[idx, "annotator_notes"] = note
            history.append(i)
            i += 1
            if i % 20 == 0:
                df.to_csv(path, index=False)
                print(f"  [autosaved at {i} labeled]")
        else:
            print("  Not a recognized command, try again.")

    df.to_csv(path, index=False)
    print(f"\nDone. All labels saved to {path}.")


if __name__ == "__main__":
    main()
