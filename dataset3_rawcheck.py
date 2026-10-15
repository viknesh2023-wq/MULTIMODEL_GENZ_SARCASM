from pathlib import Path

ROOT = Path(r"D:\NLP_PROJECT")
DRAVIDIAN_DIR = ROOT / "DravidianCodeMix"

files = [
    "kannada_sentiment_train.csv",
    "kannada_sentiment_dev.csv",
    "kannada_sentiment_test.csv",

    "mal_full_sentiment_train.csv",
    "mal_full_sentiment_dev.csv",
    "mal_full_sentiment_test.csv",

    "tamil_sentiment_full_train.csv",
    "tamil_sentiment_full_dev.csv",
    "tamil_sentiment_full_test.csv",
]

print("=" * 100)
print("DATASET 3 — RAW DRAVIDIAN FILE CHECK")
print("=" * 100)

for filename in files:

    path = DRAVIDIAN_DIR / filename

    print("\n")
    print("=" * 100)
    print("FILE:", filename)
    print("PATH:", path)
    print("=" * 100)

    if not path.exists():
        print("FILE NOT FOUND")
        continue

    with open(
        path,
        "r",
        encoding="utf-8-sig",
        errors="replace"
    ) as f:

        for i in range(5):

            line = f.readline()

            if not line:
                break

            print(
                f"RAW LINE {i + 1}:"
            )

            print(
                repr(line)
            )

print("\n")
print("=" * 100)
print("CHECK COMPLETE")
print("=" * 100)
