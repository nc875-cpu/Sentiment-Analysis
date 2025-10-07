import praw
import csv
import os
import time
from datetime import datetime

# initialize Reddit (replace values or use env vars)
reddit = praw.Reddit(
    client_id="nA6TJhKoxzg51kCwTlucSA",
    client_secret="zhpqYl0w-RCwxt6hMSpf9khQsmZDfg",
    user_agent="iphone17-scraper"
)

queries = [
    '"iphone 17 pro max"',
    '"iphone 17 pro"',
    '"iphone 17"',
    '"iPhone 17 Pro Max"',
    '"iPhone 17"'
]
subreddits = ["all", "apple", "iphone", "gadgets", "technology"]

# How many total unique posts you want
target_rows = 1000

# Output path: ../data/ (relative to this script in /code)
out_path = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(out_path, exist_ok=True)
out_file = os.path.join(out_path, "iphone17_reddit.csv")

fields = ["id", "title", "author", "created_utc", "score", "upvote_ratio", "subreddit", "permalink", "url"]

def row_from_submission(s):
    return {
        "id": s.id,
        "title": s.title,
        "author": str(s.author) if s.author else None,
        "created_utc": datetime.utcfromtimestamp(s.created_utc).isoformat(),
        "score": s.score,
        "upvote_ratio": getattr(s, "upvote_ratio", None),
        "subreddit": str(s.subreddit),
        "permalink": f"https://www.reddit.com{s.permalink}",
        "url": s.url
    }

seen = set()
written = 0

with open(out_file, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()

    # Loop over multiple (subreddit, query) combos to blow past single-query limits
    for subr in subreddits:
        for q in queries:
            # Reddit search often caps ~1000 per query. Ask for a bit more and dedupe.
            # 'new' + 'all' maximizes recall.
            gen = reddit.subreddit(subr).search(q, sort="new", time_filter="all", limit=1500)
            for s in gen:
                if s.id in seen:
                    continue
                seen.add(s.id)
                w.writerow(row_from_submission(s))
                written += 1

                if written % 100 == 0:
                    print(f"Collected {written} unique posts so far...")

                # light backoff to be polite
                time.sleep(0.15)

                if written >= target_rows:
                    break
            if written >= target_rows:
                break
        if written >= target_rows:
            break

print(f"\n✅ Done. Wrote {written} unique rows to {out_file}")