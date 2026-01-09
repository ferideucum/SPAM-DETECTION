import os
import re
import json
import math
import time
import random
import hashlib
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
from urllib.parse import urlparse, parse_qs

import pandas as pd
from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_RECENT

try:
    from youtube_comment_downloader import SORT_BY_POPULAR
    HAS_POPULAR = True
except Exception:
    SORT_BY_POPULAR = None
    HAS_POPULAR = False


# =========================
# CONFIG (sadece spam)
# =========================
TARGET_SPAM = 1500

# Buffer KAPALI -> tam 1500 topla
SPAM_BUFFER = TARGET_SPAM

EN_ONLY_SPAM = True  # spamda İngilizce-like filtre (gevşek)

# URL-only spam max (%8)
URL_ONLY_CAP = int(math.ceil(TARGET_SPAM * 0.08))  # 1500 -> 120

# Scan caps (hız için)
FAST_RECENT_CAP_SPAM = 3500
FAST_POPULAR_CAP_SPAM = 2000

# Deep caps (eksik kalırsa)
DEEP_CAP = 20000
DO_DEEP_SPAM = True

# Retry / sleep
RETRY_PER_VIDEO = 1
SLEEP_BETWEEN = (0.03, 0.09)
SLEEP_RETRY = (0.25, 0.60)

# Resume / stop
CHECKPOINT_PATH = "checkpoint_state_spam.json"
STOP_FILE = "STOP.txt"

# Outputs
OUT_SPAM_DATASET = "spam_dataset_en_1500.csv"
OUT_SPAM_ONLY = "spam_only_en_1500.csv"
OUT_COUNTS = "per_video_spam_counts.csv"

# ✅ Spam dedupe gevşetme (spam çok tekrar eder)
DEDUP_SPAM_BY_TEXT = False   # True yaparsan spamda aynı metni tamamen engeller
SPAM_TEXT_MAX = 3            # aynı spam comment text en fazla kaç kez alınsın


# =========================
# URL utils
# =========================
def extract_video_id(url: str) -> Optional[str]:
    url = (url or "").strip()
    if not url:
        return None

    m = re.search(r"youtu\.be/([^?&/]+)", url)
    if m:
        return m.group(1)

    m = re.search(r"/shorts/([^?&/]+)", url)
    if m:
        return m.group(1)

    try:
        parsed = urlparse(url)
        qs = parse_qs(parsed.query)
        if "v" in qs and qs["v"]:
            return qs["v"][0]
    except Exception:
        return None

    return None


def normalize_to_watch(url: str) -> Optional[str]:
    vid = extract_video_id(url)
    return f"https://www.youtube.com/watch?v={vid}" if vid else None


def read_urls_any(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    raw = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = (line or "").strip()
            if not line or line.startswith("#"):
                continue
            norm = normalize_to_watch(line)
            if norm:
                raw.append(norm)

    # dedupe by video id
    seen = set()
    out = []
    for u in raw:
        vid = extract_video_id(u)
        if vid and vid not in seen:
            seen.add(vid)
            out.append(u)
    return out


# =========================
# Text utils
# =========================
WS_RE = re.compile(r"\s+")
ASCII_LETTER_RE = re.compile(r"[A-Za-z]")

def clean_text(t: str) -> str:
    return WS_RE.sub(" ", (t or "")).strip()

def is_english_like_spam(text: str) -> bool:
    if not text:
        return False
    ascii_ratio = sum(1 for ch in text if ord(ch) < 128) / max(1, len(text))
    # spamda gevşek
    return ascii_ratio >= 0.70 and bool(ASCII_LETTER_RE.search(text))


# =========================
# SPAM FILTER
# =========================
URL_RE = re.compile(r"(https?://|www\.)\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", re.IGNORECASE)
PHONE_RE = re.compile(r"(\+?\d[\d \-\(\)]{7,}\d)")
SHORTENER_RE = re.compile(r"\b(bit\.ly|cutt\.ly|tinyurl\.com|t\.co|goo\.gl|ow\.ly|rb\.gy|is\.gd)\b", re.IGNORECASE)

CONTACT_LINK_RE = re.compile(
    r"\b(t\.me/|telegram\.me/|telegram\.dog/|wa\.me/|chat\.whatsapp\.com/|discord\.gg/|snapchat\.com/|kik\.me/)\S+",
    re.IGNORECASE
)

PAYPAL_RE = re.compile(r"\bpaypal\b", re.IGNORECASE)
BIO_ABOUT_RE = re.compile(r"\b(bio|about( section| page)?)\b", re.IGNORECASE)

CTA_RE = re.compile(r"\b(dm|dms|inbox|message|text|contact|follow|add me|reach me|hit me up)\b", re.IGNORECASE)
PLATFORM_RE = re.compile(r"\b(instagram|insta|ig|telegram|whatsapp|discord|snapchat|kik)\b", re.IGNORECASE)

HANDLE_RE = re.compile(r"@[A-Za-z0-9_\.]{3,}", re.IGNORECASE)
USERNAME_PHRASE_RE = re.compile(r"\b(my (user(name)?|ig|insta|telegram|whatsapp) is|username\s*[:\-])\b", re.IGNORECASE)

SELF_PROMO_PATTERNS = [
    r"\bcheck (out )?my (channel|video)\b",
    r"\bvisit my (channel|page)\b",
    r"\bgo to my channel\b",
    r"\bwatch my (new )?video\b",
    r"\bsubscribe to my channel\b"
]
SELF_PROMO_RE = [re.compile(p, re.IGNORECASE) for p in SELF_PROMO_PATTERNS]

GIVEAWAY_KW = [
    "giveaway", "winner", "congratulations", "congrats", "claim", "prize", "reward",
    "drawing", "raffle", "enter", "picked", "selected"
]
MONEY_SCAM_KW = [
    "work from home", "guaranteed profit", "100% legit", "made me rich", "easy money",
    "double your money", "investment", "crypto signals", "signal group", "forex signals",
    "binary options"
]
STORE_KW = [
    "buy now", "official store", "promo code", "discount code", "limited offer", "sale"
]

IMPERSONATION_RE = re.compile(
    r"\b(i am|i'm)\s+(mrbeast|elon musk|youtube team|support team)\b|\bofficial account\b",
    re.IGNORECASE
)

def contains_any(tl: str, kws: List[str]) -> bool:
    return any(k in tl for k in kws)

def count_urls(text: str) -> int:
    return len(URL_RE.findall(text or ""))

def is_url_only_like(text: str) -> bool:
    t = clean_text(text)
    if not t or not URL_RE.search(t):
        return False
    rest = URL_RE.sub("", t).strip()
    alnum = sum(ch.isalnum() for ch in rest)
    return alnum <= 6

def repeated_chars(text: str) -> bool:
    return bool(re.search(r"(.)\1{6,}", text))

def is_spam_with_reason(text: str) -> Tuple[bool, str]:
    t = clean_text(text)
    if not t:
        return False, "empty"

    tl = t.lower()
    url_n = count_urls(t)
    has_url = bool(URL_RE.search(t))
    has_short = bool(SHORTENER_RE.search(t))
    has_contact_link = bool(CONTACT_LINK_RE.search(t))

    if has_contact_link:
        return True, "contact_link"
    if EMAIL_RE.search(t) or PHONE_RE.search(t):
        return True, "contact_info"
    if any(rx.search(t) for rx in SELF_PROMO_RE):
        return True, "self_promo"
    if IMPERSONATION_RE.search(t):
        return True, "impersonation"

    if is_url_only_like(t):
        return True, "url_only"
    if url_n >= 2:
        return True, "multi_url"

    if PAYPAL_RE.search(t) and (BIO_ABOUT_RE.search(t) or "bio" in tl or "about" in tl):
        if contains_any(tl, GIVEAWAY_KW) or "enter" in tl or "drawing" in tl:
            return True, "paypal_giveaway"

    has_platform = bool(PLATFORM_RE.search(t))
    has_handle = bool(HANDLE_RE.search(t))
    has_username_phrase = bool(USERNAME_PHRASE_RE.search(t))

    if CTA_RE.search(t) and (has_platform or has_handle or has_username_phrase or has_url):
        return True, "contact_cta"

    if (has_url or has_short) and (contains_any(tl, GIVEAWAY_KW) or contains_any(tl, MONEY_SCAM_KW) or contains_any(tl, STORE_KW)):
        return True, "scam_kw_with_link"

    if len(t) < 10:
        if has_url or has_short or has_platform:
            return True, "short_linkish"
        return False, "too_short"

    if repeated_chars(t) and (has_url or has_short):
        return True, "repeated_chars_link"

    return False, "no_strong_signal"


# =========================
# Resume / stop
# =========================
def save_checkpoint(state: Dict):
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False)

def load_checkpoint() -> Optional[Dict]:
    if not os.path.exists(CHECKPOINT_PATH):
        return None
    try:
        with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def should_stop() -> bool:
    return os.path.exists(STOP_FILE)


# =========================
# Collector (sadece spam)
# =========================
def make_row(c: Dict, url: str, reason: str = "") -> Dict:
    return {
        "comment": clean_text(c.get("text")),
        "label": 1,
        "spam_reason": reason,
        "cid": c.get("cid"),
        "author": c.get("author"),
        "time": c.get("time"),
        "votes": c.get("votes"),
        "source_url": url,
    }

def text_sig(text: str) -> str:
    return hashlib.sha1((text or "").strip().lower().encode("utf-8")).hexdigest()


def collect_spam(spam_urls: List[str]):
    if not spam_urls:
        raise ValueError("Hiç URL bulunamadı. spam_urls.txt kontrol et.")

    downloader = YoutubeCommentDownloader()

    # bazı spam tiplerini aşırı doldurmasın diye cap
    SPAM_REASON_CAPS = {
        "contact_link": int(TARGET_SPAM * 0.25),
        "repeat_same_comment": int(TARGET_SPAM * 0.20),
    }

    ck = load_checkpoint()
    if ck:
        spam_rows = ck.get("spam_rows", [])
        url_only_taken = int(ck.get("url_only_taken", 0))
        spam_by_video = ck.get("spam_by_video", {})
        seen_cid = set(ck.get("seen_cid", []))
        seen_author_text = set(ck.get("seen_author_text", []))
        spam_reason_counts = defaultdict(int, ck.get("spam_reason_counts", {}))
        spam_text_counts = defaultdict(int, ck.get("spam_text_counts", {}))
        print(f"[RESUME] spam={len(spam_rows)} url_only={url_only_taken}")
    else:
        spam_rows = []
        url_only_taken = 0
        spam_by_video = {}
        seen_cid = set()
        seen_author_text = set()
        spam_reason_counts = defaultdict(int)
        spam_text_counts = defaultdict(int)

    def checkpoint():
        save_checkpoint({
            "spam_rows": spam_rows,
            "url_only_taken": url_only_taken,
            "spam_by_video": spam_by_video,
            "seen_cid": list(seen_cid),
            "seen_author_text": list(seen_author_text),
            "spam_reason_counts": dict(spam_reason_counts),
            "spam_text_counts": dict(spam_text_counts),
        })

    def is_unique_spam(text: str, cid: Optional[str]) -> bool:
        if not text:
            return False
        sig = text_sig(text)

        if cid and cid in seen_cid:
            return False

        if DEDUP_SPAM_BY_TEXT:
            # tam dedupe
            if spam_text_counts[sig] >= 1:
                return False
            return True

        # dedupe kapalı ama cap var
        if spam_text_counts[sig] >= SPAM_TEXT_MAX:
            return False
        return True

    def mark_used_spam(text: str, cid: Optional[str]):
        sig = text_sig(text)
        spam_text_counts[sig] += 1
        if cid:
            seen_cid.add(cid)

    def process_video(url: str, sort_by, scan_cap: int):
        nonlocal url_only_taken

        comments = downloader.get_comments_from_url(url, sort_by=sort_by)
        scanned = 0

        for c in comments:
            scanned += 1
            if scanned > scan_cap:
                break

            if should_stop():
                print("\n[STOP] STOP.txt bulundu -> kaydedip çıkıyorum.")
                checkpoint()
                raise SystemExit("STOP requested")

            if len(spam_rows) >= SPAM_BUFFER:
                break

            text = clean_text(c.get("text"))
            if not text:
                continue

            if EN_ONLY_SPAM and not is_english_like_spam(text):
                continue

            cid = c.get("cid")
            if not is_unique_spam(text, cid):
                continue

            author = (c.get("author") or "").strip().lower()
            author_text_key = f"{author}||{text.lower()}"
            repeated_by_same_author = author and (author_text_key in seen_author_text)

            spam_flag, reason = is_spam_with_reason(text)

            if repeated_by_same_author and not spam_flag:
                spam_flag, reason = True, "repeat_same_comment"

            if not spam_flag:
                continue

            if reason == "url_only":
                if url_only_taken >= URL_ONLY_CAP:
                    continue
                url_only_taken += 1

            cap = SPAM_REASON_CAPS.get(reason)
            if cap is not None and spam_reason_counts[reason] >= cap:
                continue

            spam_reason_counts[reason] += 1
            spam_rows.append(make_row(c, url, reason))
            spam_by_video[url] = int(spam_by_video.get(url, 0)) + 1

            mark_used_spam(text, cid)
            if author:
                seen_author_text.add(author_text_key)

        return

    # -------------------------
    # FAST PASS (spam_urls)
    # -------------------------
    print("[INFO] HAS_POPULAR:", HAS_POPULAR)
    print("[INFO] spam_urls:", len(spam_urls))

    for url in spam_urls:
        if len(spam_rows) >= SPAM_BUFFER:
            break

        for attempt in range(RETRY_PER_VIDEO + 1):
            try:
                process_video(url, SORT_BY_RECENT, FAST_RECENT_CAP_SPAM)
                time.sleep(random.uniform(*SLEEP_BETWEEN))
                break
            except Exception as e:
                print("[ERR SPAM FAST RECENT]", url, "->", repr(e))
                time.sleep(random.uniform(*SLEEP_RETRY))

        if HAS_POPULAR and len(spam_rows) < SPAM_BUFFER:
            for attempt in range(RETRY_PER_VIDEO + 1):
                try:
                    process_video(url, SORT_BY_POPULAR, FAST_POPULAR_CAP_SPAM)
                    time.sleep(random.uniform(*SLEEP_BETWEEN))
                    break
                except Exception as e:
                    print("[ERR SPAM FAST POP]", url, "->", repr(e))
                    time.sleep(random.uniform(*SLEEP_RETRY))

        print(f"[SPAM FAST] spam {len(spam_rows)}/{SPAM_BUFFER} | url_only {url_only_taken}/{URL_ONLY_CAP}")
        checkpoint()

    # -------------------------
    # DEEP PASS (eksik kalırsa)
    # -------------------------
    if DO_DEEP_SPAM and len(spam_rows) < SPAM_BUFFER:
        print("\n[DEEP SPAM] spam eksik, derin tarama...\n")
        urls_sorted = sorted(spam_urls, key=lambda u: int(spam_by_video.get(u, 0)), reverse=True)

        for url in urls_sorted:
            if len(spam_rows) >= SPAM_BUFFER:
                break

            for attempt in range(RETRY_PER_VIDEO + 1):
                try:
                    process_video(url, SORT_BY_RECENT, DEEP_CAP)
                    time.sleep(random.uniform(*SLEEP_BETWEEN))
                    break
                except Exception as e:
                    print("[ERR DEEP SPAM RECENT]", url, "->", repr(e))
                    time.sleep(random.uniform(*SLEEP_RETRY))

            if HAS_POPULAR and len(spam_rows) < SPAM_BUFFER:
                for attempt in range(RETRY_PER_VIDEO + 1):
                    try:
                        process_video(url, SORT_BY_POPULAR, DEEP_CAP)
                        time.sleep(random.uniform(*SLEEP_BETWEEN))
                        break
                    except Exception as e:
                        print("[ERR DEEP SPAM POP]", url, "->", repr(e))
                        time.sleep(random.uniform(*SLEEP_RETRY))

            print(f"[DEEP SPAM] spam {len(spam_rows)}/{SPAM_BUFFER} | url_only {url_only_taken}/{URL_ONLY_CAP}")
            checkpoint()

    # -------------------------
    # Save outputs
    # -------------------------
    df_spam_full = pd.DataFrame(spam_rows)
    df_spam = df_spam_full.head(TARGET_SPAM).reset_index(drop=True)

    df_spam[["comment", "label"]].to_csv(OUT_SPAM_ONLY, index=False, encoding="utf-8")

    per_video = []
    for u in spam_urls:
        per_video.append({
            "source_url": u,
            "spam_count": int((df_spam_full["source_url"] == u).sum()) if "source_url" in df_spam_full.columns else 0,
        })
    pd.DataFrame(per_video).to_csv(OUT_COUNTS, index=False, encoding="utf-8")

    print("\n[DONE]")
    print("Spam:", len(df_spam), "/", TARGET_SPAM, "->", OUT_SPAM_ONLY)
    print("Spam (full):", OUT_SPAM_DATASET)
    print("Counts:", OUT_COUNTS)

    if "spam_reason" in df_spam.columns and len(df_spam) > 0:
        print("\n[spam_reason top]")
        print(df_spam["spam_reason"].value_counts().head(20))

    if len(df_spam) < TARGET_SPAM:
        print("\n[WARN] 1500'e tamamlanamadı -> daha spam-heavy / daha çok yorumlu video ekle.")


if __name__ == "__main__":
    spam_urls = read_urls_any("spam_urls.txt")
    collect_spam(spam_urls)
