import os
import re
import json
import time
import random
import hashlib
from typing import List, Dict, Optional
from urllib.parse import urlparse, parse_qs

import pandas as pd
from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_RECENT

from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0


# =========================
# CONFIG (ROUND BASED)
# =========================
TARGET_NORMAL = 1500
NORMAL_BUFFER = int(TARGET_NORMAL * 1.15)  # 1725

EN_ONLY = True  # ✅ sadece İngilizce

FIRST_PASS_SCAN = 100          # 1. tur: her videoda 100 yorum oku
DEEP_STEP_SCAN = 2500          # 2. tur: her videoda bir seferde kaç yorum daha oku
MAX_TOTAL_SCAN_PER_VIDEO = 50000

# Retry / sleep
RETRY_PER_VIDEO = 1
SLEEP_BETWEEN = (0.03, 0.09)
SLEEP_RETRY = (0.25, 0.60)

# Resume / stop
CHECKPOINT_PATH = "checkpoint_normal_rounds.json"
STOP_FILE = "STOP.txt"

# Outputs
OUT_NORMAL_ONLY = "normal_only_en.csv"
OUT_DATASET = "dataset_only_normal_en.csv"
OUT_COUNTS = "per_video_counts_normal_only.csv"


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
MENTION_PREFIX_RE = re.compile(r"^(?:@\S+\s*)+")
NONLETTER_RE = re.compile(r"[^A-Za-z\s]+")

def clean_text(t: str) -> str:
    return WS_RE.sub(" ", (t or "")).strip()

# kısa İngilizce yorumları da kaçırmamak için mini whitelist
SHORT_EN_WHITELIST = {
    "nice", "great", "awesome", "amazing", "cool", "wow", "love", "thanks", "thank you",
    "good", "perfect", "lol", "lmao", "omg", "yes", "no", "ok", "okay", "beautiful"
}

def is_english_strict(text: str) -> bool:
    """
    Gerçek dil tespiti:
    - baştaki @mentionları kaldırır
    - yeterli uzunluk varsa langdetect ile 'en' kontrolü yapar
    - çok kısa yorumlarda whitelist ile karar verir
    """
    t = clean_text(text)
    if not t:
        return False

    # mention temizle
    t2 = MENTION_PREFIX_RE.sub("", t).strip()
    if not t2:
        return False

    # sadece harf + boşluk bırak (langdetect için)
    t3 = NONLETTER_RE.sub(" ", t2)
    t3 = clean_text(t3)

    # çok kısa -> whitelist
    low = t3.lower()
    if len(low) <= 20 and len(low.split()) <= 2:
        return low in SHORT_EN_WHITELIST

    # langdetect: en az biraz anlamlı içerik olsun
    if len(t3) < 15 or len(t3.split()) < 3:
        return False

    try:
        return detect(t3) == "en"
    except Exception:
        return False


# =========================
# Spam filter (normal toplarken spam'ı dışlamak için)
# =========================
URL_RE = re.compile(r"(https?://|www\.)\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", re.IGNORECASE)
PHONE_RE = re.compile(r"(\+?\d[\d \-\(\)]{7,}\d)")
SHORTENER_RE = re.compile(r"\b(bit\.ly|cutt\.ly|tinyurl\.com|t\.co|goo\.gl|ow\.ly|rb\.gy|is\.gd)\b", re.IGNORECASE)

CONTACT_LINK_RE = re.compile(
    r"\b(t\.me/|telegram\.me/|telegram\.dog/|wa\.me/|chat\.whatsapp\.com/|discord\.gg/|snapchat\.com/|kik\.me/)\S+",
    re.IGNORECASE
)

CTA_RE = re.compile(r"\b(dm|dms|inbox|message|text|contact|follow|add me|reach me|hit me up)\b", re.IGNORECASE)
PLATFORM_RE = re.compile(r"\b(instagram|insta|ig|telegram|whatsapp|discord|snapchat|kik)\b", re.IGNORECASE)
HANDLE_RE = re.compile(r"@[A-Za-z0-9_\.]{3,}", re.IGNORECASE)

SELF_PROMO_RE = [
    re.compile(r"\bcheck (out )?my (channel|video)\b", re.IGNORECASE),
    re.compile(r"\bvisit my (channel|page)\b", re.IGNORECASE),
    re.compile(r"\bgo to my channel\b", re.IGNORECASE),
    re.compile(r"\bwatch my (new )?video\b", re.IGNORECASE),
    re.compile(r"\bsubscribe to my channel\b", re.IGNORECASE),
]

GIVEAWAY_KW = ["giveaway", "winner", "congratulations", "congrats", "claim", "prize", "reward", "raffle", "selected"]
MONEY_SCAM_KW = ["work from home", "guaranteed profit", "easy money", "double your money", "crypto signals", "forex signals"]
STORE_KW = ["buy now", "official store", "promo code", "discount code", "limited offer", "sale"]

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

def is_spam_like(text: str) -> bool:
    t = clean_text(text)
    if not t:
        return False

    tl = t.lower()
    url_n = count_urls(t)
    has_url = bool(URL_RE.search(t))
    has_short = bool(SHORTENER_RE.search(t))
    has_contact_link = bool(CONTACT_LINK_RE.search(t))

    if has_contact_link:
        return True
    if EMAIL_RE.search(t) or PHONE_RE.search(t):
        return True
    if any(rx.search(t) for rx in SELF_PROMO_RE):
        return True
    if is_url_only_like(t):
        return True
    if url_n >= 2:
        return True

    has_platform = bool(PLATFORM_RE.search(t))
    has_handle = bool(HANDLE_RE.search(t))

    if CTA_RE.search(t) and (has_platform or has_handle or has_url):
        return True

    if (has_url or has_short) and (contains_any(tl, GIVEAWAY_KW) or contains_any(tl, MONEY_SCAM_KW) or contains_any(tl, STORE_KW)):
        return True

    if len(t) < 10:
        if has_url or has_short or has_platform:
            return True
        return False

    if repeated_chars(t) and (has_url or has_short):
        return True

    return False


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
# Collector (ROUND BASED)
# =========================
def text_sig(text: str) -> str:
    return hashlib.sha1((text or "").strip().lower().encode("utf-8")).hexdigest()

def make_row(c: Dict, url: str) -> Dict:
    return {
        "comment": clean_text(c.get("text")),
        "label": 0,
        "cid": c.get("cid"),
        "author": c.get("author"),
        "time": c.get("time"),
        "votes": c.get("votes"),
        "source_url": url,
    }


def collect_round_based(normal_urls: List[str]):
    if not normal_urls:
        raise ValueError("normal_urls.txt içinde hiç URL yok.")

    downloader = YoutubeCommentDownloader()

    ck = load_checkpoint()
    if ck:
        normal_rows = ck.get("normal_rows", [])
        normal_by_video = ck.get("normal_by_video", {})
        seen_cid = set(ck.get("seen_cid", []))
        seen_text = set(ck.get("seen_text", []))
        seen_author_text = set(ck.get("seen_author_text", []))
        scanned_by_video = ck.get("scanned_by_video", {})
        fast_done = bool(ck.get("fast_done", False))
        print(f"[RESUME] normal={len(normal_rows)} fast_done={fast_done}")
    else:
        normal_rows = []
        normal_by_video = {}
        seen_cid, seen_text, seen_author_text = set(), set(), set()
        scanned_by_video = {u: 0 for u in normal_urls}
        fast_done = False

    for u in normal_urls:
        scanned_by_video.setdefault(u, 0)
        normal_by_video.setdefault(u, 0)

    def checkpoint():
        save_checkpoint({
            "normal_rows": normal_rows,
            "normal_by_video": normal_by_video,
            "seen_cid": list(seen_cid),
            "seen_text": list(seen_text),
            "seen_author_text": list(seen_author_text),
            "scanned_by_video": scanned_by_video,
            "fast_done": fast_done,
        })

    def is_unique(text: str, cid: Optional[str]) -> bool:
        if not text:
            return False
        if cid and cid in seen_cid:
            return False
        sig = text_sig(text)
        if sig in seen_text:
            return False
        return True

    def mark_used(text: str, cid: Optional[str], author_text_key: Optional[str]):
        sig = text_sig(text)
        seen_text.add(sig)
        if cid:
            seen_cid.add(cid)
        if author_text_key:
            seen_author_text.add(author_text_key)

    def process_video_with_skip(url: str, need_scan_more: int) -> int:
        already_scanned = int(scanned_by_video.get(url, 0))
        if already_scanned >= MAX_TOTAL_SCAN_PER_VIDEO:
            return 0

        scan_limit = min(need_scan_more, MAX_TOTAL_SCAN_PER_VIDEO - already_scanned)
        if scan_limit <= 0:
            return 0

        comments = downloader.get_comments_from_url(url, sort_by=SORT_BY_RECENT)

        # SKIP
        skipped = 0
        for c in comments:
            if skipped >= already_scanned:
                first_item = c
                break
            skipped += 1
        else:
            return 0  # yorum bitti

        scanned_now = 0

        def handle_comment(cdict: Dict):
            nonlocal normal_rows

            if len(normal_rows) >= NORMAL_BUFFER:
                return

            text = clean_text(cdict.get("text"))
            if not text:
                return

            if EN_ONLY and (not is_english_strict(text)):
                return

            if is_spam_like(text):
                return

            cid = cdict.get("cid")
            if not is_unique(text, cid):
                return

            author = (cdict.get("author") or "").strip().lower()
            author_text_key = f"{author}||{text.lower()}" if author else None
            if author_text_key and author_text_key in seen_author_text:
                return

            normal_rows.append(make_row(cdict, url))
            normal_by_video[url] = int(normal_by_video.get(url, 0)) + 1
            mark_used(text, cid, author_text_key)

        # first
        handle_comment(first_item)
        scanned_now += 1

        # rest
        for c in comments:
            if should_stop():
                print("\n[STOP] STOP.txt bulundu -> kaydedip çıkıyorum.")
                checkpoint()
                raise SystemExit("STOP requested")

            if len(normal_rows) >= NORMAL_BUFFER:
                break

            scanned_now += 1
            handle_comment(c)

            if scanned_now >= scan_limit:
                break

        scanned_by_video[url] = already_scanned + scanned_now
        return scanned_now

    # ROUND 1 FAST
    if not fast_done:
        print(f"\n[ROUND-1 FAST] Her videoda ilk {FIRST_PASS_SCAN} yoruma bakıyorum...\n")
        for url in normal_urls:
            if len(normal_rows) >= NORMAL_BUFFER:
                break

            for _attempt in range(RETRY_PER_VIDEO + 1):
                try:
                    need = max(0, FIRST_PASS_SCAN - int(scanned_by_video.get(url, 0)))
                    if need > 0:
                        process_video_with_skip(url, need_scan_more=need)
                    time.sleep(random.uniform(*SLEEP_BETWEEN))
                    break
                except Exception as e:
                    print("[ERR FAST]", url, "->", repr(e))
                    time.sleep(random.uniform(*SLEEP_RETRY))

            print(f"[FAST] collected={len(normal_rows)}/{NORMAL_BUFFER} | scanned={scanned_by_video.get(url,0)} | taken={normal_by_video.get(url,0)}")
            checkpoint()

        fast_done = True
        checkpoint()
        print("\n[ROUND-1 DONE] Tüm videolar gezildi. Şimdi başa dönüp DEEP tur başlıyor.\n")

    # ROUND 2 DEEP
    print(f"[ROUND-2 DEEP] Target dolana kadar dolaşıp derin tarıyorum (step={DEEP_STEP_SCAN})...\n")

    no_progress_rounds = 0
    while len(normal_rows) < NORMAL_BUFFER:
        made_progress = False

        for url in normal_urls:
            if len(normal_rows) >= NORMAL_BUFFER:
                break

            before = len(normal_rows)

            for _attempt in range(RETRY_PER_VIDEO + 1):
                try:
                    scanned_now = process_video_with_skip(url, need_scan_more=DEEP_STEP_SCAN)
                    time.sleep(random.uniform(*SLEEP_BETWEEN))
                    break
                except Exception as e:
                    print("[ERR DEEP]", url, "->", repr(e))
                    time.sleep(random.uniform(*SLEEP_RETRY))
                    scanned_now = 0

            after = len(normal_rows)
            if after > before:
                made_progress = True

            print(f"[DEEP] collected={after}/{NORMAL_BUFFER} | scanned_now={scanned_now} | total_scanned={scanned_by_video.get(url,0)} | taken={normal_by_video.get(url,0)}")
            checkpoint()

        if not made_progress:
            no_progress_rounds += 1
            if no_progress_rounds >= 2:
                print("\n[INFO] DEEP turda ilerleme yok. Yorumlar bitmiş olabilir ya da filtre yüzünden alamıyor olabiliriz.")
                break
        else:
            no_progress_rounds = 0

    # SAVE
    df_normal = (
        pd.DataFrame(normal_rows)
        .drop_duplicates(subset=["comment"])
        .head(TARGET_NORMAL)
        .reset_index(drop=True)
    )

    df_normal[["comment", "label"]].to_csv(OUT_NORMAL_ONLY, index=False, encoding="utf-8")
    df_normal[["comment", "label"]].to_csv(OUT_DATASET, index=False, encoding="utf-8")

    per_video = []
    for u in normal_urls:
        per_video.append({
            "source_url": u,
            "normal_count": int(normal_by_video.get(u, 0)),
            "total_scanned": int(scanned_by_video.get(u, 0)),
        })
    pd.DataFrame(per_video).to_csv(OUT_COUNTS, index=False, encoding="utf-8")

    print("\n[DONE]")
    print("Normal:", len(df_normal), "/", TARGET_NORMAL, "->", OUT_NORMAL_ONLY)
    print("Dataset:", OUT_DATASET)
    print("Counts:", OUT_COUNTS)

    if len(df_normal) < TARGET_NORMAL:
        print("\n[WARN] Normal hedefe tamamlanamadı -> daha çok URL / daha çok yorumlu video gerekebilir.")


if __name__ == "__main__":
    normal_urls = read_urls_any("normal_urls.txt")
    collect_round_based(normal_urls)
