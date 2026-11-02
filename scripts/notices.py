"""Livestock-health notices for Uganda: disease outbreaks, quarantines and vaccination calls.

Collects announcements from official and trusted news sources, keeps only those about
animal disease or its control in Uganda, and writes them as JSON. Each notice keeps the
source's own title, date and link: nothing is rewritten or summarised, so the farmer
always sees what the source said and can open it.

Standard library only, so the same file runs in the agent, in tests and as a scheduled
job (GitHub Actions in the public demo repo):

    python domain/services/notices.py --out notices.json

Sources (see SOURCES): the MAAIF website feed, a Google News search limited to Ugandan
livestock-disease news (which carries the Daily Monitor, New Vision, Nile Post,
allAfrica and others), and allAfrica's Uganda headlines. A source that fails is
reported in the output and skipped; if every source fails, the previous notices are
kept rather than replaced with an empty list.
"""

from __future__ import annotations

import argparse
import datetime as dt
import email.utils
import hashlib
import html
import json
import re
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Dict, Iterable, List, Optional, Tuple

def _gnews(q: str) -> str:
    return "https://news.google.com/rss/search?" + urllib.parse.urlencode({"q": q + " when:90d", "hl": "en-UG", "gl": "UG", "ceid": "UG:en"})


SOURCES: List[Dict[str, str]] = [
    {"id": "maaif", "name": "MAAIF (agriculture.go.ug)", "kind": "official",
     "url": "https://www.agriculture.go.ug/feed/"},
    {"id": "gnews_fmd", "name": "Google News: FMD in Uganda", "kind": "news",
     "url": _gnews('Uganda ("foot and mouth" OR "foot-and-mouth" OR FMD)')},
    {"id": "gnews_diseases", "name": "Google News: other animal diseases in Uganda", "kind": "news",
     "url": _gnews('Uganda (anthrax OR "lumpy skin" OR CBPP OR "East Coast Fever" OR "Rift Valley fever" OR "swine fever" '
                   'OR "bird flu" OR "avian influenza" OR PPR OR rabies OR trypanosomiasis)')},
    {"id": "gnews_control", "name": "Google News: livestock quarantines and vaccination in Uganda", "kind": "news",
     "url": _gnews('Uganda (livestock OR cattle OR animals) (quarantine OR vaccination OR "movement ban" OR outbreak OR MAAIF)')},
    {"id": "allafrica", "name": "allAfrica: Uganda", "kind": "news",
     "url": "https://allafrica.com/tools/headlines/rdf/uganda/headlines.rdf"},
]

# Animal diseases: naming one is enough.
_DISEASE = re.compile(
    r"foot[- ]and[- ]mouth|\bFMD\b|anthrax|lumpy skin|\bCBPP\b|contagious bovine|east coast fever|\bECF\b|"
    r"rift valley|rabies|brucell|peste des petits|\bPPR\b|swine fever|\bASF\b|avian influenza|bird flu|"
    r"newcastle disease|trypanosom|tsetse|acaricide|\bticks?\b|nagana|redwater|heartwater",
    re.I)
# General control words count only when the headline is about animals (a travel
# quarantine for people is not a livestock notice).
_CONTROL = re.compile(r"quarantine|vaccinat|outbreak|movement ban|bans? .{0,20}movement|market closure|clos\w* .{0,25}markets?", re.I)
_ANIMAL = re.compile(r"\b(cattle|cows?|calves|bulls?|livestock|animals?|herds?|pigs?|piggery|goats?|sheep|poultry|chickens?|birds?|"
                     r"dogs?|farmers?|veterinar\w*|vets?|MAAIF|dairy|milk|abattoirs?|slaughter)\b", re.I)
# Google News items are kept only from these publishers (official bodies, Ugandan
# outlets and international wires). allAfrica and the MAAIF feed are sources in their own right.
TRUSTED_PUBLISHERS = (
    "monitor", "new vision", "newvision", "nile post", "nilepost", "uganda radio network", "uganda radionetwork",
    "chimpreports", "independent", "uganda media centre", "ubc", "kampala post", "pml daily", "observer",
    "ntv uganda", "nbs", "the eastafrican", "reuters", "associated press", "xinhua", "allafrica",
    "ministry of agriculture", "maaif", "fao", "woah", "world organisation for animal health", "who",
)
# Human-only health stories that share words with the topic list.
_EXCLUDE = re.compile(r"\bebola\b|\bmarburg\b|\bmpox\b|\bcholera\b|\bmalaria\b|\bpolio\b|\bmeasles\b|covid|\bsudan virus\b", re.I)

UGANDA_DISTRICTS = [
    "Abim", "Adjumani", "Agago", "Alebtong", "Amolatar", "Amudat", "Amuria", "Amuru", "Apac", "Arua", "Budaka",
    "Bududa", "Bugiri", "Bugweri", "Buhweju", "Buikwe", "Bukedea", "Bukomansimbi", "Bukwo", "Bulambuli", "Buliisa",
    "Bundibugyo", "Bunyangabu", "Bushenyi", "Busia", "Butaleja", "Butambala", "Butebo", "Buvuma", "Buyende",
    "Dokolo", "Gomba", "Gulu", "Hoima", "Ibanda", "Iganga", "Isingiro", "Jinja", "Kaabong", "Kabale", "Kabarole",
    "Kaberamaido", "Kagadi", "Kakumiro", "Kalaki", "Kalangala", "Kaliro", "Kalungu", "Kampala", "Kamuli",
    "Kamwenge", "Kanungu", "Kapchorwa", "Kapelebyong", "Karenga", "Kasanda", "Kassanda", "Kasese", "Katakwi",
    "Kayunga", "Kazo", "Kibaale", "Kiboga", "Kibuku", "Kikuube", "Kiruhura", "Kiryandongo", "Kisoro", "Kitagwenda",
    "Kitgum", "Koboko", "Kole", "Kotido", "Kumi", "Kwania", "Kween", "Kyankwanzi", "Kyegegwa", "Kyenjojo",
    "Kyotera", "Lamwo", "Lira", "Luuka", "Luwero", "Lwengo", "Lyantonde", "Madi-Okollo", "Manafwa", "Maracha",
    "Masaka", "Masindi", "Mayuge", "Mbale", "Mbarara", "Mitooma", "Mityana", "Moroto", "Moyo", "Mpigi", "Mubende",
    "Mukono", "Nabilatuk", "Nakapiripirit", "Nakaseke", "Nakasongola", "Namayingo", "Namisindwa", "Namutumba",
    "Napak", "Nebbi", "Ngora", "Ntoroko", "Ntungamo", "Nwoya", "Obongi", "Omoro", "Otuke", "Oyam", "Pader",
    "Pakwach", "Pallisa", "Rakai", "Rubanda", "Rubirizi", "Rukiga", "Rukungiri", "Rwampara", "Sembabule",
    "Serere", "Sheema", "Sironko", "Soroti", "Terego", "Tororo", "Wakiso", "Yumbe", "Zombo",
]
_DISTRICT_RE = re.compile(r"\b(" + "|".join(re.escape(d) for d in UGANDA_DISTRICTS) + r")\b")

_CATEGORY = [
    ("quarantine", re.compile(r"quarantine|movement ban|bans? .{0,25}movement|ban on .{0,25}movement|markets? (closed|closure)|clos\w* .{0,25}markets?", re.I)),
    ("vaccination", re.compile(r"vaccinat|\bjabs?\b", re.I)),
    ("outbreak", re.compile(r"outbreak|confirm\w* (case|cases)|suspected|\bhits?\b|\bkills?\b|\bdead\b|deaths?|strikes?|spread", re.I)),
]

MAX_AGE_DAYS = 90
MAX_ITEMS = 40
_UA = "DSTUganda-notices/1.0 (+https://github.com/edgarmutungisa/demo)"


def _text(el: Optional[ET.Element]) -> str:
    return (el.text or "").strip() if el is not None else ""


def _strip_html(s: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", s or ""))).strip()


def _parse_date(s: str) -> Optional[dt.datetime]:
    s = (s or "").strip()
    if not s:
        return None
    try:
        d = email.utils.parsedate_to_datetime(s)
    except (TypeError, ValueError):
        d = None
    if d is None:
        try:
            d = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
        except ValueError:
            return None
    return d if d.tzinfo else d.replace(tzinfo=dt.timezone.utc)


def parse_feed(xml_text: str) -> List[dict]:
    """RSS 2.0, RSS 1.0 (RDF) or Atom -> [{title, link, published, summary}]."""
    root = ET.fromstring(xml_text)
    items: List[dict] = []
    for el in root.iter():
        tag = el.tag.split("}")[-1]
        if tag not in ("item", "entry"):
            continue
        kids = {c.tag.split("}")[-1]: c for c in el}
        link = _text(kids.get("link"))
        if not link and kids.get("link") is not None:
            link = kids["link"].get("href", "")
        date = (_text(kids.get("pubDate")) or _text(kids.get("date")) or _text(kids.get("published"))
                or _text(kids.get("updated")))
        source_el = kids.get("source")
        items.append({
            "title": _strip_html(_text(kids.get("title"))),
            "link": link.strip(),
            "published": _parse_date(date),
            "summary": _strip_html(_text(kids.get("description")) or _text(kids.get("summary"))
                                   or _text(kids.get("encoded")))[:600],
            "publisher": _text(source_el) if source_el is not None else "",
        })
    return items


def classify(title: str, summary: str = "") -> Optional[dict]:
    """Return {category, districts} if this is a Ugandan animal-health notice, else None."""
    text = f"{title} {summary}"
    relevant = bool(_DISEASE.search(text)) or (bool(_CONTROL.search(text)) and bool(_ANIMAL.search(title)))
    if not relevant or (_EXCLUDE.search(title) and not _DISEASE.search(title)):
        return None
    districts = sorted(set(_DISTRICT_RE.findall(text)))
    if not districts and not re.search(r"\bUganda\b|\bUgandan\b|\bMAAIF\b", text):
        return None
    category = next((name for name, rx in _CATEGORY if rx.search(text)), "advisory")
    return {"category": category, "districts": districts}


def _clean_title(title: str, publisher: str) -> Tuple[str, str]:
    # Google News titles end with " - Publisher".
    m = re.match(r"^(.*\S)\s+-\s+([^-]{2,60})$", title)
    if m and (not publisher or m.group(2).strip() == publisher):
        return m.group(1), m.group(2).strip()
    return title, publisher


_TRUSTED_RE = re.compile(r"\b(" + "|".join(re.escape(t) for t in TRUSTED_PUBLISHERS) + r")\b", re.I)


def _trusted(publisher: str) -> bool:
    return bool(_TRUSTED_RE.search(publisher or ""))


_STOP = {"the", "a", "an", "of", "in", "on", "for", "to", "and", "as", "over", "after", "with", "from", "at", "by", "uganda", "ugandan", "is", "its"}


def _words(title: str) -> set:
    t = re.sub(r"foot[- ]and[- ]mouth(?: disease)?", "fmd", title.lower())
    return {w for w in re.findall(r"[a-z0-9]+", t) if w not in _STOP and len(w) > 2}


def _same_story(a: set, b: set) -> bool:
    return bool(a and b) and len(a & b) / len(a | b) >= 0.5


def build_notices(feeds: Dict[str, str], now: Optional[dt.datetime] = None, stats: Optional[dict] = None) -> List[dict]:
    """feeds: {source_id: xml_text}. Returns filtered, de-duplicated notices, newest first.
    If stats is a dict, it is filled per source with how many items were kept and why others were dropped."""
    stats = {} if stats is None else stats
    def count(sid: str, why: str, publisher: str = "") -> None:
        st = stats.setdefault(sid, {"kept": 0, "too_old": 0, "undated": 0, "untrusted_publisher": 0, "not_animal_health": 0, "duplicate": 0, "untrusted_names": [], "newest_seen": None})
        st[why] += 1
        if why == "untrusted_publisher" and publisher and publisher not in st["untrusted_names"] and len(st["untrusted_names"]) < 15:
            st["untrusted_names"].append(publisher)
    now = now or dt.datetime.now(dt.timezone.utc)
    src = {s["id"]: s for s in SOURCES}
    seen, out, stories = set(), [], []
    rank = {"official": 0, "news": 1}
    items = []
    for sid, xml_text in feeds.items():
        meta = src.get(sid, {"id": sid, "name": sid, "kind": "news"})
        items += [(sid, meta, it) for it in parse_feed(xml_text)]
    # Official items first, then newest first, so a duplicate keeps the official or most recent version.
    epoch = dt.datetime.min.replace(tzinfo=dt.timezone.utc)
    items.sort(key=lambda x: (rank.get(x[1]["kind"], 1), -(x[2]["published"] or epoch).timestamp()))
    for sid, meta, it in items:
        if not it["title"] or not it["link"].startswith("http"):
            continue
        pub = it["published"]
        if pub is not None:
            st = stats.setdefault(sid, {"kept": 0, "too_old": 0, "undated": 0, "untrusted_publisher": 0, "not_animal_health": 0, "duplicate": 0, "untrusted_names": [], "newest_seen": None})
            iso = pub.astimezone(dt.timezone.utc).strftime("%Y-%m-%d")
            if not st["newest_seen"] or iso > st["newest_seen"]:
                st["newest_seen"] = iso
        if pub is None:
            count(sid, "undated"); continue
        if (now - pub).days > MAX_AGE_DAYS or pub > now + dt.timedelta(days=1):
            count(sid, "too_old"); continue
        title, publisher = _clean_title(it["title"], it["publisher"])
        if sid.startswith("gnews") and not _trusted(publisher):
            count(sid, "untrusted_publisher", publisher); continue
        c = classify(title, it["summary"] if meta["kind"] == "official" else "")
        if c is None:
            count(sid, "not_animal_health"); continue
        key, words = re.sub(r"[^a-z0-9]+", " ", title.lower()).strip(), _words(title)
        # Same story: very similar headline, or the same district(s) and kind of notice within 3 days.
        twin = any(_same_story(words, w) or (c["districts"] and c["districts"] == d and c["category"] == cat and abs((pub - p).days) <= 3)
                   for w, d, cat, p in stories)
        if key in seen or twin:
            count(sid, "duplicate"); continue
        seen.add(key); stories.append((words, c["districts"], c["category"], pub)); count(sid, "kept")
        out.append({
            "id": hashlib.sha1(it["link"].encode()).hexdigest()[:12],
            "title": title,
            "url": it["link"],
            "published": pub.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
            "source": meta["name"],
            "source_kind": meta["kind"],
            "publisher": publisher or meta["name"],
            "category": c["category"],
            "districts": c["districts"],
        })
    out.sort(key=lambda n: n["published"], reverse=True)
    return out[:MAX_ITEMS]


def fetch(url: str, timeout: int = 30) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": _UA, "Accept": "application/rss+xml, application/xml, text/xml, */*"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode(r.headers.get_content_charset() or "utf-8", errors="replace")


def collect(sources: Iterable[dict] = SOURCES, fetcher=fetch, previous: Optional[dict] = None) -> dict:
    feeds, status = {}, []
    for s in sources:
        try:
            xml_text = fetcher(s["url"])
            n = len(parse_feed(xml_text))
            feeds[s["id"]] = xml_text
            status.append({"id": s["id"], "name": s["name"], "url": s["url"], "ok": True, "items_read": n})
        except Exception as exc:  # network, HTTP or XML error: report it and carry on
            status.append({"id": s["id"], "name": s["name"], "url": s["url"], "ok": False, "error": f"{type(exc).__name__}: {exc}"[:200]})
    stats: dict = {}
    notices = build_notices(feeds, stats=stats)
    for st in status:
        if st["ok"]:
            st.update(stats.get(st["id"], {}))
    kept_previous = False
    if not feeds and previous and previous.get("notices"):
        notices, kept_previous = previous["notices"], True
    return {
        "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
        "note": "Titles, dates and links exactly as published by each source. Always open the source and follow your District Veterinary Officer.",
        "kept_previous": kept_previous,
        "sources": status,
        "notices": notices,
    }


def for_district(data: dict, district: Optional[str]) -> dict:
    """Notices with those naming the user's district first."""
    d = (district or "").strip().lower().replace(" district", "")
    items = data.get("notices", [])
    mine = [n for n in items if d and d in [x.lower() for x in n.get("districts", [])]]
    rest = [n for n in items if n not in mine]
    return {**{k: v for k, v in data.items() if k != "notices"}, "district": district, "in_your_district": mine, "other_notices": rest}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="notices.json")
    a = ap.parse_args(argv)
    try:
        with open(a.out, encoding="utf-8") as f:
            previous = json.load(f)
    except (OSError, ValueError):
        previous = None
    data = collect(previous=previous)
    sig = lambda d: json.dumps([d.get("notices"), [(x["id"], x["ok"]) for x in d.get("sources", [])]], sort_keys=True)  # counts alone don't trigger a commit
    if previous and sig(previous) == sig(data):
        print("unchanged: notices and source status are the same as last time")
        return 0
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    ok = sum(s["ok"] for s in data["sources"])
    print(f"{len(data['notices'])} notices from {ok}/{len(data['sources'])} sources" + (" (kept previous)" if data["kept_previous"] else ""))
    for s in data["sources"]:
        print(f"  {'ok ' if s['ok'] else 'ERR'} {s['name']}: " + (f"read {s['items_read']}, kept {s.get('kept', 0)}, untrusted {s.get('untrusted_publisher', 0)} {s.get('untrusted_names', [])}, not animal health {s.get('not_animal_health', 0)}, duplicate {s.get('duplicate', 0)}, old {s.get('too_old', 0)}, undated {s.get('undated', 0)}, newest {s.get('newest_seen')}" if s['ok'] else s['error']))
    return 0


if __name__ == "__main__":
    sys.exit(main())
