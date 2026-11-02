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

_GNEWS_Q = ('Uganda ("foot and mouth" OR FMD OR anthrax OR "lumpy skin" OR CBPP OR "East Coast Fever" '
            'OR "Rift Valley" OR rabies OR "swine fever" OR "bird flu" OR quarantine OR "livestock vaccination" '
            'OR MAAIF) when:60d')

SOURCES: List[Dict[str, str]] = [
    {"id": "maaif", "name": "MAAIF (agriculture.go.ug)", "kind": "official",
     "url": "https://www.agriculture.go.ug/feed/"},
    {"id": "gnews", "name": "Google News: Uganda livestock disease", "kind": "news",
     "url": "https://news.google.com/rss/search?" + urllib.parse.urlencode(
         {"q": _GNEWS_Q, "hl": "en-UG", "gl": "UG", "ceid": "UG:en"})},
    {"id": "allafrica", "name": "allAfrica: Uganda", "kind": "news",
     "url": "https://allafrica.com/tools/headlines/rdf/uganda/headlines.rdf"},
]

# A notice must mention one of these (title or summary).
_TOPIC = re.compile(
    r"foot[- ]and[- ]mouth|\bFMD\b|anthrax|lumpy skin|\bCBPP\b|contagious bovine|east coast fever|\bECF\b|"
    r"rift valley|rabies|brucell|peste des petits|\bPPR\b|swine fever|\bASF\b|avian influenza|bird flu|"
    r"newcastle disease|trypanosom|sleeping sickness|tsetse|acaricide|\bticks?\b|quarantine|"
    r"vaccinat\w*.{0,40}(cattle|livestock|animals|goats|pigs|poultry|dogs)|(cattle|livestock|animal).{0,40}vaccinat|"
    r"livestock (market|movement)s?|cattle market|animal movement|disease outbreak.{0,40}(cattle|livestock|animals)|"
    r"(cattle|livestock|animals).{0,40}outbreak",
    re.I)
# Human-only health stories that share words with the topic list.
_EXCLUDE = re.compile(r"\bebola\b|\bmpox\b|\bcholera\b|\bmalaria\b|\bpolio\b|\bmeasles\b|covid", re.I)

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
    ("quarantine", re.compile(r"quarantine|ban on (livestock|cattle|animal) movement|markets? (closed|closure)|clos\w* .{0,20}markets?", re.I)),
    ("vaccination", re.compile(r"vaccinat", re.I)),
    ("outbreak", re.compile(r"outbreak|confirm\w* (case|cases)|deaths? of (cattle|animals)|spread", re.I)),
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
    if not _TOPIC.search(text) or (_EXCLUDE.search(title) and not re.search(r"cattle|livestock|animal", title, re.I)):
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


def build_notices(feeds: Dict[str, str], now: Optional[dt.datetime] = None) -> List[dict]:
    """feeds: {source_id: xml_text}. Returns filtered, de-duplicated notices, newest first."""
    now = now or dt.datetime.now(dt.timezone.utc)
    src = {s["id"]: s for s in SOURCES}
    seen, out = set(), []
    for sid, xml_text in feeds.items():
        meta = src.get(sid, {"id": sid, "name": sid, "kind": "news"})
        for it in parse_feed(xml_text):
            if not it["title"] or not it["link"].startswith("http"):
                continue
            pub = it["published"]
            if pub is None or (now - pub).days > MAX_AGE_DAYS or pub > now + dt.timedelta(days=1):
                continue
            title, publisher = _clean_title(it["title"], it["publisher"])
            c = classify(title, it["summary"] if meta["kind"] == "official" else "")
            if c is None:
                continue
            key = re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()
            if key in seen:
                continue
            seen.add(key)
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
    notices = build_notices(feeds)
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
    sig = lambda d: json.dumps([d.get("notices"), [(x["id"], x["ok"]) for x in d.get("sources", [])]], sort_keys=True)
    if previous and sig(previous) == sig(data):
        print("unchanged: notices and source status are the same as last time")
        return 0
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    ok = sum(s["ok"] for s in data["sources"])
    print(f"{len(data['notices'])} notices from {ok}/{len(data['sources'])} sources" + (" (kept previous)" if data["kept_previous"] else ""))
    for s in data["sources"]:
        print(f"  {'ok ' if s['ok'] else 'ERR'} {s['name']}: {s.get('items_read', s.get('error'))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
