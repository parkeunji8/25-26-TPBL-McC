from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

GAMES_CSV = Path("input_data/games.csv")               # TPBL processed 데이터의 games.csv
PLAYER_LOGS_CSV = Path("input_data/player_game_logs.csv")
API_DIR = Path("raw_api")
TEXT_DIR = Path("raw_text")
MANIFEST = API_DIR / "manifest.csv"

LEOPARDS_TEAM_ID = 5
MCC_PLAYER_ID = "10860"
PAGE_URL = "https://tpbl.basketball/schedule/{gid}/detail"
API_HOST = "api.tpbl.basketball"
# 상세 페이지에서 눌러볼 탭들 (없으면 건너뜀). 탭을 눌러야 추가 API를 호출하는 경우 대비
TAB_LABELS = ["比賽詳情", "比賽數據", "球賽總覽", "投籃分布", "出手分布"]
MIN_INTERVAL_SEC = 1.5


# ─────────────────────────── 경기 ID 선택 ───────────────────────────
def select_game_ids(scope: str) -> list[str]:
    g = pd.read_csv(GAMES_CSV, dtype={"game_id": str})
    if scope == "all":
        ids = g.game_id
    elif scope == "leopards":
        ids = g[(g.home_team_id == LEOPARDS_TEAM_ID) | (g.away_team_id == LEOPARDS_TEAM_ID)].game_id
    elif scope == "mcc":
        pg = pd.read_csv(PLAYER_LOGS_CSV, dtype={"game_id": str, "player_id": str})
        ids = pg[(pg.player_id == MCC_PLAYER_ID) & (pg.appearance == True)].game_id
    else:
        raise ValueError(scope)
    return sorted(set(ids), key=int)


# ─────────────────────────── 저장 유틸 ───────────────────────────
def endpoint_name(url: str, gid: str) -> str:
    """https://api.tpbl.basketball/api/games/1375/broadcasts?x=1 → games_broadcasts"""
    path = re.sub(r"^https?://[^/]+/api/", "", url.split("?")[0])
    path = path.replace(f"/{gid}", "").replace(f"{gid}/", "")
    return re.sub(r"[^A-Za-z0-9_-]+", "_", path).strip("_") or "root"


def append_manifest(row: dict):
    new = not MANIFEST.exists()
    with MANIFEST.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


# ─────────────────────────── 크롤링 ───────────────────────────
def crawl_one(page, gid: str) -> dict:
    out_dir = API_DIR / gid
    out_dir.mkdir(parents=True, exist_ok=True)
    captured: dict[str, int] = {}

    def on_response(resp):
        if API_HOST not in resp.url:
            return
        try:
            if "json" not in (resp.headers.get("content-type") or ""):
                return
            body = resp.body()
        except Exception:
            return  # 리다이렉트/중단된 응답
        name = endpoint_name(resp.url, gid)
        # 같은 엔드포인트가 여러 번 오면 번호를 붙여 모두 보존
        k = captured.get(name, 0)
        fname = f"{name}.json" if k == 0 else f"{name}__{k}.json"
        captured[name] = k + 1
        (out_dir / fname).write_bytes(body)
        append_manifest({
            "game_id": gid, "file": fname, "url": resp.url, "status": resp.status,
            "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest(),
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
        })

    page.on("response", on_response)
    try:
        page.goto(PAGE_URL.format(gid=gid), wait_until="networkidle", timeout=60_000)
        for label in TAB_LABELS:
            loc = page.get_by_text(label, exact=True)
            if loc.count() > 0:
                try:
                    loc.first.click(timeout=3_000)
                    page.wait_for_load_state("networkidle", timeout=15_000)
                    page.wait_for_timeout(1_000)
                except Exception:
                    pass
        # 화면 텍스트 백업 (번역하지 않은 원문)
        TEXT_DIR.mkdir(exist_ok=True)
        (TEXT_DIR / f"game_{gid}.txt").write_text(page.inner_text("body"), encoding="utf-8")
    finally:
        page.remove_listener("response", on_response)

    (out_dir / "_done.json").write_text(json.dumps(captured, ensure_ascii=False, indent=2))
    return captured


def crawl(ids: list[str], force: bool = False):
    from playwright.sync_api import sync_playwright
    API_DIR.mkdir(exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        last = 0.0
        for i, gid in enumerate(ids, 1):
            if (API_DIR / gid / "_done.json").exists() and not force:
                print(f"[{i}/{len(ids)}] {gid} 이미 수집됨, 건너뜀")
                continue
            wait = MIN_INTERVAL_SEC - (time.time() - last)
            if wait > 0:
                time.sleep(wait)
            last = time.time()
            for attempt in range(3):
                try:
                    cap = crawl_one(page, gid)
                    print(f"[{i}/{len(ids)}] {gid} 저장: {cap}")
                    if not any("broadcast" in k for k in cap):
                        print(f"   ⚠ broadcasts 응답이 안 잡힘 → 탭 이름/대기시간 확인 필요")
                    break
                except Exception as ex:
                    print(f"[{i}/{len(ids)}] {gid} 실패({attempt + 1}/3): {ex}")
                    time.sleep(3 * (attempt + 1))
        browser.close()


# ─────────────────────────── 구조 확인 ───────────────────────────
def describe(obj, depth=0, max_depth=4, key="root"):
    pad = "  " * depth
    if isinstance(obj, dict):
        print(f"{pad}{key}: dict({len(obj)}) keys={list(obj)[:25]}")
        if depth < max_depth:
            for k, v in list(obj.items())[:25]:
                if isinstance(v, (dict, list)):
                    describe(v, depth + 1, max_depth, k)
    elif isinstance(obj, list):
        print(f"{pad}{key}: list({len(obj)})")
        if obj and depth < max_depth:
            describe(obj[0], depth + 1, max_depth, f"{key}[0]")


def inspect(gid: str):
    d = API_DIR / gid
    for f in sorted(d.glob("*.json")):
        if f.name == "_done.json":
            continue
        print(f"\n===== {f.name} ({f.stat().st_size:,} bytes)")
        try:
            describe(json.loads(f.read_text(encoding="utf-8")))
        except Exception as ex:
            print("  JSON 파싱 실패:", ex)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", choices=["leopards", "mcc", "all"])
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--inspect")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.inspect:
        inspect(a.inspect)
    else:
        ids = a.ids or select_game_ids(a.scope or "leopards")
        print(f"대상 {len(ids)}경기")
        crawl(ids, a.force)
