import json
from pathlib import Path

from playwright.sync_api import sync_playwright

TEAM_NAME = "桃園台啤永豐雲豹" # 맥컬러 팀명
ID_START = 1371 # GAME 1
ID_END = 1496 # GAME 126 (25-26 시즌 정규시즌 총 경기 수)
ID_LIST_PATH = Path("leopards_25_26_game_ids.json")


def check_leopards(page, game_id: int):
    """이 game_id 페이지 제목에 桃園台啤永豐雲豹가 들어있는지 확인.
    (True/False, 페이지 제목) 튜플을 반환. 존재하지 않는 id면 (False, None)."""
    url = f"https://tpbl.basketball/schedule/{game_id}/detail"
    resp = page.goto(url, wait_until="domcontentloaded")

    if resp is None or resp.status >= 400:
        return False, None

    title = page.title()
    # 타이틀 예: "GAME 14 2025-11-02 高雄全家海神 vs 桃園台啤永豐雲豹 | 台灣職業籃球大聯盟"
    if not title or "GAME" not in title:
        return False, title

    matched = TEAM_NAME in title
    return matched, title


def main():
    matched_ids = []
    if ID_LIST_PATH.exists():
        print(f"[알림] {ID_LIST_PATH} 가 이미 있어서 이어서 사용합니다.")
        matched_ids = json.loads(ID_LIST_PATH.read_text(encoding="utf-8"))

    total = ID_END - ID_START + 1
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()

        for i, gid in enumerate(range(ID_START, ID_END + 1), 1):
            if gid in matched_ids:
                print(f"  [{i}/{total}] id={gid} 이미 확인됨, 건너뜀")
                continue
            try:
                matched, title = check_leopards(page, gid)
            except Exception as ex:
                print(f"  [{i}/{total}] id={gid} 확인 실패: {ex}")
                continue

            status = "✅ 일치" if matched else "—"
            print(f"  [{i}/{total}] id={gid} {status}  ({title})")

            if matched:
                matched_ids.append(gid)
                ID_LIST_PATH.write_text(
                    json.dumps(sorted(matched_ids), ensure_ascii=False, indent=2),
                    encoding="utf-8"
                )

        browser.close()

    print(f"\n완료! Leopards 25-26 시즌 경기 {len(matched_ids)}개 발견")
    print(f"목록 저장 위치: {ID_LIST_PATH}")
    if len(matched_ids) != 36:
        print(f"[참고] 보통 한 팀은 정규시즌 36경기를 뛰는데 {len(matched_ids)}개가 나왔어요. "
              f"ID_START/ID_END 범위를 조정해야 할 수도 있어요.")


if __name__ == "__main__":
    main()