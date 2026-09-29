
import json
import time
from pathlib import Path
 
from playwright.sync_api import sync_playwright
 
ID_LIST_PATH = Path("leopards_25_26_game_ids.json")
OUT_DIR = Path("raw_games")
 
 
def crawl_one(page, game_id: int) -> str:
    url = f"https://tpbl.basketball/schedule/{game_id}/detail"
    page.goto(url, wait_until="networkidle")
    page.click("text=比賽詳情")
    page.wait_for_timeout(2000)  # 로그가 다 그려질 때까지 대기
    return page.inner_text("body")
 
 
def main():
    if not ID_LIST_PATH.exists():
        print(f"[오류] {ID_LIST_PATH} 없음")
        return
 
    game_ids = json.loads(ID_LIST_PATH.read_text(encoding="utf-8"))
    print(f"총 {len(game_ids)}개 경기 크롤링 시작")
 
    OUT_DIR.mkdir(exist_ok=True)
 
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
 
        for i, gid in enumerate(game_ids, 1):
            out_path = OUT_DIR / f"game_{gid}.txt"
            if out_path.exists():
                print(f"  [{i}/{len(game_ids)}] id={gid} 이미 존재, 건너뜀")
                continue
            try:
                text = crawl_one(page, gid)
                out_path.write_text(text, encoding="utf-8")
                print(f"  [{i}/{len(game_ids)}] id={gid} 저장 완료 ({len(text)}자)")
            except Exception as ex:
                print(f"  [{i}/{len(game_ids)}] id={gid} 크롤링 실패: {ex}")
            time.sleep(1)
 
        browser.close()
 
    saved = list(OUT_DIR.glob("game_*.txt"))
    print(f"\n{OUT_DIR}/ 폴더에 {len(saved)}개 파일 저장 완료")
 
 
if __name__ == "__main__":
    main()
 
