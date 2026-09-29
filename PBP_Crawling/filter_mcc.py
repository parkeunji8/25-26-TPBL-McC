import json
import re
import shutil
from pathlib import Path

RAW_DIR = Path("raw_games")
DST_DIR = Path("raw_games_mcc")
OUT_PATH = Path("mcc_game_ids.json")

PLAYER_NAME = "麥卡洛"


def played_in(text: str) -> bool:
    """이 경기 텍스트 안에 맥컬러가 코트에 들어온 기록이 있는지 확인."""
    return f"{PLAYER_NAME} 出場" in text


def main():
    if not RAW_DIR.exists():
        print(f"[오류] {RAW_DIR} 폴더가 없어요.")
        return

    files = sorted(
        RAW_DIR.glob("game_*.txt"),
        key=lambda p: int(re.search(r"\d+", p.stem).group())
    )
    if not files:
        print(f"[오류] {RAW_DIR} 안에 game_*.txt 파일이 없어요.")
        return

    played_ids = []
    absent_ids = []

    for f in files:
        gid = int(re.search(r"\d+", f.stem).group())
        text = f.read_text(encoding="utf-8")
        if played_in(text):
            played_ids.append(gid)
        else:
            absent_ids.append(gid)

    OUT_PATH.write_text(
        json.dumps(played_ids, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    # 걸러낸 경기 파일들을 별도 폴더로 복사
    DST_DIR.mkdir(exist_ok=True)
    copied = 0
    for gid in played_ids:
        src = RAW_DIR / f"game_{gid}.txt"
        dst = DST_DIR / f"game_{gid}.txt"
        shutil.copy2(src, dst)
        copied += 1

    print(f"전체 {len(files)}경기 중:")
    print(f"  맥컬러 출전: {len(played_ids)}경기 -> {played_ids}")
    print(f"  맥컬러 결장: {len(absent_ids)}경기 -> {absent_ids}")
    print(f"\nid 목록 저장 위치: {OUT_PATH}")
    print(f"파일 복사 위치: {DST_DIR}/ ({copied}개, 원본 {RAW_DIR}/ 는 그대로 유지)")

    if len(played_ids) != 24:
        print(f"\n[참고] 24경기를 예상했는데 {len(played_ids)}개가 나왔어요.")
        print("결장 경기로 분류된 id들의 원본 파일을 한번 열어서,")
        print("정말 麥卡洛가 안 나오는지 직접 확인해보시면 좋을 것 같아요")
        print("(혹시 이름 표기가 다르게 찍힌 경기가 있을 수도 있어서요).")


if __name__ == "__main__":
    main()