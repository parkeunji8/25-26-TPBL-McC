import json
import re
from pathlib import Path

RAW_DIR = Path("raw_games_mcc")
OUT_PATH = Path("name_eng.json")

ACTION_KEYWORDS = [
    "出場", "離場", "助攻", "阻攻", "抄截", "犯規",
    "籃板", "失誤", "得分", "暫停",
]

TEAM_EVENT_PREFIXES = ("暫停", "失誤", "團隊防守籃板", "團隊進攻籃板")


def extract_names_from_text(text: str) -> set[str]:
    names = set()
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if not any(kw in line for kw in ACTION_KEYWORDS):
            continue
        head = line.split(" ")[0].split("-")[0].strip()
        if head.startswith(TEAM_EVENT_PREFIXES) or "團隊" in line[:6]:
            continue
        if 2 <= len(head) <= 6 and not re.search(r"[0-9a-zA-Z]", head):
            names.add(head)
    return names


def main():
    if not RAW_DIR.exists():
        print(f"[오류] {RAW_DIR} 폴더가 없음")
        return

    files = sorted(RAW_DIR.glob("game_*.txt"))
    if not files:
        print(f"[오류] {RAW_DIR} 안에 game_*.txt 파일이 없어요.")
        return

    print(f"{len(files)}개 경기 파일에서 이름 추출 중...")

    all_names = set()
    for f in files:
        text = f.read_text(encoding="utf-8")
        names = extract_names_from_text(text)
        all_names |= names

    glossary = {name: "" for name in sorted(all_names)}

    # 이미 있는 용어집이 있으면, 기존에 채워둔 번역은 보존하고 새 이름만 추가
    if OUT_PATH.exists():
        existing = json.loads(OUT_PATH.read_text(encoding="utf-8"))
        for name, translated in existing.items():
            if name in glossary and translated:
                glossary[name] = translated
        for name, translated in existing.items():
            if name not in glossary:
                glossary[name] = translated

    OUT_PATH.write_text(
        json.dumps(glossary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8"
    )

    filled = sum(1 for v in glossary.values() if v)
    print(f"\n완료! 총 {len(glossary)}개 이름 발견 (그중 {filled}개는 이미 번역됨)")
    print(f"저장 위치: {OUT_PATH}")
    print("\n다음 단계: name_eng.json을 열어서 값(영어 표기)을 채워주세요.")


if __name__ == "__main__":
    main()