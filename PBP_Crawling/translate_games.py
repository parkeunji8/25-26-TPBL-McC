import json
import re
from pathlib import Path

RAW_DIR = Path("raw_games_mcc")
OUT_DIR = Path("translated_games")
GLOSSARY_PATH = Path("name_glossary_eng.json")

def load_name_map():
    raw = json.loads(GLOSSARY_PATH.read_text(encoding="utf-8"))
    return {k: v for k, v in raw.items() if v and not v.startswith("__REMOVE__")}


PHRASE_MAP = {
    # 교체
    "出場": "교체 출전",
    "離場": "교체 아웃",

    # 기본 스탯 이벤트
    "助攻": "어시스트",
    "阻攻": "블록",
    "抄截": "스틸",
    "犯規": "파울",
    "防守籃板": "수비 리바운드",
    "進攻籃板": "공격 리바운드",
    "籃板": "리바운드",
    "失誤": "턴오버",
    "得分": "득점",
    "暫停": "타임아웃",
    "一般暫停": "일반 타임아웃",
    "各節得分": "쿼터별 득점",

    # 슛 종류 (구체적인 것부터)
    "運球上籃": "드리블 레이업",
    "後仰跳投": "페이드어웨이 점프슛",
    "轉身跳投": "턴어라운드 점프슛",
    "急停跳投": "풀업 점프슛",
    "後撤步跳投": "스텝백 점프슛",
    "空中接力": "앨리웁",
    "跳投": "점프슛",
    "上籃": "레이업",
    "灌籃": "덩크",
    "拋投": "플로터",
    "補籃": "팁인",
    "兩分球": "2점슛",
    "三分球": "3점슛",
    "罰球": "자유투",

    # 결과
    "出手得分": "슛 성공",
    "出手失手": "슛 실패",
    "進算加一": "성공+추가자유투(and-1)",

    # 턴오버 세부
    "傳球失誤": "패스 미스",
    "控球失誤": "볼 핸들링 미스",
    "出界失誤": "아웃오브바운즈",
    "走步違例": "트래블링",
    "籃下3秒違例": "3초 룰 위반",
    "團隊24秒違例": "24초 위반",

    # 파울 세부
    "球員防守個人犯規": "수비 파울",
    "球員進攻個人犯規": "공격자 파울",
    "球員防守違反運動精神犯規": "스포츠맨십 위반 파울(수비)",
    "球員進攻技術犯規": "테크니컬 파울(공격)",
    "進攻犯規": "오펜스 파울",

    # 팀 이벤트
    "團隊防守籃板": "팀 수비 리바운드",
    "團隊進攻籃板": "팀 공격 리바운드",

    # 헤더/기타
    "整場比賽": "경기 전체",
    "球賽總覽": "경기 개요",
    "比賽數據": "경기 데이터",
    "比賽詳情": "경기 상세",
    "隊伍": "팀",
    "Final": "최종",

    # 팀 이름 (네비게이션 메뉴 등에 등장)
    "桃園台啤永豐雲豹": "Taoyuan Leopards",
    "高雄全家海神": "Kaohsiung Aquas",
    "臺北台新戰神": "Taipei Mars",
    "新北中信特攻": "New Taipei DEA",
    "新北國王": "New Taipei Kings",
    "新竹御嵿攻城獅": "Hsinchu Lioneers",
    "福爾摩沙夢想家": "Formosa Dreamers",

    # 사이트 메뉴/내비게이션 (참고용 - pbp 로그와 무관)
    "關於聯盟": "리그 소개",
    "聯盟團隊": "리그 팀",
    "聯盟賽務": "리그 운영",
    "賽務規章": "운영 규정",
    "選秀辦法": "드래프트 규정",
    "選租名單": "드래프트 명단",
    "選秀名單": "드래프트 명단",
    "選秀": "드래프트",
    "球隊數據": "팀 데이터",
    "球員數據": "선수 데이터",
    "數據排行": "데이터 순위",
    "特殊表現": "특별 기록",
    "歷史紀錄": "역사 기록",
    "聯盟快訊": "리그 소식",
    "TPBL 公告": "TPBL 공지",
    "最新消息": "최신 소식",
    "創作者計畫": "크리에이터 프로그램",
    "訂閱電子報": "뉴스레터 구독",
    "排序：新到舊": "정렬: 최신순",
    "球隊": "팀",
    "賽程": "일정",
    "獎項": "시상",
    "購票": "티켓 구매",
}

_SORTED_PHRASES = sorted(PHRASE_MAP.items(), key=lambda kv: -len(kv[0]))
QUARTER_RE = re.compile(r"第(\d)節(開始|結束)")


def translate_quarter_markers(text: str) -> str:
    def repl(m):
        num, phase = m.group(1), m.group(2)
        phase_ko = "시작" if phase == "開始" else "종료"
        return f"{num}쿼터 {phase_ko}"
    return QUARTER_RE.sub(repl, text)


def replace_names(text: str, name_map: dict) -> str:
    # 이름이 길수록 먼저 치환 (부분 문자열 충돌 방지)
    for cn, en in sorted(name_map.items(), key=lambda kv: -len(kv[0])):
        if cn in text:
            text = text.replace(cn, en)
    return text


def translate_phrases(text: str) -> str:
    for cn, ko in _SORTED_PHRASES:
        if cn in text:
            # 뒤에 공백을 붙여서 여러 조각이 이어붙는 걸 방지
            text = text.replace(cn, ko + " ")
    # 중복 공백 정리
    text = re.sub(r" {2,}", " ", text).strip()
    return text


def translate_line(line: str, name_map: dict) -> str:
    line = translate_quarter_markers(line)
    line = replace_names(line, name_map)
    line = translate_phrases(line)
    return line


def translate_file(text: str, name_map: dict) -> str:
    lines = text.splitlines()
    return "\n".join(translate_line(ln, name_map) for ln in lines)


def main():
    if not RAW_DIR.exists():
        print(f"[오류] {RAW_DIR} 폴더가 없어요.")
        return
    if not GLOSSARY_PATH.exists():
        print(f"[오류] {GLOSSARY_PATH} 가 없어요.")
        return

    name_map = load_name_map()
    print(f"이름 사전 {len(name_map)}개 로드 완료.")

    files = sorted(RAW_DIR.glob("game_*.txt"))
    if not files:
        print(f"[오류] {RAW_DIR} 안에 game_*.txt 파일이 없어요.")
        return

    OUT_DIR.mkdir(exist_ok=True)

    for i, f in enumerate(files, 1):
        raw = f.read_text(encoding="utf-8")
        translated = translate_file(raw, name_map)
        out_path = OUT_DIR / f"{f.stem}_ko.txt"
        out_path.write_text(translated, encoding="utf-8")
        print(f"  [{i}/{len(files)}] {f.name} -> {out_path.name}")

    print(f"\n완료! {OUT_DIR}/ 에 {len(files)}개 번역 파일이 저장되었습니다.")
    print("한자가 그대로 남아있는 부분이 있다면, 사전에 없는 표현일 수 있어요.")
    print("그런 줄을 찾아서 알려주시면 사전에 추가해드릴게요.")


if __name__ == "__main__":
    main()