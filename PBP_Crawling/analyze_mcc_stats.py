import json
import re
import csv
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict

TRANSLATED_DIR = Path("translated_games")
GLOSSARY_PATH = Path("name_glossary_eng.json")
TARGET_PLAYER = "Chris McCullough"
QUARTER_LEN_SEC = 12 * 60

TIME_RE = re.compile(r"^(--:--|\d{1,2}:\d{2})$")
SCORE_RE = re.compile(r"^(\d+)\s*-\s*(\d+)$")
DELTA_RE = re.compile(r"^\+(\d+)$")
QUARTER_RE = re.compile(r"^(\d)쿼터\s*(시작|종료)$")

LEOPARDS_ROSTER = {
    "Chris McCullough", "Malcolm Miller", "Cheick Diallo", "Lasan Kromah",
    "Gao Jin-Wei", "Lin Sin-Kuan", "Chuang Po-Yuan", "Hsu Hong-Wei",
    "Wang Jhe-Yu", "Ting Kuang-Hao", "Wang Hao-Chi", "Huang Jhen",
    "Tung Yung-Chuan", "Tsao Xun-Xiang", "Liu Yuan-Kai",
}


def load_names():
    if not GLOSSARY_PATH.exists():
        return []
    raw = json.loads(GLOSSARY_PATH.read_text(encoding="utf-8"))
    names = {v for v in raw.values() if v and not v.startswith("__REMOVE__")}
    return sorted(names, key=lambda s: -len(s))


def extract_player_action(line: str, names):
    for name in names:
        if line == name or line.startswith(name + " "):
            return name, line[len(name):].strip()
    return None, line


@dataclass
class Event:
    seq: int
    quarter: int
    clock: str
    elapsed: int
    player: str
    text: str
    points: int = 0


def clock_to_seconds(clock: str) -> int:
    if not clock or clock == "--:--":
        return 0
    try:
        m, s = clock.split(":")
        return int(m) * 60 + int(s)
    except ValueError:
        return 0

def format_clock(quarter: int, clock: str) -> str:
    if clock == "--:--":
        return f"Q{quarter} 00:00(종료)"
    return f"Q{quarter} {clock}"


def parse_translated_text(text: str, names):
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    raw_records = []
    current = None

    for ln in lines:
        qm = QUARTER_RE.match(ln)
        if qm:
            raw_records.append({"type": "QUARTER", "quarter": int(qm.group(1)), "phase": qm.group(2)})
            continue
        if TIME_RE.match(ln):
            current = {"type": "EVENT", "clock": ln, "texts": [], "delta": 0}
            raw_records.append(current)
            continue
        sm = SCORE_RE.match(ln)
        if sm and current is not None:
            continue
        dm = DELTA_RE.match(ln)
        if dm and current is not None:
            current["delta"] = int(dm.group(1))
            continue
        if current is not None:
            current["texts"].append(ln)

    reversed_records = list(reversed(raw_records))
    events = []
    seq = 0
    current_quarter = 1
    elapsed_base = 0

    for rec in reversed_records:
        if rec["type"] == "QUARTER":
            if rec["phase"] == "시작":
                current_quarter = rec["quarter"]
                elapsed_base = (current_quarter - 1) * QUARTER_LEN_SEC
            continue
        clock_sec = clock_to_seconds(rec["clock"])
        elapsed = elapsed_base + (QUARTER_LEN_SEC - clock_sec)
        for txt in rec["texts"]:
            player, action_text = extract_player_action(txt, names)
            points = rec["delta"] if ("슛 성공" in action_text or "성공+추가자유투" in action_text) else 0
            events.append(Event(seq, current_quarter, rec["clock"], elapsed, player, action_text, points))
            seq += 1
    return events


@dataclass
class Stint:
    q_in: int
    clock_in: str
    elapsed_in: int
    q_out: int
    clock_out: str
    elapsed_out: int

    def time_str_in(self) -> str:
        return format_clock(self.q_in, self.clock_in)

    def time_str_out(self) -> str:
        return format_clock(self.q_out, self.clock_out)


def find_stints(events, target=TARGET_PLAYER):
    stints = []
    first_event = next((e for e in events if e.player == target), None)
    is_starter = False
    if first_event and "교체 출전" not in first_event.text:
        is_starter = True

    on_court = is_starter
    cur = {"q_in": 1, "clock_in": "12:00", "elapsed_in": 0} if is_starter else None

    for e in events:
        if e.player == target and "교체 출전" in e.text and not on_court:
            on_court = True
            cur = {"q_in": e.quarter, "clock_in": e.clock, "elapsed_in": e.elapsed}
        elif e.player == target and "교체 아웃" in e.text and on_court:
            on_court = False
            stints.append(Stint(cur["q_in"], cur["clock_in"], cur["elapsed_in"], e.quarter, e.clock, e.elapsed))
            cur = None

    if on_court and cur and events:
        last = events[-1]
        stints.append(Stint(cur["q_in"], cur["clock_in"], cur["elapsed_in"], last.quarter, last.clock, last.elapsed))

    return stints


def team_of(player: str) -> str:
    return "우리팀" if player in LEOPARDS_ROSTER else "상대팀"

def elapsed_to_qc(elapsed: int):
    """누적 경과초 -> (쿼터, 시계) 로 역변환"""
    quarter = elapsed // QUARTER_LEN_SEC + 1
    remainder = elapsed % QUARTER_LEN_SEC
    clock_sec = QUARTER_LEN_SEC - remainder
    if clock_sec >= QUARTER_LEN_SEC:
        clock = "12:00"
    elif clock_sec <= 0:
        clock = "00:00"
    else:
        m, s = divmod(clock_sec, 60)
        clock = f"{m}:{s:02d}"
    return quarter, clock


def team_score_in_range(events, start: int, end: int) -> int:
    """[start, end] 구간 동안의 팀 스코어(우리팀 득점 - 상대팀 득점)"""
    us_pts, opp_pts = 0, 0
    for e in events:
        if start <= e.elapsed <= end and e.points > 0 and e.player:
            if team_of(e.player) == "우리팀":
                us_pts += e.points
            else:
                opp_pts += e.points
    return us_pts - opp_pts

def build_on_court_intervals(events):

    on_court = {"우리팀": set(), "상대팀": set()}
    change_points = []
    i, n = 0, len(events)

    while i < n:
        t0 = events[i].elapsed
        batch = []
        j = i
        while j < n and events[j].elapsed == t0:
            batch.append(events[j])
            j += 1

        changed = False
        for e in batch:
            if not e.player:
                continue
            if "교체 출전" in e.text:
                on_court[team_of(e.player)].add(e.player)
                changed = True
            elif "교체 아웃" in e.text:
                on_court[team_of(e.player)].discard(e.player)
                changed = True

        if changed or not change_points:
            change_points.append((t0, frozenset(on_court["우리팀"]), frozenset(on_court["상대팀"])))
        i = j

    intervals = []
    for k in range(len(change_points)):
        start = change_points[k][0]
        end = change_points[k + 1][0] if k + 1 < len(change_points) else events[-1].elapsed
        intervals.append((start, end, change_points[k][1], change_points[k][2]))
    return intervals

def new_stat_row():
    return {
        "PTS": 0, "REB": 0, "AST": 0, "STL": 0, "BLK": 0, "TOV": 0, "PF": 0,
        "2P_M": 0, "2P_A": 0, "3P_M": 0, "3P_A": 0, "FT_M": 0, "FT_A": 0,
    }


def classify_and_accumulate(e: Event, player_stats: dict):
    if not e.player:
        return
    row = player_stats[e.player]
    t = e.text

    if "교체" in t:
        return
    if "어시스트" in t:
        row["AST"] += 1
        return
    if "리바운드" in t:
        row["REB"] += 1
        return
    if "스틸" in t:
        row["STL"] += 1
        return
    if "블록" in t:
        row["BLK"] += 1
        return
    if "턴오버" in t:
        row["TOV"] += 1
        return
    if "파울" in t:
        row["PF"] += 1
        return

    is_made = "슛 성공" in t or "성공+추가자유투" in t
    is_missed = "슛 실패" in t

    if is_made or is_missed:
        if "자유투" in t:
            row["FT_A"] += 1
            if is_made:
                row["FT_M"] += 1
                row["PTS"] += (e.points if e.points > 0 else 1)
        elif "3점슛" in t:
            row["3P_A"] += 1
            if is_made:
                row["3P_M"] += 1
                row["PTS"] += (e.points if e.points > 0 else 3)
        elif "2점슛" in t:
            row["2P_A"] += 1
            if is_made:
                row["2P_M"] += 1
                row["PTS"] += (e.points if e.points > 0 else 2)


def main():
    names = load_names()
    files = sorted(TRANSLATED_DIR.glob("game_*_ko.txt"))
    
    if not files:
        print(f"[오류] {TRANSLATED_DIR} 안에 game_*_ko.txt 파일이 없어요.")
        return

    # 온코트 구간별(Stint) 스탯을 저장할 리스트
    stint_records_us = []
    stint_records_opp = []
    co_occur_seconds = defaultdict(int)   # 팀 동료별 맥컬러와의 공존 시간(초)
    co_occur_team_score = defaultdict(int)   # 팀 동료별 팀 스코어 누적
    lineup_stint_records = []                 

    for f in files:

        text = f.read_text(encoding="utf-8")

        date_match = re.search(r"(\d{1,2})/(\d{1,2})\s*[（(].[）)]", text)

        if date_match:
            month = int(date_match.group(1))
            day = int(date_match.group(2))

            # 2. 10~12월은 2025년, 1~5월은 2026년으로 연도 지정
            year = 2025 if month >= 10 else 2026

            game_date = f"{year % 100:02d}/{month:02d}/{day:02d}"
        else:
            # 패턴을 못 찾을 경우 기본값
            game_date = "Unknown"

        events = parse_translated_text(text, names)
        stints = find_stints(events)

        # --- [복원 1] 맥컬러 온코트 구간별 우리팀/상대팀 스탯을 stint_records_us/opp에 채움 ---
        for stint_idx, s in enumerate(stints, start=1):
            window = [e for e in events if s.elapsed_in <= e.elapsed <= s.elapsed_out]
            player_rows = {"우리팀": defaultdict(new_stat_row), "상대팀": defaultdict(new_stat_row)}
            for e in window:
                if not e.player:
                    continue
                team = team_of(e.player)
                classify_and_accumulate(e, player_rows[team])
            for team in ("우리팀", "상대팀"):
                target_list = stint_records_us if team == "우리팀" else stint_records_opp
                for p, row in player_rows[team].items():
                    target_list.append({
                        "Date": game_date,
                        "OnCourt_Seq": stint_idx,
                        "In_Time": s.time_str_in(),
                        "Out_Time": s.time_str_out(),
                        "Player": p,
                        **row
                    })
        # --- [복원 1] 끝 ---

                # --- 라인업 공존 시간 + 팀 스코어 계산 ---
        intervals = build_on_court_intervals(events)
        stint_seq_in_game = 0
        for start, end, us_set, opp_set in intervals:
            if TARGET_PLAYER not in us_set:
                continue
            duration = end - start
            teammates = us_set - {TARGET_PLAYER}
            score = team_score_in_range(events, start, end)

            for tm in teammates:
                co_occur_seconds[tm] += duration
                co_occur_team_score[tm] += score

            stint_seq_in_game += 1
            q_in, c_in = elapsed_to_qc(start)
            q_out, c_out = elapsed_to_qc(end)
            lineup_stint_records.append({
                "Date": game_date,
                "Stint_Seq": stint_seq_in_game,
                "In_Time": f"Q{q_in} {c_in}",
                "Out_Time": f"Q{q_out} {c_out}",
                "Duration_Min": round(duration / 60, 1),
                "Teammates": ", ".join(sorted(teammates)),
                "Opponents": ", ".join(sorted(opp_set)),
                "Team_Score": score,
            })

    headers = [
    "Date", "OnCourt_Seq", "In_Time", "Out_Time", "Player",
    "PTS", "REB", "AST", "STL", "BLK", "TOV", "PF",
    "2P_M", "2P_A", "3P_M", "3P_A", "FT_M", "FT_A"]

    # CSV 추출
    def write_csv(path: Path, records: list):
        with path.open("w", newline="", encoding="utf-8-sig") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            writer.writerows(records)

    output_path_us = Path("mcc_oncourt_stats_우리팀.csv")
    output_path_opp = Path("mcc_oncourt_stats_상대팀.csv")

    write_csv(output_path_us, stint_records_us)
    write_csv(output_path_opp, stint_records_opp)

    print(f"[완료] 우리팀 스탯: {output_path_us}")
    print(f"[완료] 상대팀 스탯: {output_path_opp}")

        # --- 팀 동료 적합도용 집계: 선수 1명당 1행, 공존시간 + 비율 스탯 ---
    player_totals = defaultdict(new_stat_row)
    for row in stint_records_us:
        p = row["Player"]
        for k in new_stat_row().keys():
            player_totals[p][k] += row[k]

    fit_headers = [
        "Player", "CoOccur_Min", "Team_Score",
        "PTS", "REB", "AST", "STL", "BLK", "TOV", "PF",
        "2P_M", "2P_A", "3P_M", "3P_A", "FT_M", "FT_A",
        "PTS_per36", "REB_per36", "AST_per36", "STL_per36",
        "BLK_per36", "TOV_per36", "PF_per36",
        "2P_PCT", "3P_PCT", "FT_PCT",
    ]

    fit_rows = []
    for p, stat in player_totals.items():
        co_min = co_occur_seconds.get(p, 0) / 60
        if co_min <= 0:
            continue  # 공존 시간이 0이면(자료 매칭 안 됨) 제외
        row = {"Player": p, "CoOccur_Min": round(co_min, 1), "Team_Score": co_occur_team_score.get(p, 0),
               **stat}
        for k in ("PTS", "REB", "AST", "STL", "BLK", "TOV", "PF"):
            row[f"{k}_per36"] = round(stat[k] / co_min * 36, 2) if co_min else 0
        row["2P_PCT"] = round(stat["2P_M"] / stat["2P_A"] * 100, 1) if stat["2P_A"] else 0
        row["3P_PCT"] = round(stat["3P_M"] / stat["3P_A"] * 100, 1) if stat["3P_A"] else 0
        row["FT_PCT"] = round(stat["FT_M"] / stat["FT_A"] * 100, 1) if stat["FT_A"] else 0
        fit_rows.append(row)

    fit_rows.sort(key=lambda r: -r["CoOccur_Min"])

    # --- [복원 2] mcc_teammate_fit.csv 실제 저장 (이 블록이 빠져있었음) ---
    output_path_fit = Path("mcc_우리팀_fit.csv")
    with output_path_fit.open("w", newline="", encoding="utf-8-sig") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fit_headers)
        writer.writeheader()
        writer.writerows(fit_rows)
    print(f"[완료] 팀 동료 적합도 분석용: {output_path_fit}")
    # --- [복원 2] 끝 ---

    lineup_headers = ["Date", "Stint_Seq", "In_Time", "Out_Time",
                       "Duration_Min", "Teammates", "Opponents", "Team_Score"]

    output_path_lineup = Path("mcc_lineup_stints.csv")
    with output_path_lineup.open("w", newline="", encoding="utf-8-sig") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=lineup_headers)
        writer.writeheader()
        writer.writerows(lineup_stint_records)

    print(f"[완료] 온코트 구간별 라인업+팀스코어: {output_path_lineup}")


if __name__ == "__main__":
    main()