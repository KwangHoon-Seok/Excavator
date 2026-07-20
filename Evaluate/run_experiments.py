"""
EXPERIMENTS.md의 CASES를 파싱 → 케이스별 config 생성 → max_parallel 지켜 train.py 실행.
자기 자신은 백그라운드로 돌며 동시 실행 수를 관리(슬롯 나면 다음 케이스 시작). 완료까지 대기.

실행: conda run -n seok python Evaluate/run_experiments.py [--md ...] [--dry]
  --dry: config만 생성하고 실행은 안 함(점검용)
"""
import os, sys, time, subprocess, argparse
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable


def parse_md(md_path):
    settings, cases = {}, []
    section = None
    for ln in open(md_path):
        s = ln.strip()
        if s.startswith("## "):
            section = "settings" if "설정" in s or "SETTINGS" in s else ("cases" if "케이스" in s or "CASES" in s else None)
            continue
        if section == "settings" and s.startswith("- "):
            body = s[2:].split("#")[0].strip()          # 주석 제거
            if ":" in body:
                k, v = body.split(":", 1); settings[k.strip()] = v.strip()
        elif section == "cases" and s.startswith("|"):
            cells = [c.strip() for c in s.strip("|").split("|")]
            if not cells or set("".join(cells)) <= set("-| "):
                continue                                 # 구분선 스킵
            cases.append(cells)
    header, rows = (cases[0], cases[1:]) if cases else ([], [])
    return settings, header, rows


def to_ramp(cell):
    """'0.05→0.2@150000' → [0.05, 0.2, 150000]. '-'/'' → None."""
    cell = cell.replace("->", "→")
    if not cell or cell == "-":
        return None
    ab, n = cell.split("@")
    a, b = ab.split("→")
    return [float(a), float(b), int(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default=os.path.join(ROOT, "Model", "logs", "Version3", "EXPERIMENTS.md"))
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    settings, header, rows = parse_md(a.md)
    budget = int(settings.get("budget_steps", 2000000))
    max_par = int(settings.get("max_parallel", 3))
    base_cfg_path = os.path.join(ROOT, settings.get("base_config", "Model/config.yaml"))
    log_root_rel = settings.get("log_root", "Model/logs/Version3")
    train_script = os.path.join(ROOT, settings.get("train_script", "Model/train.py"))  # 버전별 코드 스냅샷 지원
    base = yaml.safe_load(open(base_cfg_path))
    ver = os.path.basename(log_root_rel)                 # 예: Version4
    for sub in ("checkpoints", "logs", os.path.join("..", "Evaluate", "plots")):
        os.makedirs(os.path.join(ROOT, "Model", sub, ver), exist_ok=True)

    # 헤더 기반: id/note/w_ttc_ramp/resume 외 컬럼명은 그대로 config 키로(숫자면 float, 아니면 문자열)
    jobs = []
    for cells in rows:
        row = {h: v for h, v in zip(header, cells)}
        cid = row.get("id", "").strip()
        if not cid:
            continue
        cfg = dict(base)
        cfg["run_name"] = f"{ver}/{cid}"
        cfg["total_steps"] = budget
        cfg["obstacle_training"] = True
        resume = ""
        for k, v in row.items():
            if k in ("id", "note") or v in ("", "-"):
                continue
            if k == "w_ttc_ramp":
                r = to_ramp(v)
                if r: cfg["w_ttc_ramp"] = r; cfg["w_ttc"] = r[0]
            elif k == "resume":
                resume = v
            else:
                try: cfg[k] = float(v)          # 숫자 컬럼(w_ttc, ttc_max, curr_up, ...)
                except ValueError: cfg[k] = v   # 문자열 컬럼(curriculum_mode 등)
        cfg_path = os.path.join(ROOT, log_root_rel, f"{cid}_config.yaml")
        os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
        yaml.safe_dump(cfg, open(cfg_path, "w"))
        jobs.append((cid, cfg_path, resume))
        ov = {k: row[k] for k in row if k not in ("id", "note") and row[k] not in ("", "-")}
        print(f"[cfg] {cid}: {ov} steps={budget}")

    if a.dry:
        print("[dry] config만 생성. 실행 생략."); return

    # max_parallel 지켜 실행
    running, pending = [], list(jobs)
    while pending or running:
        running = [(cid, p) for cid, p in running if p.poll() is None]
        while pending and len(running) < max_par:
            cid, cfg_path, resume = pending.pop(0)
            logf = open(os.path.join(ROOT, log_root_rel, f"{cid}.out"), "w")
            cmd = [PY, train_script, cfg_path]
            if resume not in ("", "-"): cmd += ["--resume", resume]
            p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=ROOT)
            running.append((cid, p))
            print(f"[run] {cid} 시작 (pid {p.pid}) — 로그 {log_root_rel}/{cid}.out", flush=True)
        time.sleep(10)
    print("[done] 모든 케이스 완료", flush=True)


if __name__ == "__main__":
    main()
