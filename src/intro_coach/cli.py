"""intro-coach serve | analyze <file-or-url>"""
import argparse
import json
import sys

from .analysis import SCENARIOS

SEV = {"critical": "\033[31m●\033[0m", "serious": "\033[33m●\033[0m", "warning": "\033[90m●\033[0m"}


def print_report(r: dict) -> None:
    m, s = r["metrics"], r["scores"]
    print(f"\n\033[1mIntro score {r['overall']}/100\033[0m   " + "  ".join(f"{k} {v}" for k, v in s.items()))
    print(f"{m['speaking_time']:.0f}s of {r['target_seconds']:.0f}s target · {m['wpm']:.0f} wpm "
          f"({m['articulation_wpm']:.0f} while talking) · {m['silence_pct']:.0f}% silence · "
          f"{m['hesitations']} mid-thought pauses · {m['fillers'] + m['hidden_fillers']} fillers · "
          f"pitch range {m['pitch_range_st']} st · {m['uptalk']} rising endings\n")
    print("\033[1mStructure\033[0m")
    for e in r["structure"]["elements"]:
        mark = "\033[32m✓\033[0m" if e["found"] else "\033[33m~\033[0m" if e["partial"] else "\033[31m✗\033[0m"
        print(f"  {mark} {e['label']}")
    print("\n\033[1mWork on\033[0m")
    for t in r["tips"][:7]:
        print(f"  {SEV[t['severity']]} {t['title']}\n    {t['detail']}")
        for e in t["evidence"][:2]:
            print(f"      {e['t']:6.1f}s  {e['quote']}")
        if t.get("drill"):
            print(f"    drill: {t['drill']}")
    if r["strengths"]:
        print("\n\033[1mKeep\033[0m")
        for x in r["strengths"]:
            print(f"  \033[32m+\033[0m {x}")
    print(f"\n\033[2mtranscript: {r['transcript']}\033[0m\n")


def main() -> None:
    p = argparse.ArgumentParser(prog="intro-coach", description="Practise your self-introduction.")
    sub = p.add_subparsers(dest="cmd", required=True)
    sv = sub.add_parser("serve", help="start the web app")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8765)
    an = sub.add_parser("analyze", help="coach a recording (file, URL or .m3u8)")
    an.add_argument("source")
    an.add_argument("--scenario", choices=list(SCENARIOS), default="interview")
    an.add_argument("--target", type=float, help="target length in seconds (default depends on scenario)")
    an.add_argument("--ai", action="store_true", help="add content feedback and a rewritten intro (OpenRouter Nemotron by default)")
    an.add_argument("--json", action="store_true", help="print the full report as JSON")
    an.add_argument("--no-save", action="store_true", help="don't add this take to your history")
    a = p.parse_args()

    if a.cmd == "serve":
        import uvicorn

        from . import server

        print(f"Intro Coach on http://{a.host}:{a.port}")
        uvicorn.run(server.app, host=a.host, port=a.port, log_level="warning")
        return

    from . import coach

    r = coach.run(a.source, a.scenario, a.target, save=not a.no_save)
    if a.ai:
        from . import ai

        r["ai"] = ai.feedback(r)
        if not a.no_save:
            coach.save_take(r)
    if a.json:
        json.dump(r, sys.stdout, indent=1)
        return
    print_report(r)
    if "ai" in r:
        f = r["ai"]
        print(f"\033[1mContent ({f['model']})\033[0m\n  {f['summary']}\n")
        for x in f["fixes"]:
            print(f"  “{x['said']}” → {x['issue']}\n    try: {x['try']}")
        print(f"\n\033[1mRewrite (~{f['rewrite_seconds']}s)\033[0m\n{f['rewrite']}\n\nNext take: {f['next_take_focus']}\n")


if __name__ == "__main__":
    main()
