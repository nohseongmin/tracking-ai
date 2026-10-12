#!/usr/bin/env python3
"""Count ESTA files, ticks, stored frames, and player records."""

import argparse
import json
import lzma
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path, help="ESTA directory containing data/lan and data/online")
    parser.add_argument("--output", type=Path, default=Path("dataset_summary.json"))
    parser.add_argument("--limit", type=int, help="Read only the first N files for a quick check")
    args = parser.parse_args()

    data_dir = args.dataset / "data" if (args.dataset / "data").is_dir() else args.dataset
    files = sorted(data_dir.rglob("*.json.xz"))
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be at least 1")
        files = files[: args.limit]
    if not files:
        parser.error(f"No .json.xz files found under {data_dir}")

    totals = Counter()
    maps = Counter()
    tick_rates = Counter()
    parse_rates = Counter()
    errors = []

    for index, file_path in enumerate(files, 1):
        try:
            with lzma.open(file_path, "rt", encoding="utf-8") as source:
                match = json.load(source)

            totals["files_ok"] += 1
            totals["playback_ticks"] += int(match.get("playbackTicks") or 0)
            maps[str(match.get("mapName") or "unknown")] += 1
            tick_rates[str(match.get("tickRate") or "unknown")] += 1
            parse_rates[str((match.get("parserParameters") or {}).get("parseRate") or "unknown")] += 1

            for game_round in match.get("gameRounds") or []:
                totals["rounds"] += 1
                for frame in game_round.get("frames") or []:
                    totals["stored_frames"] += 1
                    for side in ("t", "ct"):
                        for player in ((frame.get(side) or {}).get("players") or []):
                            totals["player_records"] += 1
                            if player.get("isAlive"):
                                totals["alive_player_records"] += 1
        except (OSError, EOFError, lzma.LZMAError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
            errors.append({"file": str(file_path), "error": str(error)})

        if index % 50 == 0 or index == len(files):
            print(f"{index}/{len(files)} files processed", flush=True)

    playback_ticks = totals["playback_ticks"]
    dominant_tick_rate = int(tick_rates.most_common(1)[0][0]) if tick_rates and tick_rates.most_common(1)[0][0].isdigit() else None
    result = {
        "dataset_path": str(args.dataset.resolve()),
        "files_found": len(files),
        **dict(totals),
        "files_failed": len(errors),
        "approx_playback_hours": round(playback_ticks / dominant_tick_rate / 3600, 2) if dominant_tick_rate else None,
        "tick_rates": dict(tick_rates),
        "parse_rates": dict(parse_rates),
        "maps": dict(maps),
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
