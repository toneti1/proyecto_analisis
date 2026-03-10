import argparse
import json
import os
import time
import traceback
from pathlib import Path

from automatizador import process_video_file


def read_job(job_path: Path) -> dict:
    try:
        with open(job_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def write_job(job_path: Path, payload: dict) -> None:
    job_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = job_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)
    tmp_path.replace(job_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--clip-count", type=int, default=12)
    parser.add_argument("--job", required=True)
    args = parser.parse_args()

    job_path = Path(args.job)
    job = read_job(job_path)
    job.update(
        {
            "state": "running",
            "input_path": args.input,
            "clip_count": int(args.clip_count),
            "started_at": job.get("started_at", int(time.time())),
        }
    )
    write_job(job_path, job)

    try:
        result = process_video_file(args.input, clip_count=int(args.clip_count))
        job.update(
            {
                "state": "done",
                "result": result,
                "finished_at": int(time.time()),
            }
        )
    except Exception as exc:
        job.update(
            {
                "state": "error",
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "finished_at": int(time.time()),
            }
        )
    finally:
        write_job(job_path, job)
        try:
            if os.path.exists(args.input):
                os.remove(args.input)
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
