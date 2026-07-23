"""Stream a large file to stdout as a sequence of ordered HTTP Range requests.

The OOPS host (oops.cs.columbia.edu) throttles SUSTAINED single connections to ~0 after a
big transfer from an IP, but serves short ranged requests at full speed. This fetches the
file in ordered, individually-retried chunks and writes the reassembled byte stream to
stdout — contiguous and gzip-safe — so it can be piped into `tar xz`. Only the extracted
files touch disk, never the 45GB archive.

    python3 scripts/chunked_fetch.py URL [chunk_mb] | tar xzOf - inner.tar.gz | tar xzvf ...
"""
import sys
import time
import urllib.request

URL = sys.argv[1]
CHUNK = int(sys.argv[2]) * 1024 * 1024 if len(sys.argv) > 2 else 16 * 1024 * 1024
# The OOPS host throttles to ~0 after ~8GB of SUSTAINED pull, then recovers for fresh short
# connections after a cooldown window. Each urlopen is a new connection, so the survival
# strategy is: many retries, and after a run of failures wait out the throttle window with a
# long cooldown rather than giving up (the un-resumable stream must not die).
MAX_RETRY = 40
COOLDOWN = 300   # seconds to wait out a throttle window after repeated failures


def head_size(url):
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"])


def fetch_range(url, start, end):
    """Return exactly bytes [start, end] (inclusive). Survives throttle windows: after a few
    quick retries fail, sleep a long cooldown to let the per-IP throttle reset, then keep
    trying — up to MAX_RETRY total before giving up."""
    for attempt in range(MAX_RETRY):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=90) as r:
                data = r.read()
            if len(data) == end - start + 1:
                return data
            sys.stderr.write(f"chunk {start}: got {len(data)} want {end-start+1}, retry\n")
        except Exception as e:
            sys.stderr.write(f"chunk {start} attempt {attempt}: {type(e).__name__} {e}\n")
        # quick backoff for the first few; then long cooldowns to outlast a throttle window
        wait = min(20, 2 ** attempt) if attempt < 5 else COOLDOWN
        if wait >= COOLDOWN:
            sys.stderr.write(f"  cooldown {wait}s (throttle) at chunk {start}, attempt {attempt}\n")
        sys.stderr.flush()
        time.sleep(wait)
    raise RuntimeError(f"chunk {start}-{end} failed after {MAX_RETRY} retries")


def main():
    total = head_size(URL)
    sys.stderr.write(f"total {total} bytes, chunk {CHUNK}, ~{total//CHUNK+1} chunks\n")
    sys.stderr.flush()
    out = sys.stdout.buffer
    off = 0
    done = 0
    t0 = time.time()
    while off < total:
        end = min(off + CHUNK - 1, total - 1)
        out.write(fetch_range(URL, off, end))
        out.flush()
        off = end + 1
        done += 1
        if done % 20 == 0:
            mb = off / 1e6
            rate = mb / (time.time() - t0 + 1e-9)
            sys.stderr.write(f"  {mb:.0f}/{total/1e6:.0f} MB ({100*off//total}%) {rate:.1f} MB/s\n")
            sys.stderr.flush()
    sys.stderr.write("chunked_fetch: complete\n")


if __name__ == "__main__":
    main()
