#!/usr/bin/env bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Resolve task-image tags to digest-pinned references the cluster can pull.
#
#   resolve_images.sh IMAGES_TXT REGISTRY_REPO [OUT_JSON] [PARALLEL]
#
# IMAGES_TXT     one Docker Hub reference per line (from `python -m uni_agent_substrate.datasets --images-out`)
# REGISTRY_REPO  either an Artifact Registry *remote* repository that proxies Docker Hub
#                (e.g. REGION-docker.pkg.dev/PROJECT/<dockerhub-remote-repo>): no copy, the digest is
#                resolved through the proxy and the image is fetched lazily on first pull;
#                or, with MODE=mirror, a standard repository to copy images into by digest
#                (same approach as benchmarking/swebench/mirror-images.sh).
#                MODE=arapi lists an Artifact Registry *standard* repository through the AR API
#                (`gcloud artifacts docker images list --include-tags`; no registry traffic, works
#                where *-docker.pkg.dev is unreachable) and maps each tag to its digest there.
#                MODE=hubapi resolves digests through Docker Hub's tags API instead of registry
#                manifest requests (the proxy path is still what gets written): thousands of tags
#                resolve in minutes without hitting Docker Hub's pull rate limit. Pulling the images
#                later still counts against that limit, so give the remote repository upstream
#                Docker Hub credentials before booting thousands of distinct images.
# OUT_JSON       mapping {tag ref: digest ref} consumed by `datasets --image-digests` and
#                by the ate_env provider (default images.json)
#
# MIRROR_NAME_SEP replaces the "/" between Docker Hub namespace and repository when building the
# registry path (default "/"): a mirror that flattens names, e.g. r2e-gym/namanjain12-aiohttp_final,
# is addressed with MIRROR_NAME_SEP=- (proxy, mirror and hubapi modes alike).
#
# Needs crane (go install github.com/google/go-containerregistry/cmd/crane@latest) and
# registry credentials (gcloud auth configure-docker REGION-docker.pkg.dev); hubapi mode needs neither.
set -euo pipefail
IMAGES="${1:?images.txt}"; REPO="${2:?registry repo}"; OUT="${3:-images.json}"; PAR="${4:-8}"
MODE="${MODE:-proxy}"
SEP="${MIRROR_NAME_SEP:-/}"
case "$MODE" in hubapi|arapi) ;; *) command -v crane >/dev/null || { echo "crane not found on PATH" >&2; exit 1; } ;; esac
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT

resolve_one() {  # prints "<tag ref> <digest ref>"
  local ref="$1" name tag src dst digest
  name="${ref%%:*}"; tag="${ref##*:}"; [ "$name" = "$ref" ] && tag=latest
  if [ "$MODE" = "mirror" ]; then
    src="$ref"; dst="$REPO/$(echo "${name#docker.io/}" | sed "s|/|$SEP|g")"
    digest=$(crane digest "$src") || return 1
    crane copy "$src@$digest" "$dst:$tag" >/dev/null 2>&1 || crane copy "$src" "$dst:$tag" >/dev/null || return 1
    echo "$ref $dst@$digest"
  else
    dst="$REPO/$(echo "${name#docker.io/}" | sed "s|/|$SEP|g")"
    digest=$(crane digest "$dst:$tag") || return 1   # resolving through the proxy also caches the image
    echo "$ref $dst@$digest"
  fi
}
export -f resolve_one; export REPO MODE SEP
: > "$tmp/failed.txt"; export FAILED_LOG="$tmp/failed.txt"
if [ "$MODE" = "hubapi" ]; then
  python3 - "$IMAGES" "$REPO" "$SEP" > "$tmp/map.txt" <<'PY'
import collections, json, os, sys, time, urllib.parse, urllib.request
refs = [l.strip() for l in open(sys.argv[1]) if l.strip() and not l.startswith("#")]
repo, sep = sys.argv[2], sys.argv[3]
def dst(name):
    return f"{repo}/{name.removeprefix('library/').replace('/', sep)}"
by_repo = collections.defaultdict(dict)
for ref in refs:
    name, _, tag = ref.rpartition(":")
    name = name.replace("docker.io/", "")
    if "/" not in name:
        name = "library/" + name
    by_repo[name][tag or "latest"] = ref
def get(url, tries=8):
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "resolve_images.sh"}), timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (403, 429, 500, 502, 503) and i < tries - 1:   # 403 is how hub.docker.com throttles too
                time.sleep(2 ** i); continue
            raise
missing = 0
for name, tags in by_repo.items():
    wanted = dict(tags)
    url = f"https://hub.docker.com/v2/repositories/{name}/tags?page_size=100"
    while url and wanted:
        try:
            page = get(url)
        except Exception as e:
            print(f"FAILED listing {name}: {e}", file=sys.stderr); break   # its tags fall through to the direct lookups
        for t in page.get("results", []):
            if t["name"] in wanted and t.get("digest"):
                print(wanted.pop(t["name"]), f"{dst(name)}@{t['digest']}")
        url = page.get("next")
    for tag, ref in wanted.items():   # not on a page (deleted tag?): one direct lookup
        try:
            d = get(f"https://hub.docker.com/v2/repositories/{name}/tags/{urllib.parse.quote(tag)}")
            print(ref, f"{dst(name)}@{d['digest']}")
        except Exception as e:
            missing += 1; print(f"FAILED {ref}: {e}", file=sys.stderr); open(os.environ["FAILED_LOG"], "a").write(ref + "\n")
print(f"hubapi: {len(refs) - missing} resolved, {missing} failed", file=sys.stderr)
PY
elif [ "$MODE" = "arapi" ]; then
  gcloud artifacts docker images list "$REPO" --include-tags --format=json > "$tmp/ar.json"
  python3 - "$IMAGES" "$REPO" "$SEP" "$tmp/ar.json" > "$tmp/map.txt" <<'PY'
import json, os, sys
refs = [l.strip() for l in open(sys.argv[1]) if l.strip() and not l.startswith("#")]
repo, sep = sys.argv[2], sys.argv[3]
index = {}
for img in json.load(open(sys.argv[4])):
    tags = img.get("tags") or []
    if isinstance(tags, str):
        tags = [t for t in tags.split(",") if t]
    for t in tags:
        index[(img["package"], t)] = img["version"]
missing = 0
for ref in refs:
    name, _, tag = ref.rpartition(":")
    name = name.replace("docker.io/", "")
    if "/" not in name:
        name = "library/" + name
    package = f"{repo}/{name.removeprefix('library/').replace('/', sep)}"
    digest = index.get((package, tag or "latest"))
    if digest:
        print(ref, f"{package}@{digest}")
    else:
        missing += 1; print(f"FAILED {ref}: not in {repo}", file=sys.stderr); open(os.environ["FAILED_LOG"], "a").write(ref + "\n")
print(f"arapi: {len(refs) - missing} resolved, {missing} failed", file=sys.stderr)
PY
else
  grep -vE '^\s*(#|$)' "$IMAGES" | xargs -P "$PAR" -I{} bash -c 'resolve_one "$1" || { echo "FAILED $1" >&2; echo "$1" >> "$FAILED_LOG"; }' _ {} > "$tmp/map.txt"
fi
python3 - "$tmp/map.txt" "$OUT" <<'PY'
import json, sys
pairs = dict(line.split(None, 1) for line in open(sys.argv[1]) if line.strip())
existing = {}
try:
    existing = json.load(open(sys.argv[2]))
except FileNotFoundError:
    pass
existing.update({k: v.strip() for k, v in pairs.items()})
json.dump(existing, open(sys.argv[2], "w"), indent=1, sort_keys=True)
print(f"resolved {len(pairs)} images -> {sys.argv[2]} ({len(existing)} total)")
PY
if [ -s "$tmp/failed.txt" ]; then
  echo "$(wc -l < "$tmp/failed.txt") images did not resolve (listed above); the map is partial" >&2
  exit 1
fi
