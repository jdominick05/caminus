#!/usr/bin/env bash
# The nightly build, run on the maintainer's machine by a timer (packaging/nightly/; docs/dev/releasing.md, "Nightly
# builds"): when the branch head has a green CI run and no nightly yet, build it the way a release is built and publish it
# as a GitHub pre-release, then start .github/workflows/nightly.yml on the branch, which verifies the pre-release's files
# and uploads the wheel to PyPI as caminus X.Y.Z.devYYYYMMDD. It runs here and not in CI for the release's reason: CI does not have the toolchain
# that builds the shipped code objects.
#
#   tools/release/nightly.sh                  build and publish, or say why there is nothing to do (exit 0 either way)
#   tools/release/nightly.sh --dry-run        everything up to the verified files; no tag is pushed, no release is created
#   tools/release/nightly.sh --dry-run --no-ci-check    the same without asking GitHub for the commit's CI run
#
# What a run does:
#   1. nothing, when $NIGHTLY_HOME/HOLD exists or a guarded GPU job other than the server is running (two kernel builds
#      would disturb its timings); the commit is picked up by the next run
#   2. reads the branch head and the nightly tags; a head whose nightly tag and pre-release both exist is done. The test
#      is stateless (tools/release/nightly_plan.py built): a skipped or failed night repairs itself at the next run
#   3. needs a completed, successful ci.yml push run on that commit
#   4. clone A at the head: release_build.sh --write-hashes (the kernels, built by the real toolchain), then ONE commit
#      on top of the head holding the version X.Y.Z.devYYYYMMDD and that build's code-object hashes, tagged
#      vX.Y.Z.devYYYYMMDD; the commit is on no branch
#   5. clone B at the tag: release_build.sh (the kernels again: the second build must reproduce the committed hashes),
#      the wheel, the sdist, SHA256SUMS, verify_release.py
#   6. nightly_plan.py check-tag and the privacy scan of the tagged tree; only then the tag is pushed and the pre-release
#      created with the three files (gh release create --prerelease --latest=false)
#   7. starts the nightly workflow on the branch for that tag (gh workflow run nightly.yml --ref BRANCH -f tag=TAG).
#      GitHub runs the workflow file of the ref it is started on, so the checks are always the reviewed branch's, never a
#      tag's copy. A start that GitHub refuses (it allows one only once nightly.yml is on the default branch) is
#      reported and is not a failure: the pre-release stands
# A head that has its tag but no pre-release (a run that died between the two) is finished from the tag: clone B only.
# A pre-release without its PyPI upload: start the workflow by hand
# (gh workflow run nightly.yml --ref testing -f tag=TAG).
#
# Environment (the timer's env file: packaging/nightly/nightly.env.example):
#   NIGHTLY_GIT_NAME, NIGHTLY_GIT_EMAIL   the identity of the nightly commit and tag (required; it becomes public)
#   PY, ROCM_PATH, IGNITE, TORCHLIB, PATCHELF   as tools/release/release_build.sh reads them; PATH must find git, gh, go
#   GH_TOKEN                              for gh and git when no login session holds gh's token (a timer after a reboot)
#   NIGHTLY_HOME       state: mirror.git, builds/, HOLD, last (default $XDG_STATE_HOME/caminus-nightly)
#   NIGHTLY_REPO       owner/name (default jdominick05/caminus); NIGHTLY_BRANCH (default testing); NIGHTLY_URL (default
#                      https://github.com/$NIGHTLY_REPO.git)
#   NIGHTLY_KEEP_BUILDS   build directories kept under $NIGHTLY_HOME/builds (default 2; older ones are deleted, the only
#                      thing this script deletes: each holds two clones with their build output)
#   NIGHTLY_DATE       YYYYMMDD (default: today, local time). One nightly a day: a PyPI version cannot be reused
# CPU only: nothing here opens a GPU.
set -euo pipefail
ROOT=$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)
dry=0; noci=0
for a in "$@"; do
    case "$a" in
        --dry-run) dry=1 ;;
        --no-ci-check) noci=1 ;;
        *) echo "usage: tools/release/nightly.sh [--dry-run [--no-ci-check]]" >&2; exit 2 ;;
    esac
done
say() { echo "nightly: $*"; }
die() { echo "nightly: $*" >&2; exit 1; }
[ "$noci" = 0 ] || [ "$dry" = 1 ] || die "--no-ci-check goes with --dry-run only"

REPO=${NIGHTLY_REPO:-jdominick05/caminus}
BRANCH=${NIGHTLY_BRANCH:-testing}
URL=${NIGHTLY_URL:-https://github.com/$REPO.git}
NH=${NIGHTLY_HOME:-${XDG_STATE_HOME:-$HOME/.local/state}/caminus-nightly}
KEEP=${NIGHTLY_KEEP_BUILDS:-2}
DATE=${NIGHTLY_DATE:-$(date +%Y%m%d)}
export PY=${PY:-python3}
PLAN=(env -u PYTHONPATH "$PY" -I "$ROOT/tools/release/nightly_plan.py")
case "$NH" in /*) ;; *) die "NIGHTLY_HOME must be an absolute path, not '$NH'" ;; esac
[[ "$KEEP" =~ ^[1-9][0-9]*$ ]] || die "NIGHTLY_KEEP_BUILDS must be a count of at least 1, not '$KEEP'"
for t in git go flock tar "$PY"; do command -v "$t" > /dev/null || die "$t is missing (PATH=$PATH)"; done
[ -n "${NIGHTLY_GIT_NAME:-}" ] && [ -n "${NIGHTLY_GIT_EMAIL:-}" ] || die "NIGHTLY_GIT_NAME and NIGHTLY_GIT_EMAIL are not set (the nightly commit's identity)"
ident=(-c "user.name=$NIGHTLY_GIT_NAME" -c "user.email=$NIGHTLY_GIT_EMAIL" -c commit.gpgsign=false -c tag.gpgsign=false)
need_gh=1; [ "$dry" = 1 ] && [ "$noci" = 1 ] && need_gh=0
if [ "$need_gh" = 1 ]; then
    command -v gh > /dev/null || die "gh is missing (PATH=$PATH)"
    gh auth status > /dev/null 2>&1 || die "gh has no usable login here (a timer without a login session cannot read a keyring: set GH_TOKEN in the env file)"
fi

mkdir -p "$NH/builds"
exec 9> "$NH/lock"
flock -n 9 || die "another nightly run holds $NH/lock"
finish() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" > "$NH/last"; say "$*"; }

# ---- 1. reasons to do nothing tonight
if [ -e "$NH/HOLD" ]; then finish "held ($NH/HOLD exists): nothing done"; exit 0; fi
busy=$(pgrep -af 'guard\.sh' | grep -Ev 'caminus_serve\.py|caminus serve' || true)
if [ -n "$busy" ]; then finish "a guarded GPU job is running: not building beside it (the next run picks the commit up)"; exit 0; fi

# ---- 2. the branch head and its nightly, from a mirror that holds only the branch and the version tags
M=$NH/mirror.git
[ -d "$M" ] || git init -q --bare "$M"
git -C "$M" fetch -q --prune --no-tags "$URL" "+refs/heads/$BRANCH:refs/heads/$BRANCH" '+refs/tags/v*:refs/tags/v*' \
    || die "cannot fetch $BRANCH from $URL"
head=$(git -C "$M" rev-parse --verify --quiet "refs/heads/$BRANCH^{commit}") || die "no branch $BRANCH at $URL"
short=${head:0:12}
resume=0
if tag=$("${PLAN[@]}" --repo "$M" built --base "$head"); then
    if [ "$need_gh" = 1 ] && gh release view "$tag" -R "$REPO" > /dev/null 2>&1; then
        finish "$BRANCH at $short is published as $tag: nothing to do"; exit 0
    fi
    [ "$need_gh" = 1 ] || { finish "$BRANCH at $short has the nightly tag $tag (its pre-release was not looked up: --no-ci-check): nothing to do"; exit 0; }
    say "$BRANCH at $short has the tag $tag but no pre-release: finishing it from the tag"
    resume=1; VER=${tag#v}
else
    rc=$?; [ "$rc" = 3 ] || die "nightly_plan.py built failed (exit $rc)"
    # ---- 3. CI on that commit
    if [ "$noci" = 0 ]; then
        green=$(gh run list -R "$REPO" --workflow ci.yml --commit "$head" --event push --json status,conclusion \
                   --jq 'map(select(.status == "completed" and .conclusion == "success")) | length') || die "cannot read the CI runs of $short"
        if [ "$green" -lt 1 ]; then finish "$BRANCH at $short has no completed, successful ci.yml push run: not built (the next run asks again)"; exit 0; fi
    fi
    VER=$("${PLAN[@]}" --repo "$M" version --commit "$head" --date "$DATE") || die "no version for $short"
    if git -C "$M" rev-parse --verify --quiet "refs/tags/v$VER" > /dev/null; then
        finish "v$VER exists for another commit and a version is published once: $BRANCH at $short waits for the next date"; exit 0
    fi
fi
say "caminus $VER from $BRANCH at $head ($([ "$dry" = 1 ] && echo "dry run" || echo "to be published"))"

# ---- the build directory; older ones go (each holds two clones with their build output)
mapfile -t old < <(ls -1 "$NH/builds" | grep -E '^[0-9]{8}-[0-9a-f]{12}-[0-9]{6}$' | sort | head -n -"$((KEEP - 1))")
for d in "${old[@]}"; do
    [[ "$d" =~ ^[0-9]{8}-[0-9a-f]{12}-[0-9]{6}$ ]] || die "refusing to delete '$d'"
    say "deleting the old build directory $NH/builds/$d"
    rm -rf -- "$NH/builds/$d"
done
W=$NH/builds/$DATE-$short-$(date +%H%M%S)
mkdir "$W"
build() {   # build DIR LOG ARGS...: release_build.sh in a clone, its output in LOG
    local dir=$1 log=$2; shift 2
    (cd "$dir" && tools/release/release_build.sh "$@") > "$log" 2>&1 || { tail -40 "$log" >&2; die "release_build.sh $* failed in $dir ($log)"; }
    tail -3 "$log"
}

# ---- 4. clone A: the code objects' hashes, then the nightly commit and its tag
if [ "$resume" = 0 ]; then
    git clone -q --no-checkout "$M" "$W/a"   # (the mirror has no HEAD: check the commit out by name)
    git -C "$W/a" fetch -q origin '+refs/tags/v*:refs/tags/v*'
    git -C "$W/a" checkout -q --detach "$head"
    say "clone A: the kernels and their hashes ($W/a.log)"
    build "$W/a" "$W/a.log" --write-hashes
    sed -i "s/^__version__ *=.*/__version__ = '$VER'/" "$W/a/python/caminus/__init__.py"
    grep -qx "__version__ = '$VER'" "$W/a/python/caminus/__init__.py" || die "could not set __version__ in clone A"
    git -C "$W/a" "${ident[@]}" commit -q -m "release : nightly $VER ($BRANCH at $short)" \
        -m "The version and the code-object hashes of this night's build on top of $head. This commit is on no branch: the tag v$VER is its only name (docs/dev/releasing.md, Nightly builds)." \
        -- python/caminus/__init__.py packaging/gfx1100-kernels.sha256
    git -C "$W/a" "${ident[@]}" tag -a "v$VER" -m "Caminus nightly $VER: $BRANCH at $head"
    src=$W/a
else
    src=$M
fi

# ---- 5. clone B at the tag: the release files, verified
git clone -q --no-checkout "$src" "$W/b"
git -C "$W/b" fetch -q origin '+refs/tags/v*:refs/tags/v*'
git -C "$W/b" checkout -q --detach "refs/tags/v$VER"
say "clone B at v$VER: the release files ($W/b.log)"
build "$W/b" "$W/b.log"
dist=$W/b/dist
files=("$dist/caminus-$VER-py3-none-any.whl" "$dist/caminus-$VER.tar.gz" "$dist/SHA256SUMS")
for f in "${files[@]}"; do [ -f "$f" ] || die "$f is missing"; done

# ---- 6. the last checks before anything is public
"${PLAN[@]}" --repo "$W/b" check-tag "v$VER" --base "$head" || die "v$VER is not a nightly of $short (nothing was pushed)"
mkdir "$W/scan"
git -C "$W/b" archive "refs/tags/v$VER" | tar -x -C "$W/scan"
env -u PYTHONPATH "$PY" -I "$ROOT/tools/release/privacy_scan.py" "$W/scan" | tail -3 || die "the privacy scan refused the tagged tree (nothing was pushed)"

prev=$(git -C "$M" tag -l 'v*' --sort=-creatordate | grep -vx "v$VER" | head -1 || true)
since=$prev; [[ "$prev" == *.dev* ]] && since="$prev^"
{
    echo "Nightly build of \`$BRANCH\` at [\`$short\`](https://github.com/$REPO/commit/$head)."
    echo
    echo "**Not a release.** CI was green on that commit and the files were built and verified like a release's"
    echo "(the code objects by the maintainer's toolchain: docs/dev/releasing.md), but a release's GPU gates and manual"
    echo "tests were not run on this build. Expect rough edges; issues are welcome."
    echo
    echo '```'
    echo "pip install \"caminus==$VER\"        # this nightly (PyPI, once the nightly workflow has verified these files)"
    echo "pip install \"caminus>=${VER%.dev*}.dev0\"   # the newest nightly of this version"
    echo "pip install caminus                      # stays on the last release"
    echo '```'
    echo
    echo "Name a dev version as above rather than passing \`--pre\`: \`--pre\` also takes pre-releases of every dependency."
    if [ -n "$prev" ]; then
        echo; echo "Changes since \`$prev\`:"; echo
        git -C "$M" log --no-merges --format='- %s' "$since..$head" | head -60
    fi
} > "$W/notes.md"

if [ "$dry" = 1 ]; then
    say "dry run: v$VER would be pushed and published with:"; cat "$dist/SHA256SUMS"; echo "--- notes ($W/notes.md)"; cat "$W/notes.md"
    finish "dry run of $VER ($BRANCH at $short) passed: $dist"; exit 0
fi
[ "$resume" = 1 ] || git -C "$W/b" push -q "$URL" "refs/tags/v$VER:refs/tags/v$VER" || die "cannot push the tag v$VER"
gh release create "v$VER" "${files[@]}" -R "$REPO" --verify-tag --prerelease --latest=false \
    --title "Caminus nightly $VER" --notes-file "$W/notes.md" || die "the tag v$VER is pushed but its pre-release was not created: the next run finishes it"
# ---- 7. the PyPI upload: the workflow of the branch, never of the tag
pypi="the PyPI workflow is started"
gh workflow run nightly.yml -R "$REPO" --ref "$BRANCH" -f "tag=v$VER" > "$W/dispatch.log" 2>&1 \
    || pypi="the PyPI workflow was NOT started ($(tr '\n' ' ' < "$W/dispatch.log" | cut -c1-160)); by hand: gh workflow run nightly.yml -R $REPO --ref $BRANCH -f tag=v$VER"
finish "published $VER ($BRANCH at $short): https://github.com/$REPO/releases/tag/v$VER; $pypi"
