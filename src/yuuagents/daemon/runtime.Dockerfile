FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN set -eux; \
    for i in 1 2 3; do \
        apt-get update -y \
        && apt-get install -y --no-install-recommends --fix-missing \
            bash \
            ca-certificates \
            coreutils \
            diffutils \
            findutils \
            gawk \
            patch \
            sed \
            tmux \
        && break; \
        sleep 2; \
    done; \
    rm -rf /var/lib/apt/lists/*

RUN cat > /usr/local/bin/yagents-apply-patch <<'EOF'
#!/usr/bin/env bash
set +e

target="$1"
if [ -z "$target" ]; then
  echo "missing target path"
  echo "__YAGENTS_EXIT_CODE__=2"
  exit 0
fi
case "$target" in
  /*) ;;
  *)
    echo "target path must be absolute: $target"
    echo "__YAGENTS_EXIT_CODE__=2"
    exit 0
    ;;
esac

before="$(mktemp)"
after="$(mktemp)"
patch_raw="$(mktemp)"
patch_rewritten="$(mktemp)"

cleanup() {
  rm -f "$before" "$after" "$patch_raw" "$patch_rewritten"
}
trap cleanup EXIT

mkdir -p "$(dirname "$target")"

if [ -f "$target" ]; then
  cat "$target" > "$before"
else
  : > "$before"
fi

cat > "$patch_raw"
if [ "$?" -ne 0 ]; then
  echo "failed to read patch"
  echo "__YAGENTS_EXIT_CODE__=3"
  exit 0
fi

awk -v tgt="$target" '
BEGIN { done_old=0; done_new=0 }
{
  if (!done_old && $0 ~ /^--- /) { print "--- " tgt; done_old=1; next }
  if (!done_new && $0 ~ /^\+\+\+ /) { print "+++ " tgt; done_new=1; next }
  print $0
}
' "$patch_raw" > "$patch_rewritten"

patch -p0 -u -N --silent < "$patch_rewritten"
code=$?
if [ "$code" -ne 0 ]; then
  echo "patch failed (exit=$code)"
  echo "__YAGENTS_EXIT_CODE__=$code"
  exit 0
fi

if [ -f "$target" ]; then
  cat "$target" > "$after"
else
  : > "$after"
fi

if cmp -s "$before" "$after"; then
  echo "$target"
  echo "no-op"
  echo "hunks: 0"
  echo "lines: +0 -0"
  echo "__YAGENTS_EXIT_CODE__=0"
  exit 0
fi

read hunks added removed <<EOF2
$(diff -u "$before" "$after" | awk '
BEGIN { h=0; a=0; r=0 }
/^@@ / { h++ }
/^\+\+\+ / { next }
/^--- / { next }
/^\+/ { a++; next }
/^-/ { r++; next }
END { printf "%d %d %d\n", h, a, r }
')
EOF2

echo "$target"
echo "hunks: $hunks"
echo "lines: +$added -$removed"
echo "__YAGENTS_EXIT_CODE__=0"
exit 0
EOF

RUN chmod +x /usr/local/bin/yagents-apply-patch
