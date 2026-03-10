FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# -- Use USTC mirror (DEB822 format, replaces default sources) -------------
RUN rm -f /etc/apt/sources.list /etc/apt/sources.list.d/ubuntu.sources && \
    cat > /etc/apt/sources.list.d/ubuntu.sources <<'SOURCES'
Types: deb
URIs: http://mirrors.ustc.edu.cn/ubuntu
Suites: noble noble-updates noble-backports
Components: main restricted universe multiverse
Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg

Types: deb
URIs: http://mirrors.ustc.edu.cn/ubuntu
Suites: noble-security
Components: main restricted universe multiverse
Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg
SOURCES

# -- System packages -------------------------------------------------------
RUN set -eux; \
    for i in 1 2 3; do \
        apt-get update -y \
        && apt-get install -y --no-install-recommends --fix-missing \
            bash \
            build-essential \
            ca-certificates \
            coreutils \
            curl \
            diffutils \
            file \
            findutils \
            gawk \
            git \
            jq \
            less \
            make \
            openssh-client \
            patch \
            python3 \
            python3-pip \
            python3-venv \
            ripgrep \
            sed \
            tmux \
            tree \
            unzip \
            wget \
            zip \
        && break; \
        sleep 2; \
    done; \
    rm -rf /var/lib/apt/lists/*

# -- Node.js 22.x (LTS) via nodesource -------------------------------------
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# -- uv (Python package manager) -------------------------------------------
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

# -- Bun (JavaScript runtime) ----------------------------------------------
# NOTE: Downloads from GitHub. Needs proxy in restricted networks (--build-arg https_proxy=...).
RUN curl -fsSL https://bun.sh/install | BUN_INSTALL=/usr/local bash \
    && ln -sf /usr/local/bin/bun /usr/local/bin/bunx

# -- opencode ---------------------------------------------------------------
# NOTE: Downloads from GitHub. Needs proxy in restricted networks (--build-arg https_proxy=...).
RUN curl -fsSL https://opencode.ai/install | bash \
    && cp /root/.opencode/bin/opencode /usr/local/bin/opencode \
    && chmod +x /usr/local/bin/opencode

# -- yagents-apply-patch helper ---------------------------------------------
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

patch -p0 -u -N --silent --unsafe-paths < "$patch_rewritten"
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
