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

