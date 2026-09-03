#!/usr/bin/env bash
# Install make, Claude Code, and Docker, then verify each works.
#
# Supported platforms:
#   - Linux (Debian/Ubuntu, apt) — tested on Ubuntu EC2
#   - macOS (Homebrew)
#   - Windows (Git Bash / MSYS / Cygwin, winget)
#
# Usage:
#   bash install.sh

set -eu

# Pinned Docker versions for reproducibility. Bump these to upgrade.
DOCKER_ENGINE_VERSION="29.4.3"     # apt-based Linux (docker-ce / docker-ce-cli)
DOCKER_DESKTOP_VERSION="4.37.2"    # Windows winget (Docker.DockerDesktop)

DOCKER_GROUP_ADDED=0
SUDO=""
if [ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null 2>&1; then
    SUDO="sudo"
fi

OS="$(uname -s)"

log()  { printf '\033[1;34m[install]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31m[fail]\033[0m %s\n' "$*" >&2; exit 1; }

need_node() {
    if ! command -v node >/dev/null 2>&1; then
        log "installing Node.js (required for Claude Code)"
        case "$OS" in
            Linux)
                curl -fsSL https://deb.nodesource.com/setup_20.x | $SUDO -E bash -
                $SUDO apt-get install -y nodejs
                ;;
            Darwin)
                brew install node
                ;;
            MINGW*|MSYS*|CYGWIN*)
                winget install --id OpenJS.NodeJS.LTS --silent \
                    --accept-source-agreements --accept-package-agreements \
                    || fail "winget failed to install Node.js"
                ;;
        esac
    fi
}

# Daemon reachability check that bypasses the current shell's group
# membership. After `usermod -aG docker $USER` the new group isn't active
# in this shell, so a plain `docker info` would fail for permission
# reasons — we'd misread that as "daemon down". Prefer sudo, fall back
# to `sg docker`, then to a bare check.
docker_daemon_up() {
    if [ -n "$SUDO" ]; then
        $SUDO docker info >/dev/null 2>&1
    elif [ "$DOCKER_GROUP_ADDED" -eq 1 ] && command -v sg >/dev/null 2>&1; then
        sg docker -c "docker info" >/dev/null 2>&1
    else
        docker info >/dev/null 2>&1
    fi
}

# Poll the daemon until it answers or we give up. Default 60s covers
# Docker Desktop cold-start on macOS and systemd unit activation on Linux.
wait_for_docker() {
    timeout="${1:-60}"
    log "waiting up to ${timeout}s for docker daemon"
    i=0
    while [ "$i" -lt "$timeout" ]; do
        if docker_daemon_up; then
            return 0
        fi
        sleep 1
        i=$((i + 1))
    done
    return 1
}

# Best-effort start of the docker daemon. Linux: systemd unit. macOS /
# Windows: launch Docker Desktop.
start_docker_daemon() {
    case "$OS" in
        Linux)
            if command -v systemctl >/dev/null 2>&1; then
                log "starting docker via systemctl"
                $SUDO systemctl start docker || warn "systemctl start docker failed"
            elif command -v service >/dev/null 2>&1; then
                log "starting docker via service"
                $SUDO service docker start || warn "service docker start failed"
            else
                warn "no systemctl/service found — start the docker daemon manually"
            fi
            ;;
        Darwin)
            if [ -d "/Applications/Docker.app" ]; then
                log "launching Docker Desktop"
                open -a Docker
            else
                warn "Docker Desktop not found at /Applications/Docker.app"
            fi
            ;;
        MINGW*|MSYS*|CYGWIN*)
            dd_exe="/c/Program Files/Docker/Docker/Docker Desktop.exe"
            if [ -x "$dd_exe" ]; then
                log "launching Docker Desktop"
                "$dd_exe" >/dev/null 2>&1 &
                disown 2>/dev/null || true
            else
                warn "Docker Desktop not found at C:\\Program Files\\Docker\\Docker\\Docker Desktop.exe — start it manually"
            fi
            ;;
    esac
}

install_linux() {
    log "detected Linux — using apt"
    command -v apt-get >/dev/null 2>&1 || fail "apt-get not found; this script supports Debian/Ubuntu"

    $SUDO apt-get update -y
    $SUDO apt-get install -y curl ca-certificates gnupg lsb-release make

    if ! command -v docker >/dev/null 2>&1; then
        log "installing Docker Engine ${DOCKER_ENGINE_VERSION}"
        $SUDO install -m 0755 -d /etc/apt/keyrings
        curl -fsSL https://download.docker.com/linux/ubuntu/gpg | $SUDO gpg --dearmor -o /etc/apt/keyrings/docker.gpg
        $SUDO chmod a+r /etc/apt/keyrings/docker.gpg
        echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
            | $SUDO tee /etc/apt/sources.list.d/docker.list >/dev/null
        $SUDO apt-get update -y

        # apt versions for docker-ce are codename-suffixed
        # (e.g. "5:27.5.1-1~ubuntu.22.04~jammy"). Resolve the matching
        # version string for this host's codename instead of hard-coding it.
        pkg_ver=$(apt-cache madison docker-ce | awk '{print $3}' | grep -m1 -- "${DOCKER_ENGINE_VERSION}" || true)
        [ -n "$pkg_ver" ] || fail "Docker Engine ${DOCKER_ENGINE_VERSION} not available for $(lsb_release -cs)"
        $SUDO apt-get install -y \
            docker-ce="${pkg_ver}" \
            docker-ce-cli="${pkg_ver}" \
            containerd.io \
            docker-buildx-plugin \
            docker-compose-plugin
    fi

    if command -v systemctl >/dev/null 2>&1; then
        log "enabling docker daemon at boot"
        $SUDO systemctl enable docker || warn "could not enable docker via systemctl"
    fi
    start_docker_daemon

    target_user="${SUDO_USER:-$(id -un)}"
    if [ "$target_user" != "root" ] && ! id -nG "$target_user" | tr ' ' '\n' | grep -qx docker; then
        log "adding $target_user to the docker group"
        $SUDO usermod -aG docker "$target_user"
        DOCKER_GROUP_ADDED=1
    fi

    need_node
    if ! command -v claude >/dev/null 2>&1; then
        log "installing Claude Code via npm"
        $SUDO npm install -g @anthropic-ai/claude-code
    fi
}

install_macos() {
    log "detected macOS — using Homebrew"
    if ! command -v brew >/dev/null 2>&1; then
        log "installing Homebrew"
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi

    brew install make
    # Homebrew Cask doesn't support pinning to arbitrary historical
    # versions without maintaining a versioned tap, so macOS gets the
    # current stable Docker Desktop cask. Bump the Linux/Windows pins
    # above when this drifts too far from the team's target.
    brew list --cask docker >/dev/null 2>&1 || brew install --cask docker

    start_docker_daemon

    need_node
    command -v claude >/dev/null 2>&1 || npm install -g @anthropic-ai/claude-code
}

# Windows path runs from Git Bash / MSYS / Cygwin and shells out to
# winget (built into Windows 10+). winget may trigger UAC prompts when
# installing system-scope packages like Docker Desktop — that's expected.
install_windows() {
    log "detected Windows ($OS) — using winget"
    command -v winget >/dev/null 2>&1 \
        || fail "winget not found; install 'App Installer' from the Microsoft Store"

    if ! command -v make >/dev/null 2>&1; then
        log "installing GNU make"
        winget install --id ezwinports.make --silent \
            --accept-source-agreements --accept-package-agreements \
            || warn "winget failed to install make — install manually if needed"
    fi

    if ! command -v docker >/dev/null 2>&1; then
        log "installing Docker Desktop ${DOCKER_DESKTOP_VERSION}"
        winget install --id Docker.DockerDesktop --version "${DOCKER_DESKTOP_VERSION}" --silent \
            --accept-source-agreements --accept-package-agreements \
            || fail "winget failed to install Docker Desktop ${DOCKER_DESKTOP_VERSION}"
    fi

    start_docker_daemon

    need_node
    if ! command -v claude >/dev/null 2>&1; then
        log "installing Claude Code via npm"
        npm install -g @anthropic-ai/claude-code
    fi
}

case "$OS" in
    Linux)                install_linux ;;
    Darwin)               install_macos ;;
    MINGW*|MSYS*|CYGWIN*) install_windows ;;
    *)                    fail "unsupported OS: $OS (Linux, macOS, and Windows only)" ;;
esac

log "verifying installations"

check() {
    name="$1"; shift
    if "$@" >/dev/null 2>&1; then
        printf '  \033[1;32mok\033[0m   %-8s — %s\n' "$name" "$("$@" 2>&1 | head -n 1)"
    else
        warn "$name check failed: $*"
        return 1
    fi
}

rc=0
check make   make --version    || rc=1
check claude claude --version  || rc=1
check docker docker --version  || rc=1

if ! docker info >/dev/null 2>&1; then
    # Bare `docker info` may fail because (a) the daemon is starting, or
    # (b) usermod just added us to the docker group and that membership
    # isn't active in this shell. wait_for_docker uses a privileged path
    # (sudo / sg docker) so it isolates the daemon-up question from the
    # group-membership question.
    if wait_for_docker 60; then
        if docker info >/dev/null 2>&1; then
            log "docker daemon reachable"
        else
            current_user="$(id -un)"
            # `id -nG <user>` reads /etc/group (configured membership);
            # plain `id -nG` reads the current process's group set
            # (snapshotted at login). When the former contains "docker"
            # but the latter doesn't, the user is in the group on disk
            # — they just need to start a new login shell for it to take
            # effect. This also covers re-runs where DOCKER_GROUP_ADDED
            # is 0 because membership was established on a prior run.
            if id -nG "$current_user" 2>/dev/null | tr ' ' '\n' | grep -qx docker; then
                log "docker daemon up — current shell's group set doesn't yet include 'docker'"
                # We can't update the parent shell's groups from here.
                # Best alternative: exec into `newgrp docker` so the same
                # process becomes a docker-group-active shell. CWD is
                # preserved — `make build` will work immediately. Note
                # that SSH connection multiplexing (ControlMaster) can
                # cause even a "fresh" SSH login to inherit the original
                # session's stale groups, so newgrp is the most reliable
                # fix.
                if [ -t 0 ] && [ -t 1 ] && command -v newgrp >/dev/null 2>&1 && [ "$rc" -eq 0 ]; then
                    log "all good — entering a 'newgrp docker' shell so docker is usable here"
                    log "    CWD is preserved; type 'exit' to return to your previous shell"
                    exec newgrp docker
                fi
                log "run 'newgrp docker' to activate the docker group in this shell, then retry"
            else
                warn "docker daemon up but '$current_user' is not in the 'docker' group"
                warn "  run: sudo usermod -aG docker $current_user  then log out/in"
                rc=1
            fi
        fi
    else
        warn "docker daemon not reachable after 60s"
        warn "  Linux:   check 'sudo systemctl status docker' and 'sudo journalctl -u docker --no-pager -n 50'"
        warn "  macOS:   open Docker Desktop, accept any first-run dialog, then re-run this script"
        warn "  Windows: open Docker Desktop, accept any first-run dialog, then re-run this script"
        rc=1
    fi
fi

[ $rc -eq 0 ] && log "all good" || fail "one or more checks failed"
