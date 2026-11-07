#!/usr/bin/env bash
# Run a command against an untrusted repository inside an isolated sandbox.
#   sbx.sh <workdir> [--site DIR] [--net] -- <cmd...>
# The workdir is mounted writable at /mnt (cwd); --site DIR is mounted read-only
# at /opt and put on PYTHONPATH (or writable with --net, for installing into it).
# Without --net there is no network. The root filesystem is read-only; /tmp,
# /home and /root are private. Install packages with:
#   sbx.sh <workdir> --site <site> --net -- python3 -m pip install --target /opt <pkgs>
set -euo pipefail
work=$(realpath "$1"); shift
site=""; net=0
while [ $# -gt 0 ] && [ "$1" != "--" ]; do
  case "$1" in --site) site=$(realpath "$2"); shift 2 ;; --net) net=1; shift ;; *) echo "unknown $1" >&2; exit 2 ;; esac
done
shift
mkdir -p "${site:-/nonexistent-ok}" 2>/dev/null || true
nflag="-n"; [ $net = 1 ] && nflag=""
ro="mount -o remount,bind,ro /opt;"; [ $net = 1 ] && ro=""
siteMount=""; [ -n "$site" ] && siteMount="mount --bind '$site' /opt; $ro"
# With network (installing only), the proxy's CA bundle is bound over the system bundle so /root can stay hidden.
ca=""; caEnv=""
if [ $net = 1 ] && [ -n "${SSL_CERT_FILE:-}" ] && [ -f "$SSL_CERT_FILE" ]; then ca="mount --bind '$SSL_CERT_FILE' /etc/ssl/certs/ca-certificates.crt;"; caEnv="SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt PIP_CERT=/etc/ssl/certs/ca-certificates.crt REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt"; fi
exec unshare -r $nflag -m --fork env -i PATH=/usr/local/bin:/usr/bin:/bin HOME=/tmp LANG=C.UTF-8 PYTHONPATH=/opt:/mnt \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PIP_CACHE_DIR=/var/cache/mlh-pip \
  $( [ $net = 1 ] && echo "${HTTPS_PROXY:+HTTPS_PROXY=$HTTPS_PROXY}" ) $caEnv \
  sh -c "set -e; mount --bind '$work' /mnt; $siteMount $ca mount -t tmpfs none /tmp; mount -t tmpfs none /home; mount -t tmpfs none /root; mount -o remount,bind,ro / 2>/dev/null || true; cd /mnt; exec \"\$@\"" sh "$@"
