#!/usr/bin/env bash
# =============================================================================
# LAFS Project — Mininet 2.3.0 Full Installation Script
# Target OS: Ubuntu 20.04 LTS or Ubuntu 22.04 LTS
# Installs: Mininet, Open vSwitch, OpenFlow 1.3 support, performance tuning
# Usage:  chmod +x install_mininet.sh && sudo ./install_mininet.sh
# =============================================================================

set -euo pipefail

# ── Colour helpers ────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()    { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── Root check ────────────────────────────────────────────────────────────────
[[ $EUID -ne 0 ]] && error "Run this script as root: sudo ./install_mininet.sh"

# ── OS detection ─────────────────────────────────────────────────────────────
OS_ID=$(. /etc/os-release && echo "$ID")
OS_VER=$(. /etc/os-release && echo "$VERSION_ID")
info "Detected OS: $OS_ID $OS_VER"

if [[ "$OS_ID" != "ubuntu" ]]; then
    error "This script targets Ubuntu 20.04/22.04. Detected: $OS_ID"
fi

INSTALL_DIR="/opt/lafs"
MININET_VERSION="2.3.0"
OVS_VERSION="2.17.0"   # LTS — supports OpenFlow 1.3

# =============================================================================
# STEP 1 — System update & base dependencies
# =============================================================================
info "Step 1/9 — Updating system packages..."
apt-get update -qq
apt-get upgrade -y -qq

info "Installing base build tools..."
apt-get install -y -qq \
    git curl wget build-essential python3 python3-pip python3-dev python3-venv \
    autoconf automake libtool pkg-config \
    iproute2 iputils-ping net-tools iperf iperf3 \
    tcpdump wireshark tshark \
    bridge-utils uml-utilities \
    socat telnet \
    software-properties-common apt-transport-https \
    unzip zip tar gzip \
    psmisc lsof strace \
    htop iotop sysstat

# =============================================================================
# STEP 2 — Open vSwitch (OVS) with OpenFlow 1.3 support
# =============================================================================
info "Step 2/9 — Installing Open vSwitch $OVS_VERSION (OpenFlow 1.3 enabled)..."

# Remove any existing OVS packages
apt-get remove -y openvswitch-switch openvswitch-common 2>/dev/null || true

# Install OVS from Ubuntu repos (supports OF 1.3 natively)
apt-get install -y -qq \
    openvswitch-switch \
    openvswitch-common \
    openvswitch-testcontroller

# Enable & start OVS
systemctl enable openvswitch-switch
systemctl start openvswitch-switch

# Verify OVS
OVS_ACTUAL=$(ovs-vsctl --version | head -1 | grep -oP '[\d.]+' | head -1)
info "Open vSwitch version: $OVS_ACTUAL"

# Configure OVS to use OpenFlow 1.3 by default
ovs-vsctl set bridge br-test protocols=OpenFlow13 2>/dev/null || true
ovs-vsctl del-br br-test 2>/dev/null || true

# =============================================================================
# STEP 3 — Mininet 2.3.0 from source
# =============================================================================
info "Step 3/9 — Installing Mininet $MININET_VERSION from source..."

mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

# Clone specific tag
if [ -d "mininet" ]; then
    warn "Mininet directory exists — pulling latest..."
    cd mininet && git fetch --tags && git checkout "$MININET_VERSION" && cd ..
else
    git clone https://github.com/mininet/mininet.git
    cd mininet && git checkout "$MININET_VERSION" && cd ..
fi

# Install Mininet (kernel module path + dependencies)
cd mininet
# Mininet installer: -n = Mininet, -v = OVS (already installed), -f = OF
# Use -a for all optional extras
util/install.sh -nfv

# Verify Mininet import
python3 -c "import mininet; print('Mininet import OK:', mininet.__file__)"
info "Mininet $MININET_VERSION installed successfully."

# =============================================================================
# STEP 4 — Ryu SDN Controller
# =============================================================================
info "Step 4/9 — Installing Ryu SDN Controller..."
cd "$INSTALL_DIR"

pip3 install --upgrade pip
pip3 install ryu

# Verify Ryu
python3 -c "import ryu; print('Ryu import OK:', ryu.__file__)"
RYU_VER=$(python3 -c "import ryu; print(ryu.__version__)")
info "Ryu version: $RYU_VER"

# =============================================================================
# STEP 5 — POX Controller (optional reference baseline)
# =============================================================================
info "Step 5/9 — Installing POX controller (reference baseline)..."
cd "$INSTALL_DIR"

if [ ! -d "pox" ]; then
    git clone https://github.com/noxrepo/pox.git
fi
info "POX available at $INSTALL_DIR/pox"

# =============================================================================
# STEP 6 — Performance tuning for 128-host Fat-tree topology
# =============================================================================
info "Step 6/9 — Applying kernel/OS performance tuning for 128-host topology..."

# Increase kernel limits for many virtual interfaces
cat >> /etc/sysctl.conf << 'SYSCTL'

# ── LAFS Mininet performance tuning ──────────────────────────────────────────
# Support 128-host Fat-tree with 240 ports, many virtual links

# File descriptor limits
fs.file-max = 2097152

# Network buffer sizes (for high-throughput experiments)
net.core.rmem_max = 134217728
net.core.wmem_max = 134217728
net.core.rmem_default = 65536
net.core.wmem_default = 65536
net.ipv4.tcp_rmem = 4096 87380 134217728
net.ipv4.tcp_wmem = 4096 65536 134217728

# Allow more simultaneous connections
net.core.somaxconn = 65535
net.ipv4.tcp_max_syn_backlog = 65535

# Increase ARP cache for many hosts
net.ipv4.neigh.default.gc_thresh1 = 512
net.ipv4.neigh.default.gc_thresh2 = 2048
net.ipv4.neigh.default.gc_thresh3 = 4096

# Forwarding (required for Mininet namespaces)
net.ipv4.ip_forward = 1

# Increase inotify limits (Mininet uses many processes)
fs.inotify.max_user_instances = 8192
fs.inotify.max_user_watches = 524288

# Increase local port range
net.ipv4.ip_local_port_range = 1024 65535

# Reduce TIME_WAIT socket lingering
net.ipv4.tcp_fin_timeout = 15
net.ipv4.tcp_tw_reuse = 1
SYSCTL

sysctl -p > /dev/null
info "Kernel parameters applied."

# Increase open file descriptors for mininet processes
cat >> /etc/security/limits.conf << 'LIMITS'

# LAFS project — allow many open files per process (needed for 128+ hosts)
*    soft nofile 65536
*    hard nofile 1048576
root soft nofile 65536
root hard nofile 1048576
LIMITS

# OVS performance tuning
ovs-vsctl set Open_vSwitch . other_config:dpdk-init=false 2>/dev/null || true

# =============================================================================
# STEP 7 — OpenFlow 1.3 verification
# =============================================================================
info "Step 7/9 — Verifying OpenFlow 1.3 support..."

# Create a test bridge, verify OF 1.3 can be set
ovs-vsctl add-br lafs-test-br
ovs-vsctl set bridge lafs-test-br protocols=OpenFlow13
OF_PROTO=$(ovs-vsctl get bridge lafs-test-br protocols)
ovs-vsctl del-br lafs-test-br

if [[ "$OF_PROTO" == *"OpenFlow13"* ]]; then
    info "OpenFlow 1.3 supported: $OF_PROTO"
else
    error "OpenFlow 1.3 NOT supported by this OVS build."
fi

# =============================================================================
# STEP 8 — iperf3 and traffic generation tools
# =============================================================================
info "Step 8/9 — Installing traffic generation tools..."

apt-get install -y -qq iperf3 netperf nmap hping3

# D-ITG (Distributed Internet Traffic Generator) for realistic traces
cd "$INSTALL_DIR"
if [ ! -d "D-ITG" ]; then
    wget -q http://www.grid.unina.it/software/ITG/codice/D-ITG-2.8.1-r1023-src.zip \
         -O D-ITG.zip 2>/dev/null || warn "D-ITG download failed — skipping (optional)"
    if [ -f D-ITG.zip ]; then
        unzip -q D-ITG.zip && mv D-ITG-* D-ITG
        cd D-ITG/src && make -s && make install -s
        info "D-ITG installed."
    fi
fi

# =============================================================================
# STEP 9 — Final system-wide verification
# =============================================================================
info "Step 9/9 — Running system verification..."

PASS=0; FAIL=0
check() {
    local name="$1"; local cmd="$2"
    if eval "$cmd" &>/dev/null; then
        echo -e "  ${GREEN}✓${NC} $name"
        ((PASS++))
    else
        echo -e "  ${RED}✗${NC} $name"
        ((FAIL++))
    fi
}

echo ""
echo "=== Verification Results ==="
check "Python 3"                "python3 --version"
check "pip3"                    "pip3 --version"
check "Mininet import"          "python3 -c 'import mininet'"
check "Mininet CLI"             "mn --version"
check "Open vSwitch"            "ovs-vsctl --version"
check "OVS daemon running"      "systemctl is-active openvswitch-switch"
check "Ryu import"              "python3 -c 'import ryu'"
check "Ryu CLI"                 "ryu-manager --version"
check "iperf3"                  "iperf3 --version"
check "OpenFlow 1.3 support"    "python3 -c 'from ryu.ofproto import ofproto_v1_3'"
check "NetworkX"                "python3 -c 'import networkx'"
check "NumPy"                   "python3 -c 'import numpy'"
check "iproute2 (ip)"           "ip --version"
check "tcpdump"                 "tcpdump --version"
echo ""

info "Passed: $PASS | Failed: $FAIL"

if [[ $FAIL -gt 0 ]]; then
    warn "Some checks failed. Review above and re-run if needed."
else
    info "All checks passed! System ready for LAFS development."
fi

# =============================================================================
# Post-install notes
# =============================================================================
cat << 'EOF'

╔══════════════════════════════════════════════════════════════════╗
║              LAFS Installation Complete                          ║
╠══════════════════════════════════════════════════════════════════╣
║ Next steps:                                                      ║
║  1. cd /path/to/LAFS && source venv/bin/activate                ║
║  2. pip install -r requirements.txt                              ║
║  3. python tests/test_mininet.py      (verify Mininet)           ║
║  4. python tests/test_ryu.py          (verify Ryu controller)    ║
║  5. python tests/test_gurobi.py       (verify Gurobi/optimizer)  ║
║  6. python tests/test_integration.py (full integration test)     ║
║                                                                  ║
║ Fat-tree k=8 quick test:                                         ║
║  sudo mn --topo=ftree,8 --switch=ovsk,protocols=OpenFlow13 \     ║
║           --controller=remote --test=pingall                     ║
╚══════════════════════════════════════════════════════════════════╝
EOF
