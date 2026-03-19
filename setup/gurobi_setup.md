# Gurobi Academic License — Setup Guide
**LAFS Project, COMP 6910 — Group 7**

Gurobi is a commercial MILP solver that offers **free academic licenses** for students and researchers. This guide covers the full setup from license registration to Python binding verification.

> **Fallback:** If Gurobi cannot be licensed in time, the project uses **PuLP + CBC** (open-source, already installed). Performance will differ but the approach remains valid.

---

## 1. Register for an Academic License

1. Go to: https://www.gurobi.com/academia/academic-program-and-licenses/
2. Click **"Request a Free Academic License"**
3. Register with your **university email** (e.g., `@concordia.ca`)
4. Verify your email — you will receive a **license key** (looks like: `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`)

> Concordia University qualifies for the Named-User Academic License (single machine, free).

---

## 2. Download & Install Gurobi

### Ubuntu 20.04 / 22.04

```bash
# Download Gurobi 11.0 (matches gurobipy==11.0.3 in requirements.txt)
cd /opt
sudo wget https://packages.gurobi.com/11.0/gurobi11.0.3_linux64.tar.gz
sudo tar -xzf gurobi11.0.3_linux64.tar.gz
sudo mv gurobi1103 gurobi11

# Set environment variables (add to ~/.bashrc)
echo 'export GUROBI_HOME="/opt/gurobi11/linux64"'     >> ~/.bashrc
echo 'export PATH="${PATH}:${GUROBI_HOME}/bin"'        >> ~/.bashrc
echo 'export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${GUROBI_HOME}/lib"' >> ~/.bashrc
source ~/.bashrc
```

### Verify binary installation
```bash
gurobi_cl --version
# Expected: Gurobi 11.0.3 ...
```

---

## 3. Activate the License

```bash
# Replace <YOUR_KEY> with the key from your Gurobi account
grbgetkey <YOUR_KEY>

# Follow prompts:
#   License file location: press Enter to accept default (/opt/gurobi/gurobi.lic)
#   OR specify: /home/<username>/gurobi.lic
```

The license file (`gurobi.lic`) will be saved to your home directory or the path you specified.

### Set license path (if not in default location)
```bash
# Add to ~/.bashrc
echo 'export GRB_LICENSE_FILE="/home/$USER/gurobi.lic"' >> ~/.bashrc
source ~/.bashrc
```

---

## 4. Install the Python Binding

```bash
# Activate LAFS venv first
source venv/bin/activate

# Install gurobipy matching your Gurobi version
pip install gurobipy==11.0.3

# OR install directly from Gurobi installation (alternative)
cd /opt/gurobi11/linux64
pip install -e .
```

---

## 5. Verify Gurobi Works

```bash
python tests/test_gurobi.py
```

Or manually:

```python
import gurobipy as gp
from gurobipy import GRB

# Create a minimal MILP model (like LAFS flow placement)
m = gp.Model("lafs_test")
m.setParam("OutputFlag", 1)

# Variables: flow assignment binary
x = m.addVar(vtype=GRB.BINARY, name="x")
y = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="y")

# Objective: minimize y (proxy for FCT)
m.setObjective(y, GRB.MINIMIZE)

# Constraints
m.addConstr(y >= 2 * x, "fct_constraint")
m.addConstr(x + y >= 1, "feasibility")

m.optimize()

print(f"Status  : {m.status}")          # Should be 2 (Optimal)
print(f"Obj val : {m.ObjVal:.4f}")      # Should be 0.0 or 1.0
print(f"x = {x.X:.0f},  y = {y.X:.4f}")
```

**Expected output:**
```
Gurobi 11.0.3 ...
Optimize a model with 2 rows, 2 columns and 3 nonzeros ...
Optimal solution found
Status  : 2
Obj val : 0.0000
```

---

## 6. Troubleshooting

### Error: `No Gurobi license found`
```bash
# Check license file exists
ls -la ~/gurobi.lic
cat ~/gurobi.lic   # should show LICENSE, HOST, etc.

# Verify GRB_LICENSE_FILE is set
echo $GRB_LICENSE_FILE

# Re-run license activation
grbgetkey <YOUR_KEY>
```

### Error: `gurobipy not found`
```bash
source venv/bin/activate
pip install gurobipy==11.0.3
```

### Error: `License expired` or `License validation failed`
- Academic Named-User licenses are valid for **1 year** — renew at gurobi.com
- Ensure you are on a network where the license server can validate (Concordia VPN or campus network may be required)

### Error: `No module named 'gurobipy'` inside Mininet namespace
- Mininet host namespaces share the host kernel but NOT the Python venv
- Run the MILP optimizer on the controller process (outside Mininet), which is the correct LAFS architecture

---

## 7. Using PuLP / CBC as Fallback

If Gurobi is unavailable, the LAFS optimizer automatically falls back to PuLP:

```python
# In src/optimizer/milp_solver.py — controlled by LAFS_SOLVER env var
import os
SOLVER = os.environ.get("LAFS_SOLVER", "gurobi")

if SOLVER == "gurobi":
    import gurobipy as gp
    # ... Gurobi model
else:
    import pulp
    # ... PuLP model
```

To switch to PuLP:
```bash
export LAFS_SOLVER=pulp
# OR edit .env: LAFS_SOLVER=pulp
```

> **Note for report:** Gurobi typically solves 1000-variable MILPs 10–100× faster than CBC. If using PuLP, expect longer solve times but the same mathematical formulation and correctness.

---

## 8. MILP Solver Performance Comparison

| Solver | Type | Speed (1000 vars) | License |
|---|---|---|---|
| **Gurobi 11** | Commercial | ~50ms | Free academic |
| PuLP + CBC | Open-source | ~1–10s | Open source |
| scipy linprog | LP only | ~5ms | Open source |
| CPLEX (alt) | Commercial | ~60ms | Free academic |

For our <100ms target from the proposal (§3.2), Gurobi is strongly preferred.
