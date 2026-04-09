#!/bin/bash
# Run tests in batches to avoid PyTorch segfault when 160+ tests
# share a single process (torch memory fragmentation issue).
VENV=".venv/bin/python"
PASS=0
FAIL=0
TOTAL=0

run_batch() {
    local label="$1"
    shift
    echo "=== $label ==="
    if $VENV -m pytest "$@" -q 2>&1; then
        PASS=$((PASS + 1))
    else
        FAIL=$((FAIL + 1))
        echo "FAILED: $label"
    fi
    TOTAL=$((TOTAL + 1))
}

# Batch 1: Brain core (neurons, synapses, modulators, core, persistence)
run_batch "Brain Core" tests/test_neurons.py tests/test_synapses.py tests/test_rstdp.py tests/test_modulators.py tests/test_core.py tests/test_persistence.py

# Batch 2a: Pure sensor tests (no torch)
run_batch "Sensors (pure)" tests/test_sensor_app.py tests/test_sensor_bus.py tests/test_sensor_idle.py tests/test_sensor_keymouse.py tests/test_sensor_mic.py tests/test_sensor_time.py

# Batch 2b: Adapter tests (torch + pyobjc — segfault-prone combo)
run_batch "Adapter (encoding)" tests/test_mac_adapter.py

# Batch 2c: Brain+Adapter integration
run_batch "Brain+Adapter" tests/test_brain_with_adapter.py

# Batch 3: Server + API
run_batch "Server & API" tests/test_server.py tests/test_chat_endpoint.py tests/test_braind_cli.py tests/test_llm_router.py

# Batch 4: Bridge + Tools
run_batch "Bridge & Tools" tests/test_exporter.py tests/test_memory_tools.py tests/test_episode_log.py tests/test_stt.py

# Batch 5: Capabilities
run_batch "Capabilities" tests/test_grants.py tests/test_grant_endpoints.py tests/test_capability_tools.py tests/test_tool_registry.py

# Batch 6: E2E (skip if port in use)
if ! lsof -i :8765 -sTCP:LISTEN > /dev/null 2>&1; then
    run_batch "E2E" tests/test_e2e_smoke.py
else
    echo "=== E2E === SKIPPED (port 8765 in use)"
fi

echo ""
echo "========================================"
echo "Results: $PASS/$TOTAL batches passed"
if [ $FAIL -gt 0 ]; then
    echo "$FAIL batch(es) FAILED"
    exit 1
fi
echo "All batches PASSED"
