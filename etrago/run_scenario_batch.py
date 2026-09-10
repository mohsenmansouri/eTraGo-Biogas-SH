"""Run selected eTraGo scenarios sequentially in isolated processes."""

from __future__ import annotations

import argparse
import copy
import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scenario_config import ScenarioConfigError, load_config, resolve_config


SELECTION_ENVIRONMENT_VARIABLES = {
    "fossil_gas_price_case": "ETRAGO_FOSSIL_GAS_PRICE_CASE",
    "biomethane_price_case": "ETRAGO_BIOMETHANE_PRICE_CASE",
    "heat_pump_case": "ETRAGO_HEAT_PUMP_CASE",
    "swfl_unit_case": "ETRAGO_SWFL_UNIT_CASE",
    "biomethane_use_case": "ETRAGO_BIOMETHANE_USE_CASE",
    "biogas_route_case": "ETRAGO_BIOGAS_ROUTE_CASE",
}


def _without_selection_environment_overrides() -> None:
    """Make the YAML batch authoritative in the launcher process."""
    for variable in SELECTION_ENVIRONMENT_VARIABLES.values():
        os.environ.pop(variable, None)


def _scenario_overrides(raw: Mapping[str, Any], index: int) -> dict[str, str]:
    """Return one normalized batch entry."""
    nested = raw.get("selection")
    if nested is not None:
        if not isinstance(nested, Mapping):
            raise ScenarioConfigError(
                f"batch.scenarios[{index}].selection must be a mapping."
            )
        overrides = dict(nested)
    else:
        overrides = {
            key: value
            for key, value in raw.items()
            if key != "name"
        }

    unknown = sorted(
        set(overrides) - set(SELECTION_ENVIRONMENT_VARIABLES)
    )
    if unknown:
        raise ScenarioConfigError(
            f"Unknown selection keys in batch.scenarios[{index}]: "
            f"{', '.join(unknown)}"
        )

    return {str(key): str(value) for key, value in overrides.items()}


def resolve_batch(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Merge, validate, and resolve every configured batch scenario."""
    batch = config.get("batch", {})
    if not isinstance(batch, Mapping):
        raise ScenarioConfigError("'batch' must be a mapping.")
    if not bool(batch.get("enabled", False)):
        raise ScenarioConfigError(
            "Batch execution is disabled. Set batch.enabled to true."
        )

    raw_scenarios = batch.get("scenarios", [])
    if not isinstance(raw_scenarios, Sequence) or isinstance(
        raw_scenarios,
        (str, bytes),
    ):
        raise ScenarioConfigError("batch.scenarios must be a list.")
    if not raw_scenarios:
        raise ScenarioConfigError("batch.scenarios must not be empty.")

    base_selection = config.get("selection", {})
    if not isinstance(base_selection, Mapping):
        raise ScenarioConfigError("'selection' must be a mapping.")

    resolved_scenarios: list[dict[str, Any]] = []
    seen_names: set[str] = set()

    for index, raw in enumerate(raw_scenarios, start=1):
        if not isinstance(raw, Mapping):
            raise ScenarioConfigError(
                f"batch.scenarios[{index}] must be a mapping."
            )

        candidate = copy.deepcopy(dict(config))
        candidate_selection = {
            str(key): str(value)
            for key, value in base_selection.items()
        }
        candidate_selection.update(_scenario_overrides(raw, index))
        candidate["selection"] = candidate_selection

        resolved = resolve_config(candidate)
        scenario_name = str(resolved["scenario_name"])
        if scenario_name in seen_names:
            raise ScenarioConfigError(
                "Batch scenarios must be unique; duplicate resolved name: "
                f"{scenario_name}"
            )
        seen_names.add(scenario_name)

        resolved_scenarios.append(
            {
                "label": str(raw.get("name", scenario_name)),
                "scenario_name": scenario_name,
                "selection": dict(resolved["selection"]),
            }
        )

    return resolved_scenarios


def _child_environment(selection: Mapping[str, str]) -> dict[str, str]:
    """Build a complete, explicit environment for one appl.py run."""
    environment = os.environ.copy()
    for key, variable in SELECTION_ENVIRONMENT_VARIABLES.items():
        environment[variable] = str(selection[key])
    return environment


def run_batch(
    config_path: Path,
    appl_path: Path,
    *,
    dry_run: bool = False,
    keep_going: bool = False,
) -> int:
    """Run all configured scenarios and return a process exit code."""
    _without_selection_environment_overrides()
    config = load_config(config_path)
    scenarios = resolve_batch(config)

    if appl_path.name == "appl.py":
        expected_config_path = appl_path.with_name("config.yaml").resolve()
        if config_path.resolve() != expected_config_path:
            raise ScenarioConfigError(
                "appl.py loads config.yaml from its own directory. "
                f"Use that file for the batch: {expected_config_path}"
            )

    batch = config["batch"]
    stop_on_error = bool(batch.get("stop_on_error", True)) and not keep_going

    if not dry_run and not appl_path.is_file():
        raise ScenarioConfigError(f"appl.py not found: {appl_path}")

    command = [sys.executable, "-u", str(appl_path)]
    failures: list[tuple[str, int]] = []
    total = len(scenarios)

    print(f"Validated {total} batch scenarios.", flush=True)

    for number, scenario in enumerate(scenarios, start=1):
        label = scenario["label"]
        scenario_name = scenario["scenario_name"]
        print("\n" + "=" * 72, flush=True)
        print(f"BATCH {number}/{total}: {label}", flush=True)
        print(f"Resolved name: {scenario_name}", flush=True)
        print("=" * 72, flush=True)

        if dry_run:
            continue

        started = time.monotonic()
        completed = subprocess.run(
            command,
            cwd=appl_path.parent,
            env=_child_environment(scenario["selection"]),
            check=False,
        )
        elapsed_minutes = (time.monotonic() - started) / 60.0

        if completed.returncode == 0:
            print(
                f"PASS: {label} ({elapsed_minutes:.1f} min)",
                flush=True,
            )
            continue

        failures.append((label, completed.returncode))
        print(
            f"FAIL: {label} exited with code {completed.returncode} "
            f"after {elapsed_minutes:.1f} min.",
            file=sys.stderr,
            flush=True,
        )
        if stop_on_error:
            print("Stopping because batch.stop_on_error is true.", flush=True)
            break

    print("\nBatch summary", flush=True)
    print(f"  configured: {total}", flush=True)
    print(f"  attempted:  {number}", flush=True)
    print(f"  failed:     {len(failures)}", flush=True)
    for label, return_code in failures:
        print(f"    - {label}: exit code {return_code}", flush=True)

    return 1 if failures else 0


def _main() -> None:
    script_directory = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Run configured eTraGo scenarios sequentially."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=script_directory / "config.yaml",
        help="Scenario YAML (default: config.yaml beside this script).",
    )
    parser.add_argument(
        "--appl",
        type=Path,
        default=script_directory / "appl.py",
        help="eTraGo entry point (default: appl.py beside this script).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and list scenarios without starting eTraGo.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Try remaining scenarios after a failed run.",
    )
    arguments = parser.parse_args()

    try:
        return_code = run_batch(
            arguments.config.resolve(),
            arguments.appl.resolve(),
            dry_run=arguments.dry_run,
            keep_going=arguments.keep_going,
        )
    except ScenarioConfigError as error:
        parser.error(str(error))

    raise SystemExit(return_code)


if __name__ == "__main__":
    _main()
