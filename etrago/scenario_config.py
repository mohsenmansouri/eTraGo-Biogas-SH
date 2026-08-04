"""Scenario configuration helpers for the Biogas.SH / SWFL eTraGo model."""

from __future__ import annotations

import argparse
import copy
import csv
import itertools
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Sequence, Tuple

import yaml


CONFIG_VERSION = 2


class ScenarioConfigError(ValueError):
    """Raised when the scenario configuration is incomplete or inconsistent."""


def _require(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ScenarioConfigError(f"Missing '{key}' in {context}.")
    return mapping[key]


def _require_mapping(
    mapping: Mapping[str, Any],
    key: str,
    context: str,
) -> Mapping[str, Any]:
    value = _require(mapping, key, context)
    if not isinstance(value, Mapping):
        raise ScenarioConfigError(f"'{key}' in {context} must be a mapping.")
    return value


def _named_case(
    parent: Mapping[str, Any],
    section: str,
    name: str,
) -> Mapping[str, Any]:
    cases = _require_mapping(parent, section, "configuration")
    if name not in cases:
        options = ", ".join(sorted(map(str, cases)))
        raise ScenarioConfigError(
            f"Unknown case '{name}' in '{section}'. Available: {options}"
        )
    value = cases[name]
    if not isinstance(value, Mapping):
        raise ScenarioConfigError(
            f"Case '{section}.{name}' must be a mapping."
        )
    return value


def _selection_with_environment_overrides(
    config: Mapping[str, Any],
) -> Dict[str, str]:
    selection = {
        str(key): str(value)
        for key, value in _require_mapping(
            config,
            "selection",
            "configuration",
        ).items()
    }

    environment_variables = {
        "natural_gas_price_case": "ETRAGO_NATURAL_GAS_PRICE_CASE",
        "biomethane_price_case": "ETRAGO_BIOMETHANE_PRICE_CASE",
        "heat_pump_case": "ETRAGO_HEAT_PUMP_CASE",
        "swfl_unit_case": "ETRAGO_SWFL_UNIT_CASE",
        "biomethane_use_case": "ETRAGO_BIOMETHANE_USE_CASE",
        "biogas_route_case": "ETRAGO_BIOGAS_ROUTE_CASE",
    }

    for key, variable in environment_variables.items():
        override = os.getenv(variable)
        if override:
            selection[key] = override

    return selection


def _validate_selection(
    config: Mapping[str, Any],
    selection: Mapping[str, str],
) -> None:
    required_dimensions = (
        "natural_gas_price_case",
        "biomethane_price_case",
        "heat_pump_case",
        "swfl_unit_case",
        "biomethane_use_case",
        "biogas_route_case",
    )
    missing = [key for key in required_dimensions if key not in selection]
    if missing:
        raise ScenarioConfigError(
            f"Missing selection dimensions: {', '.join(missing)}"
        )

    price_cases = _require_mapping(config, "price_cases", "configuration")
    _named_case(
        price_cases,
        "natural_gas",
        selection["natural_gas_price_case"],
    )
    _named_case(
        price_cases,
        "biomethane",
        selection["biomethane_price_case"],
    )

    heat_pump_case = _named_case(
        config,
        "heat_pump_cases",
        selection["heat_pump_case"],
    )
    unit_case = _named_case(
        config,
        "swfl_unit_cases",
        selection["swfl_unit_case"],
    )
    biomethane_case = _named_case(
        config,
        "biomethane_use_cases",
        selection["biomethane_use_case"],
    )
    route_case = _named_case(
        config,
        "biogas_route_cases",
        selection["biogas_route_case"],
    )

    technical = _require_mapping(config, "technical", "configuration")
    swfl = _require_mapping(technical, "swfl", "technical")
    biogas = _require_mapping(technical, "biogas_sh", "technical")

    available_heat_pumps = set(
        _require_mapping(swfl, "heat_pumps", "technical.swfl")
    )
    available_boilers = set(
        _require_mapping(swfl, "boilers", "technical.swfl")
    )
    available_resistive = set(
        _require_mapping(swfl, "resistive_heaters", "technical.swfl")
    )

    selected_heat_pumps = set(map(str, heat_pump_case.get("active_units", [])))
    selected_boilers = set(map(str, unit_case.get("boilers", [])))
    selected_resistive = set(map(str, unit_case.get("resistive_heaters", [])))
    eligible_biomethane = set(
        map(str, biomethane_case.get("eligible_units", []))
    )

    unknown_heat_pumps = selected_heat_pumps - available_heat_pumps
    unknown_boilers = selected_boilers - available_boilers
    unknown_resistive = selected_resistive - available_resistive

    if unknown_heat_pumps:
        raise ScenarioConfigError(
            "Unknown heat pumps in selected case: "
            f"{sorted(unknown_heat_pumps)}"
        )
    if unknown_boilers:
        raise ScenarioConfigError(
            f"Unknown boilers in selected case: {sorted(unknown_boilers)}"
        )
    if unknown_resistive:
        raise ScenarioConfigError(
            "Unknown resistive heaters in selected case: "
            f"{sorted(unknown_resistive)}"
        )

    inactive_eligible = eligible_biomethane - selected_boilers
    if inactive_eligible:
        raise ScenarioConfigError(
            "Biomethane-eligible boilers are inactive in the selected "
            f"SWFL unit case: {sorted(inactive_eligible)}"
        )

    storage = _require_mapping(biogas, "storage", "technical.biogas_sh")
    storage_required = bool(
        route_case.get("add_gas_grid_generation", False)
        or route_case.get("add_swfl_direct_supply", False)
    )
    if storage_required and not bool(storage.get("active", False)):
        raise ScenarioConfigError(
            "The selected Biogas.SH route requires central storage, "
            "but technical.biogas_sh.storage.active is false."
        )


def validate_config(config: Mapping[str, Any]) -> None:
    """Validate the selected scenario and the configured matrix dimensions."""
    _validate_selection(config, _selection_with_environment_overrides(config))

    matrix = config.get("scenario_matrix", {})
    if not isinstance(matrix, Mapping) or not matrix.get("enabled", False):
        return

    dimensions = matrix.get("dimensions", {})
    if not isinstance(dimensions, Mapping) or not dimensions:
        raise ScenarioConfigError(
            "scenario_matrix.enabled is true, but no dimensions are defined."
        )

    base_selection = _selection_with_environment_overrides(config)
    for key, values in dimensions.items():
        if key not in base_selection:
            raise ScenarioConfigError(
                f"Unknown scenario-matrix dimension: {key}"
            )
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise ScenarioConfigError(
                f"scenario_matrix.dimensions.{key} must be a list."
            )
        for value in values:
            candidate = dict(base_selection)
            candidate[str(key)] = str(value)
            _validate_selection(config, candidate)


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load and validate a version-2 YAML scenario configuration."""
    config_path = Path(path)
    if not config_path.exists():
        raise ScenarioConfigError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ScenarioConfigError("The YAML root must be a mapping.")

    version = int(config.get("version", 0))
    if version != CONFIG_VERSION:
        raise ScenarioConfigError(
            f"Unsupported config version {version}; expected {CONFIG_VERSION}."
        )

    validate_config(config)
    return config


def resolve_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Resolve one selected scenario into a compact reproducibility record."""
    selection = _selection_with_environment_overrides(config)
    _validate_selection(config, selection)

    price_cases = _require_mapping(config, "price_cases", "configuration")
    natural_gas = _named_case(
        price_cases,
        "natural_gas",
        selection["natural_gas_price_case"],
    )
    biomethane = _named_case(
        price_cases,
        "biomethane",
        selection["biomethane_price_case"],
    )
    onsite = _require_mapping(price_cases, "onsite", "price_cases")
    onsite_electricity_cases = _require_mapping(
        onsite,
        "electricity_marginal_cost_eur_per_mwh",
        "price_cases.onsite",
    )
    onsite_case = str(
        _require(onsite, "selected_electricity_case", "price_cases.onsite")
    )
    if onsite_case not in onsite_electricity_cases:
        raise ScenarioConfigError(
            f"Unknown onsite electricity-price case: {onsite_case}"
        )

    ordered_dimensions = (
        "natural_gas_price_case",
        "biomethane_price_case",
        "heat_pump_case",
        "swfl_unit_case",
        "biomethane_use_case",
        "biogas_route_case",
    )
    scenario_name = "__".join(selection[key] for key in ordered_dimensions)

    return {
        "config_version": CONFIG_VERSION,
        "scenario_name": scenario_name,
        "selection": selection,
        "prices": {
            "natural_gas_swfl_import_adder_eur_per_mwh_hs": float(
                natural_gas["swfl_import_adder_eur_per_mwh_hs"]
            ),
            "biomethane_marginal_cost_eur_per_mwh_hs": float(
                biomethane["marginal_cost_eur_per_mwh_hs"]
            ),
            "onsite_electricity_marginal_cost_eur_per_mwh": float(
                onsite_electricity_cases[onsite_case]
            ),
            "onsite_heat_marginal_cost_eur_per_mwh": float(
                onsite["heat_marginal_cost_eur_per_mwh"]
            ),
        },
        "heat_pumps": copy.deepcopy(
            _named_case(
                config,
                "heat_pump_cases",
                selection["heat_pump_case"],
            )
        ),
        "swfl_units": copy.deepcopy(
            _named_case(
                config,
                "swfl_unit_cases",
                selection["swfl_unit_case"],
            )
        ),
        "biomethane_use": copy.deepcopy(
            _named_case(
                config,
                "biomethane_use_cases",
                selection["biomethane_use_case"],
            )
        ),
        "biogas_routes": copy.deepcopy(
            _named_case(
                config,
                "biogas_route_cases",
                selection["biogas_route_case"],
            )
        ),
        "technical": copy.deepcopy(
            _require_mapping(config, "technical", "configuration")
        ),
        "run": copy.deepcopy(config.get("run", {})),
    }


def _mutable_mapping(
    mapping: MutableMapping[str, Any],
    key: str,
    context: str,
) -> MutableMapping[str, Any]:
    value = _require(mapping, key, context)
    if not isinstance(value, MutableMapping):
        raise ScenarioConfigError(f"'{key}' in {context} must be a mapping.")
    return value


def _update_named_assets(
    assets: Iterable[MutableMapping[str, Any]],
    selected_names: set[str],
    technical_data: Mapping[str, Mapping[str, Any]],
) -> None:
    by_name = {
        str(asset.get("name")): asset
        for asset in assets
        if isinstance(asset, MutableMapping) and asset.get("name") is not None
    }

    missing = selected_names - set(by_name)
    if missing:
        raise ScenarioConfigError(
            "Selected units are absent from the eTraGo args: "
            f"{sorted(missing)}"
        )

    for name, asset in by_name.items():
        asset["active"] = name in selected_names
        settings = technical_data.get(name, {})

        if "heat_capacity_mw" in settings:
            asset["heat_capacity_mw"] = float(settings["heat_capacity_mw"])
        if "efficiency" in settings:
            asset["efficiency"] = float(settings["efficiency"])
        if "cop" in settings:
            asset["cop"] = float(settings["cop"])
        if "planned_year" in settings:
            asset["planned_year"] = settings["planned_year"]


def _effective_ac_clusters(
    args: Mapping[str, Any],
) -> int | None:
    clustering = args.get(
        "network_clustering"
    )

    if not isinstance(
        clustering,
        Mapping,
    ):
        return None

    electricity_grid = clustering.get(
        "electricity_grid"
    )

    if not isinstance(
        electricity_grid,
        Mapping,
    ):
        return None

    value = electricity_grid.get(
        "n_clusters"
    )

    return (
        int(value)
        if value is not None
        else None
    )


def _apply_run_settings(
    args: MutableMapping[str, Any],
    resolved: MutableMapping[str, Any],
) -> None:
    """Apply snapshot, AC-clustering and result-directory settings."""
    run = resolved.get("run", {})

    if not isinstance(run, Mapping):
        raise ScenarioConfigError(
            "'run' must be a mapping."
        )

    start_override = run.get("start_snapshot")
    end_override = run.get("end_snapshot")
    ac_clusters_override = run.get("ac_clusters")

    if start_override is not None:
        args["start_snapshot"] = int(
            start_override
        )

    if end_override is not None:
        args["end_snapshot"] = int(
            end_override
        )

    if ac_clusters_override is not None:
        clustering = _mutable_mapping(
            args,
            "network_clustering",
            "eTraGo args",
        )

        electricity_grid = _mutable_mapping(
            clustering,
            "electricity_grid",
            "args.network_clustering",
        )

        electricity_grid["n_clusters"] = int(
            ac_clusters_override
        )

    start = args.get("start_snapshot")
    end = args.get("end_snapshot")

    represented_hours = None

    if start is not None and end is not None:
        start = int(start)
        end = int(end)

        represented_hours = end - start + 1

        if represented_hours <= 0:
            raise ScenarioConfigError(
                "end_snapshot must be greater than or "
                "equal to start_snapshot."
            )

    ac_clusters = _effective_ac_clusters(
        args
    )

    result_name_template = run.get(
        "result_name_template"
    )

    if result_name_template:
        if represented_hours is None:
            raise ScenarioConfigError(
                "result_name_template requires "
                "start_snapshot and end_snapshot."
            )

        if ac_clusters is None:
            raise ScenarioConfigError(
                "result_name_template requires a "
                "configured AC cluster count at "
                "args['network_clustering']"
                "['electricity_grid']['n_clusters']."
            )

        args["csv_export"] = str(
            result_name_template
        ).format(
            scenario=resolved["scenario_name"],
            hours=represented_hours,
            ac_clusters=ac_clusters,
        )

    resolved["effective_run"] = {
        "start_snapshot": start,
        "end_snapshot": end,
        "represented_hours": represented_hours,
        "ac_clusters": ac_clusters,
        "csv_export": args.get("csv_export"),
    }


def apply_config_to_args(
    args: MutableMapping[str, Any],
    resolved: MutableMapping[str, Any],
) -> MutableMapping[str, Any]:
    """
    Apply the resolved YAML scenario to the current top-level eTraGo args.

    Static topology, component names, carriers and file paths remain in appl.py.
    YAML controls scenario selections, prices, activation, capacities,
    efficiencies, storage assumptions and optional run settings.
    """
    swfl = _mutable_mapping(args, "swfl_real_system", "eTraGo args")
    biogas = _mutable_mapping(args, "biogas_sh", "eTraGo args")

    extra_functionality = args.get("extra_functionality")
    if not isinstance(extra_functionality, MutableMapping):
        extra_functionality = {}
        args["extra_functionality"] = extra_functionality

    prices = resolved["prices"]
    units = resolved["swfl_units"]
    heat_pump_case = resolved["heat_pumps"]
    biomethane_use = resolved["biomethane_use"]
    routes = resolved["biogas_routes"]
    technical = resolved["technical"]
    technical_swfl = technical["swfl"]
    technical_biogas = technical["biogas_sh"]

    # Prices.
    swfl_direct = _mutable_mapping(biogas, "swfl_direct", "args.biogas_sh")
    swfl_direct["grid_supply_marginal_cost"] = float(
        prices["natural_gas_swfl_import_adder_eur_per_mwh_hs"]
    )
    biogas["biomethane_price_override_eur_per_mwh"] = float(
        prices["biomethane_marginal_cost_eur_per_mwh_hs"]
    )
    biogas["electricity_marginal_cost"] = float(
        prices["onsite_electricity_marginal_cost_eur_per_mwh"]
    )
    biogas["heat_marginal_cost"] = float(
        prices["onsite_heat_marginal_cost_eur_per_mwh"]
    )

    # Biogas.SH routes.
    biogas["scenario_mode"] = "custom"
    for key in (
        "add_local_generation",
        "add_gas_grid_generation",
        "add_swfl_direct_supply",
    ):
        biogas[key] = bool(routes[key])

    # SWFL gas-to-power.
    central_gas_to_power = _mutable_mapping(
        swfl,
        "central_gas_chp",
        "args.swfl_real_system",
    )
    gas_to_power_active = bool(units["central_gas_to_power"])
    central_gas_to_power["active"] = gas_to_power_active
    central_gas_to_power["add_electric_link"] = gas_to_power_active

    # SWFL boilers and resistive heaters.
    central_heat = _mutable_mapping(
        swfl,
        "central_heat_units",
        "args.swfl_real_system",
    )
    selected_boilers = set(map(str, units.get("boilers", [])))
    selected_resistive = set(map(str, units.get("resistive_heaters", [])))

    boiler_data = technical_swfl["boilers"]
    resistive_data = technical_swfl["resistive_heaters"]

    _update_named_assets(
        central_heat["boilers"],
        selected_boilers,
        boiler_data,
    )
    _update_named_assets(
        central_heat["resistive_heaters"],
        selected_resistive,
        resistive_data,
    )

    selected_heat_capacity = sum(
        float(boiler_data[name]["heat_capacity_mw"])
        for name in selected_boilers
    )
    selected_heat_capacity += sum(
        float(resistive_data[name]["heat_capacity_mw"])
        for name in selected_resistive
    )

    central_heat["active"] = bool(selected_boilers or selected_resistive)
    central_heat["expected_total_heat_capacity_mw"] = selected_heat_capacity

    reserve = _mutable_mapping(
        swfl,
        "reserve_gas_boiler",
        "args.swfl_real_system",
    )
    reserve["active"] = bool(units.get("reserve_gas_boiler", False))

    # Biomethane-eligible SWFL boilers.
    eligible_units = list(map(str, biomethane_use.get("eligible_units", [])))
    central_heat["biomethane_mode"] = str(biomethane_use["mode"])
    central_heat["planned_biomethane_units"] = eligible_units
    central_heat["custom_biomethane_units"] = eligible_units

    # Planned heat pumps.
    future_heat_pumps = _mutable_mapping(
        swfl,
        "future_heat_pumps",
        "args.swfl_real_system",
    )
    selected_heat_pumps = list(
        map(str, heat_pump_case.get("active_units", []))
    )
    future_heat_pumps["active"] = bool(selected_heat_pumps)
    future_heat_pumps["active_units"] = selected_heat_pumps

    _update_named_assets(
        future_heat_pumps["units"],
        set(selected_heat_pumps),
        technical_swfl["heat_pumps"],
    )

    # Buses and selected SWFL area.
    buses = technical_swfl["buses"]
    swfl["swfl_ac_bus"] = str(buses["ac"])
    swfl["swfl_heat_bus"] = str(buses["heat"])
    swfl["swfl_ch4_bus"] = str(buses["natural_gas"])
    swfl["selected_mv_grid_district_ids"] = list(
        map(str, technical_swfl["selected_mv_grid_district_ids"])
    )

    central_heat["natural_gas_bus"] = str(buses["natural_gas"])
    central_heat["biomethane_bus"] = str(buses["biomethane"])
    central_heat["ac_bus"] = str(buses["ac"])
    central_heat["heat_bus"] = str(buses["heat"])

    central_gas_to_power["gas_bus"] = str(buses["natural_gas"])
    central_gas_to_power["ac_bus"] = str(buses["ac"])
    central_gas_to_power["heat_bus"] = str(buses["heat"])

    # SWFL load assumptions.
    loads = technical_swfl["loads"]
    heat_load = _mutable_mapping(
        swfl,
        "heat_load",
        "args.swfl_real_system",
    )
    ac_load = _mutable_mapping(
        swfl,
        "ac_load",
        "args.swfl_real_system",
    )

    if loads.get("heat_csv_path"):
        heat_load["csv_path"] = str(loads["heat_csv_path"])
    heat_load["year"] = int(loads["heat_year"])
    heat_load["datetime_column"] = str(loads["heat_datetime_column"])
    heat_load["column"] = str(loads["heat_column"])
    ac_load["target_annual_demand_mwh"] = float(
        loads["ac_target_annual_demand_mwh"]
    )

    # SWFL technology assumptions.
    gas_to_power_data = technical_swfl["central_gas_to_power"]
    central_gas_to_power["electric_capacity_mw"] = float(
        gas_to_power_data["electric_capacity_mw"]
    )
    central_gas_to_power["electric_efficiency"] = float(
        gas_to_power_data["electric_efficiency"]
    )
    central_gas_to_power["marginal_cost"] = float(
        gas_to_power_data["marginal_cost_eur_per_mwh"]
    )

    reserve_data = technical_swfl["reserve_gas_boiler"]
    reserve["heat_capacity_mw"] = float(
        reserve_data["heat_capacity_mw"]
    )
    reserve["efficiency"] = float(reserve_data["efficiency"])

    # Public-grid natural-gas supply to SWFL.
    public_supply = technical_biogas["public_grid_to_swfl"]
    swfl_direct["grid_supply_p_nom"] = float(
        public_supply["power_capacity_mw"]
    )
    swfl_direct["grid_supply_efficiency"] = float(
        public_supply["efficiency"]
    )
    swfl_direct["grid_supply_capital_cost"] = float(
        public_supply["capital_cost_eur_per_mw"]
    )

    # Shared regional Biogas.SH resource constraint.
    resource_settings = technical_biogas["resource_constraint"]
    if bool(resource_settings.get("active", True)):
        resource = extra_functionality.setdefault(
            "biogas_sh_resource",
            {},
        )
        if not isinstance(resource, MutableMapping):
            raise ScenarioConfigError(
                "args.extra_functionality.biogas_sh_resource "
                "must be a mapping."
            )

        csv_path = resource.get("csv_path") or biogas.get("csv_path")
        if not csv_path:
            raise ScenarioConfigError(
                "The active Biogas.SH resource constraint requires "
                "csv_path in args.extra_functionality.biogas_sh_resource "
                "or args.biogas_sh.csv_path."
            )

        resource["csv_path"] = csv_path
        efficiencies = technical_biogas["efficiencies"]
        resource["eta_el"] = float(efficiencies["onsite_electricity"])
        resource["eta_heat"] = float(efficiencies["onsite_heat"])
        resource["eta_upgrade"] = float(efficiencies["upgrading"])
        resource["ignore_missing_components"] = bool(
            resource_settings["ignore_missing_components"]
        )
    else:
        extra_functionality.pop("biogas_sh_resource", None)

    # Central Biogas.SH storage.
    storage = technical_biogas["storage"]
    storage_args = _mutable_mapping(
        biogas,
        "gas_storage",
        "args.biogas_sh",
    )
    storage_args["active"] = bool(storage["active"])
    storage_args["e_nom_mwh"] = float(storage["energy_capacity_mwh"])
    storage_args["e_initial"] = float(storage["initial_energy_mwh"])
    storage_args["e_cyclic"] = bool(storage["cyclic"])
    storage_args["standing_loss"] = float(storage["standing_loss"])

    storage_args["input_link_efficiency"] = float(
        storage["plant_to_storage"]["efficiency"]
    )
    storage_args["input_link_marginal_cost"] = float(
        storage["plant_to_storage"]["marginal_cost_eur_per_mwh"]
    )

    storage_args["grid_link_p_nom_mw"] = float(
        storage["storage_to_public_grid"]["power_capacity_mw"]
    )
    storage_args["grid_link_efficiency"] = float(
        storage["storage_to_public_grid"]["efficiency"]
    )
    storage_args["grid_link_marginal_cost"] = float(
        storage["storage_to_public_grid"]["marginal_cost_eur_per_mwh"]
    )

    storage_args["swfl_link_p_nom_mw"] = float(
        storage["storage_to_swfl"]["power_capacity_mw"]
    )
    storage_args["swfl_link_efficiency"] = float(
        storage["storage_to_swfl"]["efficiency"]
    )
    storage_args["swfl_link_marginal_cost"] = float(
        storage["storage_to_swfl"]["marginal_cost_eur_per_mwh"]
    )
    storage_args["swfl_target_bus"] = str(buses["biomethane"])

    _apply_run_settings(args, resolved)
    args["biogas_sh_scenario_name"] = str(resolved["scenario_name"])
    return args


def load_and_apply_config(
    args: MutableMapping[str, Any],
    path: str | Path = "config.yaml",
) -> Tuple[MutableMapping[str, Any], Dict[str, Any]]:
    """Load, resolve and apply one YAML scenario."""
    config = load_config(path)
    resolved = resolve_config(config)
    apply_config_to_args(args, resolved)
    return args, resolved


def scenario_summary(resolved: Mapping[str, Any]) -> str:
    """Return a concise scenario and run summary."""
    selection = resolved["selection"]
    prices = resolved["prices"]
    heat_pumps = resolved["heat_pumps"].get("active_units", [])
    units = resolved["swfl_units"]
    technical_biogas = resolved["technical"]["biogas_sh"]
    effective_run = resolved.get("effective_run", {})

    lines = [
        "",
        "=== BIOGAS.SH / SWFL SCENARIO ===",
        f"name:                       {resolved['scenario_name']}",
        f"natural-gas price case:     {selection['natural_gas_price_case']}",
        (
            "SWFL gas import adder:     "
            f"{prices['natural_gas_swfl_import_adder_eur_per_mwh_hs']:.2f} "
            "EUR/MWh_Hs"
        ),
        f"biomethane price case:      {selection['biomethane_price_case']}",
        (
            "biomethane cost:           "
            f"{prices['biomethane_marginal_cost_eur_per_mwh_hs']:.2f} "
            "EUR/MWh_Hs"
        ),
        f"heat-pump case:             {selection['heat_pump_case']}",
        (
            "active heat pumps:          "
            f"{', '.join(heat_pumps) if heat_pumps else 'none'}"
        ),
        f"SWFL unit case:             {selection['swfl_unit_case']}",
        (
            "active boilers:             "
            f"{', '.join(units.get('boilers', [])) or 'none'}"
        ),
        (
            "active resistive heaters:   "
            f"{', '.join(units.get('resistive_heaters', [])) or 'none'}"
        ),
        (
            "gas-to-power active:        "
            f"{bool(units.get('central_gas_to_power', False))}"
        ),
        f"biomethane use case:        {selection['biomethane_use_case']}",
        f"Biogas.SH route case:       {selection['biogas_route_case']}",
        (
            "resource constraint active: "
            f"{bool(technical_biogas['resource_constraint']['active'])}"
        ),
        (
            "central storage active:     "
            f"{bool(technical_biogas['storage']['active'])}"
        ),
    ]

    if effective_run:
        lines.extend(
            [
                f"represented hours:          {effective_run.get('represented_hours')}",
                f"AC clusters:                {effective_run.get('ac_clusters')}",
                f"result directory:           {effective_run.get('csv_export')}",
            ]
        )

    lines.append("=====================================")
    return "\n".join(lines)


def write_resolved_config(
    resolved: Mapping[str, Any],
    path: str | Path,
) -> Path:
    """Write the effective scenario beside the results."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            dict(resolved),
            handle,
            sort_keys=False,
            allow_unicode=True,
        )
    return output_path


def expand_scenario_matrix(
    config: Mapping[str, Any],
) -> list[Dict[str, str]]:
    """Expand and validate the configured factorial scenario matrix."""
    matrix = config.get("scenario_matrix", {})
    if not isinstance(matrix, Mapping) or not matrix.get("enabled", False):
        return []

    dimensions = matrix.get("dimensions", {})
    if not isinstance(dimensions, Mapping) or not dimensions:
        return []

    keys = list(dimensions)
    values = [list(dimensions[key]) for key in keys]
    base_selection = _selection_with_environment_overrides(config)
    records: list[Dict[str, str]] = []

    for combination in itertools.product(*values):
        selection = dict(base_selection)
        record = dict(zip(keys, map(str, combination)))
        selection.update(record)
        _validate_selection(config, selection)

        record["scenario_name"] = "__".join(
            selection[key]
            for key in (
                "natural_gas_price_case",
                "biomethane_price_case",
                "heat_pump_case",
                "swfl_unit_case",
                "biomethane_use_case",
                "biogas_route_case",
            )
        )
        records.append(record)

    return records


def write_scenario_matrix(
    config: Mapping[str, Any],
    path: str | Path,
) -> Path:
    """Write the expanded factorial matrix as CSV."""
    records = expand_scenario_matrix(config)
    if not records:
        raise ScenarioConfigError(
            "scenario_matrix is disabled or has no dimensions."
        )

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["scenario_name"] + [
        key for key in records[0] if key != "scenario_name"
    ]

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    return output_path


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect or expand the Biogas.SH / SWFL YAML configuration."
    )
    parser.add_argument(
        "config",
        nargs="?",
        default="config.yaml",
        help="Path to config.yaml",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate", help="Validate the YAML configuration.")
    subparsers.add_parser("show", help="Print the resolved selected scenario.")

    matrix_parser = subparsers.add_parser(
        "matrix",
        help="Expand scenario_matrix into a CSV file.",
    )
    matrix_parser.add_argument(
        "--output",
        default="scenario_matrix.csv",
        help="Output CSV path.",
    )

    cli = parser.parse_args()
    config = load_config(cli.config)

    if cli.command == "validate":
        print(f"PASS: {cli.config} is valid.")
        return

    if cli.command == "show":
        resolved = resolve_config(config)
        print(scenario_summary(resolved))
        return

    if cli.command == "matrix":
        output = write_scenario_matrix(config, cli.output)
        print(
            f"Wrote {len(expand_scenario_matrix(config))} scenarios to {output}"
        )
        return


if __name__ == "__main__":
    _main()
