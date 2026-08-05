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


CONFIG_VERSION = 3


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
        "fossil_gas_price_case": (
            "ETRAGO_FOSSIL_GAS_PRICE_CASE"
        ),
        "biomethane_price_case": (
            "ETRAGO_BIOMETHANE_PRICE_CASE"
        ),
        "heat_pump_case": "ETRAGO_HEAT_PUMP_CASE",
        "swfl_unit_case": "ETRAGO_SWFL_UNIT_CASE",
        "biomethane_use_case": (
            "ETRAGO_BIOMETHANE_USE_CASE"
        ),
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
        "fossil_gas_price_case",
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
        "fossil_gas",
        selection["fossil_gas_price_case"],
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
    """Load and validate a version-3 YAML scenario configuration."""
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


def resolve_config(
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    """Resolve one selected scenario into a reproducibility record."""

    selection = _selection_with_environment_overrides(
        config
    )
    _validate_selection(
        config,
        selection,
    )

    price_cases = _require_mapping(
        config,
        "price_cases",
        "configuration",
    )

    fossil_gas = _named_case(
        price_cases,
        "fossil_gas",
        selection["fossil_gas_price_case"],
    )

    biomethane = _named_case(
        price_cases,
        "biomethane",
        selection["biomethane_price_case"],
    )

    onsite = _require_mapping(
        price_cases,
        "onsite",
        "price_cases",
    )

    onsite_electricity_cases = _require_mapping(
        onsite,
        "electricity_marginal_cost_eur_per_mwh",
        "price_cases.onsite",
    )

    onsite_case = str(
        _require(
            onsite,
            "selected_electricity_case",
            "price_cases.onsite",
        )
    )

    if onsite_case not in onsite_electricity_cases:
        raise ScenarioConfigError(
            "Unknown onsite electricity-price case: "
            f"{onsite_case}"
        )

    try:
        for case_name, value in onsite_electricity_cases.items():
            float(value)

        float(
            _require(
                onsite,
                "heat_marginal_cost_eur_per_mwh",
                "price_cases.onsite",
            )
        )
    except (TypeError, ValueError) as exc:
        raise ScenarioConfigError(
            "All onsite marginal costs must be numeric."
        ) from exc

    # --------------------------------------------------------
    # Carbon-inclusive fossil natural-gas cost
    # --------------------------------------------------------
    gas_commodity_price = float(
        fossil_gas[
            "gas_commodity_price_eur_per_mwh_fuel"
        ]
    )

    co2_price = float(
        fossil_gas[
            "co2_price_eur_per_tco2"
        ]
    )

    emission_factor = float(
        fossil_gas[
            "emission_factor_tco2_per_mwh_fuel"
        ]
    )

    co2_cost_on_gas = (
        co2_price
        * emission_factor
    )

    final_ch4_ng_cost = (
        gas_commodity_price
        + co2_cost_on_gas
    )

    swfl_import_adder = float(
        fossil_gas.get(
            "swfl_import_adder_eur_per_mwh_fuel",
            0.0,
        )
    )

    ordered_dimensions = (
        "fossil_gas_price_case",
        "biomethane_price_case",
        "heat_pump_case",
        "swfl_unit_case",
        "biomethane_use_case",
        "biogas_route_case",
    )

    scenario_name = "__".join(
        selection[key]
        for key in ordered_dimensions
    )

    return {
        "config_version": CONFIG_VERSION,
        "scenario_name": scenario_name,
        "selection": selection,
        "prices": {
            "gas_commodity_price_eur_per_mwh_fuel": (
                gas_commodity_price
            ),
            "co2_price_eur_per_tco2": (
                co2_price
            ),
            "emission_factor_tco2_per_mwh_fuel": (
                emission_factor
            ),
            "co2_cost_on_gas_eur_per_mwh_fuel": (
                co2_cost_on_gas
            ),
            "final_ch4_ng_marginal_cost_eur_per_mwh_fuel": (
                final_ch4_ng_cost
            ),
            "swfl_import_adder_eur_per_mwh_fuel": (
                swfl_import_adder
            ),
            "gas_source": str(
                fossil_gas.get(
                    "gas_source",
                    "",
                )
            ),
            "co2_source": str(
                fossil_gas.get(
                    "co2_source",
                    "",
                )
            ),
            "gas_value_status": str(
                fossil_gas.get(
                    "gas_value_status",
                    "",
                )
            ),
            "biomethane_marginal_cost_eur_per_mwh_hs": float(
                biomethane[
                    "marginal_cost_eur_per_mwh_hs"
                ]
            ),
            "onsite_electricity_marginal_cost_eur_per_mwh": float(
                onsite_electricity_cases[
                    onsite_case
                ]
            ),
            "onsite_heat_marginal_cost_eur_per_mwh": float(
                onsite[
                    "heat_marginal_cost_eur_per_mwh"
                ]
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
            _require_mapping(
                config,
                "technical",
                "configuration",
            )
        ),
        "run": copy.deepcopy(
            config.get(
                "run",
                {},
            )
        ),
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


def apply_network_price_scenario(
    network,
    resolved: Mapping[str, Any],
) -> None:
    """
    Apply the selected fossil-gas price to CH4_NG generators and
    the selected biomethane price to the 21 custom Biogas.SH
    generators.

    Run after adjust_CH4_gen_carriers() and after custom
    Biogas.SH assets are added, but before clustering.
    """
    prices = resolved["prices"]

    fossil_gas_cost = float(
        prices[
            "final_ch4_ng_marginal_cost_eur_per_mwh_fuel"
        ]
    )

    biomethane_cost = float(
        prices[
            "biomethane_marginal_cost_eur_per_mwh_hs"
        ]
    )

    # Original fossil natural-gas generators.
    ch4_ng_ids = network.generators.index[
        network.generators["carrier"]
        .astype(str)
        .eq("CH4_NG")
    ]

    if len(ch4_ng_ids) == 0:
        raise ScenarioConfigError(
            "No CH4_NG generators found. "
            "adjust_CH4_gen_carriers() must run first."
        )

    network.generators.loc[
        ch4_ng_ids,
        "marginal_cost",
    ] = fossil_gas_cost

    # Only the custom Biogas.SH biomethane generators.
    generator_names = (
        network.generators.index
        .to_series()
        .astype(str)
    )

    generator_buses = (
        network.generators["bus"]
        .astype(str)
    )

    custom_biomethane_mask = (
        network.generators["carrier"]
        .astype(str)
        .eq("CH4_biogas")
    )

    custom_biomethane_mask &= (
        generator_names.str.startswith(
            "biogas_sh_ch4_bus_"
        )
        | generator_buses.str.startswith(
            "biogas_sh_ch4_bus_"
        )
    )

    custom_biomethane_ids = (
        network.generators.index[
            custom_biomethane_mask
        ]
    )

    routes = resolved["biogas_routes"]

    biomethane_route_active = bool(
        routes.get("add_gas_grid_generation", False)
        or routes.get("add_swfl_direct_supply", False)
    )

    expected_biomethane_generators = (
        21 if biomethane_route_active else 0
    )

    if len(custom_biomethane_ids) != expected_biomethane_generators:
        raise ScenarioConfigError(
            "Expected "
            f"{expected_biomethane_generators} custom Biogas.SH "
            "biomethane generators for the selected route, "
            f"but found {len(custom_biomethane_ids)}."
        )

    network.generators.loc[
        custom_biomethane_ids,
        "marginal_cost",
    ] = biomethane_cost

    print("\n=== APPLIED FUEL PRICES ===")
    print(
        "Fossil-gas case:",
        resolved["selection"]["fossil_gas_price_case"],
    )
    print(
        "Gas commodity:",
        f"{prices['gas_commodity_price_eur_per_mwh_fuel']:.4f}",
        "EUR/MWh_fuel",
    )
    print(
        "CO2 price:",
        f"{prices['co2_price_eur_per_tco2']:.4f}",
        "EUR/tCO2",
    )
    print(
        "CO2 cost on gas:",
        f"{prices['co2_cost_on_gas_eur_per_mwh_fuel']:.4f}",
        "EUR/MWh_fuel",
    )
    print(
        "Final CH4_NG cost:",
        f"{fossil_gas_cost:.4f}",
        "EUR/MWh_fuel",
    )
    print(
        "CH4_NG generators updated:",
        len(ch4_ng_ids),
    )
    print(
        "Custom biomethane generators updated:",
        len(custom_biomethane_ids),
    )
    print("===========================")


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
        prices[
            "swfl_import_adder_eur_per_mwh_fuel"
        ]
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


def scenario_summary(
    resolved: Mapping[str, Any],
) -> str:
    """Return a readable summary of the resolved scenario and run settings."""

    selection = resolved["selection"]
    prices = resolved["prices"]
    heat_pumps = resolved["heat_pumps"]
    units = resolved["swfl_units"]
    biomethane_use = resolved["biomethane_use"]
    biogas_routes = resolved["biogas_routes"]
    technical_biogas = resolved["technical"]["biogas_sh"]
    effective_run = resolved.get("effective_run", {})

    active_heat_pumps = list(
        map(
            str,
            heat_pumps.get("active_units", []),
        )
    )

    active_boilers = list(
        map(
            str,
            units.get("boilers", []),
        )
    )

    active_resistive_heaters = list(
        map(
            str,
            units.get("resistive_heaters", []),
        )
    )

    biomethane_eligible_units = list(
        map(
            str,
            biomethane_use.get("eligible_units", []),
        )
    )

    gas_source = (
        str(prices.get("gas_source", "")).strip()
        or "not specified"
    )

    co2_source = (
        str(prices.get("co2_source", "")).strip()
        or "not specified"
    )

    gas_value_status = (
        str(prices.get("gas_value_status", "")).strip()
        or "not specified"
    )

    def value_or_none(values: list[str]) -> str:
        return ", ".join(values) if values else "none"

    def yes_no(value: Any) -> str:
        return "yes" if bool(value) else "no"

    lines = [
        "",
        "============================================================",
        "BIOGAS.SH / SWFL SCENARIO",
        "============================================================",
        "",
        "SCENARIO IDENTIFICATION",
        "------------------------------------------------------------",
        f"Scenario name:                  {resolved['scenario_name']}",
        f"Configuration version:          {resolved['config_version']}",
        "",
        "FOSSIL NATURAL-GAS PRICE",
        "------------------------------------------------------------",
        (
            "Price scenario:                 "
            f"{selection['fossil_gas_price_case']}"
        ),
        (
            "Gas commodity price:            "
            f"{prices['gas_commodity_price_eur_per_mwh_fuel']:.4f} "
            "EUR/MWh_fuel"
        ),
        (
            "CO2 certificate price:          "
            f"{prices['co2_price_eur_per_tco2']:.4f} "
            "EUR/tCO2"
        ),
        (
            "Natural-gas emission factor:    "
            f"{prices['emission_factor_tco2_per_mwh_fuel']:.4f} "
            "tCO2/MWh_fuel"
        ),
        (
            "CO2 cost on natural gas:        "
            f"{prices['co2_cost_on_gas_eur_per_mwh_fuel']:.4f} "
            "EUR/MWh_fuel"
        ),
        (
            "Final CH4_NG marginal cost:     "
            f"{prices['final_ch4_ng_marginal_cost_eur_per_mwh_fuel']:.4f} "
            "EUR/MWh_fuel"
        ),
        (
            "SWFL gas-import adder:          "
            f"{prices['swfl_import_adder_eur_per_mwh_fuel']:.4f} "
            "EUR/MWh_fuel"
        ),
        f"Gas-price source:              {gas_source}",
        f"CO2-price source:              {co2_source}",
        f"Gas-value status:              {gas_value_status}",
        "",
        "BIOGAS.SH COST ASSUMPTIONS",
        "------------------------------------------------------------",
        (
            "Biomethane price case:          "
            f"{selection['biomethane_price_case']}"
        ),
        (
            "Biomethane marginal cost:       "
            f"{prices['biomethane_marginal_cost_eur_per_mwh_hs']:.4f} "
            "EUR/MWh_Hs"
        ),
        (
            "Onsite electricity cost:        "
            f"{prices['onsite_electricity_marginal_cost_eur_per_mwh']:.4f} "
            "EUR/MWh_el"
        ),
        (
            "Onsite heat cost:               "
            f"{prices['onsite_heat_marginal_cost_eur_per_mwh']:.4f} "
            "EUR/MWh_th"
        ),
        "",
        "SWFL TECHNOLOGY CONFIGURATION",
        "------------------------------------------------------------",
        (
            "Heat-pump case:                 "
            f"{selection['heat_pump_case']}"
        ),
        (
            "Active heat pumps:              "
            f"{value_or_none(active_heat_pumps)}"
        ),
        (
            "SWFL unit case:                 "
            f"{selection['swfl_unit_case']}"
        ),
        (
            "Gas-to-power active:            "
            f"{yes_no(units.get('central_gas_to_power', False))}"
        ),
        (
            "Active gas boilers:             "
            f"{value_or_none(active_boilers)}"
        ),
        (
            "Active resistive heaters:       "
            f"{value_or_none(active_resistive_heaters)}"
        ),
        (
            "Reserve gas boiler active:      "
            f"{yes_no(units.get('reserve_gas_boiler', False))}"
        ),
        "",
        "BIOMETHANE USE AT SWFL",
        "------------------------------------------------------------",
        (
            "Biomethane-use case:            "
            f"{selection['biomethane_use_case']}"
        ),
        (
            "Biomethane-use mode:            "
            f"{biomethane_use.get('mode', 'not specified')}"
        ),
        (
            "Biomethane-eligible units:      "
            f"{value_or_none(biomethane_eligible_units)}"
        ),
        "",
        "BIOGAS.SH ROUTES",
        "------------------------------------------------------------",
        (
            "Route case:                     "
            f"{selection['biogas_route_case']}"
        ),
        (
            "Onsite electricity/heat:        "
            f"{yes_no(biogas_routes.get('add_local_generation', False))}"
        ),
        (
            "Storage to public gas grid:     "
            f"{yes_no(biogas_routes.get('add_gas_grid_generation', False))}"
        ),
        (
            "Storage to SWFL:                "
            f"{yes_no(biogas_routes.get('add_swfl_direct_supply', False))}"
        ),
        (
            "Regional resource constraint:   "
            f"{yes_no(technical_biogas['resource_constraint']['active'])}"
        ),
        (
            "Central biomethane storage:     "
            f"{yes_no(technical_biogas['storage']['active'])}"
        ),
    ]

    if effective_run:
        lines.extend(
            [
                "",
                "RUN SETTINGS",
                "------------------------------------------------------------",
                (
                    "Start snapshot:                 "
                    f"{effective_run.get('start_snapshot')}"
                ),
                (
                    "End snapshot:                   "
                    f"{effective_run.get('end_snapshot')}"
                ),
                (
                    "Represented hours:              "
                    f"{effective_run.get('represented_hours')}"
                ),
                (
                    "AC clusters:                    "
                    f"{effective_run.get('ac_clusters')}"
                ),
                (
                    "Result directory:               "
                    f"{effective_run.get('csv_export')}"
                ),
            ]
        )

    lines.extend(
        [
            "",
            "PRICE ACCOUNTING",
            "------------------------------------------------------------",
            (
                "CH4_NG cost = gas commodity price "
                "+ CO2 price × emission factor"
            ),
            (
                "Fuel and CO2 costs are assigned upstream to "
                "CH4_NG generators."
            ),
            (
                "The SWFL boiler and gas-to-power Links therefore "
                "do not repeat these costs."
            ),
            "============================================================",
        ]
    )

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
                    "fossil_gas_price_case",
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
