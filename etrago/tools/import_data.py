#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 13:26:20 2026

@author: Mohsen Mansouri
"""
from __future__ import annotations

"""
Central path configuration for Biogas-SH and SWFL input data.

This file avoids hardcoded absolute paths in appl.py.

Default folder structure:

    eTraGo/
    └── etrago/
        └── data/
            ├── biogas-sh/
            │   ├── biogas_sh_cluster1.csv
            │   ├── biogas_sh_mapped_to_etrago_buses.csv
            │   ├── ding0_mv_grid_districts.gpkg
            │   └── focus_regions/
            │       └── schleswig_holstein_focus.shp
            └── swfl/
                ├── stadtwerke_flensburg_hourly_heat.csv
                └── stadtwerke_flensburg_assumptions.xlsx

All paths can be overwritten with environment variables, for example:

    export SWFL_HEAT_CSV="/path/to/real_swfl_heat.csv"
    export DING0_MV_GPKG="/path/to/ding0_mv_grid_districts.gpkg"
"""



import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


def env_path(name: str, default: Path) -> Path:
    """
    Return path from environment variable if available,
    otherwise use the given default path.
    """
    return Path(os.environ.get(name, default)).expanduser().resolve()


@dataclass(frozen=True)
class EtragoInputPaths:
    # Root folders
    ETRAGO_PACKAGE_DIR: Path
    ETRAGO_ROOT_DIR: Path
    ETRAGO_DATA_DIR: Path

    # Biogas-SH files
    BIOGAS_SH_DIR: Path
    BIOGAS_SH_CSV: Path
    BIOGAS_SH_MAPPED_CSV: Path
    DING0_MV_GPKG: Path
    BIOGAS_SH_FOCUS_REGION: Path

    # SWFL files
    SWFL_DIR: Path
    SWFL_HEAT_CSV: Path

    # Output/debug
    OUTPUT_DIR: Path
    DEBUG_LOG_PATH: Path


def get_data_paths(validate: bool = False) -> EtragoInputPaths:
    """
    Build all project input/output paths.

    Parameters
    ----------
    validate:
        If True, raise FileNotFoundError if required input files are missing.

    Returns
    -------
    EtragoInputPaths
        Dataclass containing all paths used in appl.py.
    """

    # This file is:
    #   eTraGo/etrago/tools/import_data.py
    #
    # Therefore:
    #   parents[0] = eTraGo/etrago/tools
    #   parents[1] = eTraGo/etrago
    #   parents[2] = eTraGo
    etrago_package_dir = Path(__file__).resolve().parents[1]
    etrago_root_dir = Path(__file__).resolve().parents[2]

    etrago_data_dir = env_path(
        "ETRAGO_DATA_DIR",
        etrago_package_dir / "data",
    )

    # ------------------------------------------------------------------
    # Biogas-SH
    # ------------------------------------------------------------------

    biogas_sh_dir = env_path(
        "BIOGAS_SH_DIR",
        etrago_data_dir / "biogas-sh",
    )

    biogas_sh_csv = env_path(
        "BIOGAS_SH_CSV",
        biogas_sh_dir / "biogas_sh_cluster1.csv",
    )

    biogas_sh_mapped_csv = env_path(
        "BIOGAS_SH_MAPPED_CSV",
        biogas_sh_dir / "biogas_sh_mapped_to_etrago_buses.csv",
    )

    ding0_mv_gpkg = env_path(
        "DING0_MV_GPKG",
        biogas_sh_dir / "ding0_mv_grid_districts.gpkg",
    )

    biogas_sh_focus_region = env_path(
        "BIOGAS_SH_FOCUS_REGION",
        biogas_sh_dir / "focus_regions" / "schleswig_holstein_focus.shp",
    )

    # ------------------------------------------------------------------
    # Stadtwerke Flensburg / SWFL
    # ------------------------------------------------------------------

    swfl_dir = env_path(
        "SWFL_DIR",
        etrago_data_dir / "swfl",
    )

    swfl_heat_csv = env_path(
        "SWFL_HEAT_CSV",
        swfl_dir / "stadtwerke_flensburg_hourly_heat.csv",
    )

    # ------------------------------------------------------------------
    # Output/debug
    # ------------------------------------------------------------------

    output_dir = env_path(
        "ETRAGO_OUTPUT_DIR",
        etrago_root_dir,
    )

    debug_log_path = env_path(
        "BIOGAS_SH_DEBUG_LOG",
        output_dir / "biogas_sh_debug.log",
    )

    paths = EtragoInputPaths(
        ETRAGO_PACKAGE_DIR=etrago_package_dir,
        ETRAGO_ROOT_DIR=etrago_root_dir,
        ETRAGO_DATA_DIR=etrago_data_dir,
        BIOGAS_SH_DIR=biogas_sh_dir,
        BIOGAS_SH_CSV=biogas_sh_csv,
        BIOGAS_SH_MAPPED_CSV=biogas_sh_mapped_csv,
        DING0_MV_GPKG=ding0_mv_gpkg,
        BIOGAS_SH_FOCUS_REGION=biogas_sh_focus_region,
        SWFL_DIR=swfl_dir,
        SWFL_HEAT_CSV=swfl_heat_csv,
        OUTPUT_DIR=output_dir,
        DEBUG_LOG_PATH=debug_log_path,
    )

    if validate:
        validate_required_paths(
            paths,
            required=[
                "BIOGAS_SH_CSV",
                "DING0_MV_GPKG",
                "BIOGAS_SH_FOCUS_REGION",
                "SWFL_HEAT_CSV",
            ],
        )

    return paths


def validate_required_paths(
    paths: EtragoInputPaths,
    required: Optional[Iterable[str]] = None,
) -> None:
    """
    Validate that selected required input paths exist.
    """
    required = list(required or [])

    missing = []

    for name in required:
        path = getattr(paths, name)
        if not Path(path).exists():
            missing.append((name, path))

    if missing:
        msg = "\nMissing required input files:\n"
        for name, path in missing:
            msg += f"  {name}: {path}\n"
        raise FileNotFoundError(msg)


def print_data_paths(paths: EtragoInputPaths) -> None:
    """
    Optional helper for debugging.
    """
    print("\nInput/output paths used by appl.py")
    for name, value in paths.__dict__.items():
        print(f"  {name}: {value}")


def fix_custom_component_scn_names(
    network,
    scn_name: str = "eGon2035",
    prefixes=None,
    fill_missing: bool = True,
) -> None:
    """
    Set scn_name for custom components added outside the original eGon database.

    PyPSA/eTraGo clustering can fail if components inside one cluster have
    inconsistent static metadata such as scn_name.

    This function fixes:
      1. Custom components whose names start with selected prefixes.
      2. Optionally, missing/empty scn_name values in all relevant component tables.

    Parameters
    ----------
    network :
        PyPSA network.
    scn_name :
        Scenario name to assign, for example "eGon2035".
    prefixes :
        Component name prefixes to treat as custom components.
    fill_missing :
        If True, also fill missing or empty scn_name values.
    """
    if prefixes is None:
        prefixes = (
            "swfl_real_",
            "swfl_gwp_",
            "biogas_sh_",
        )

    table_names = [
        "loads",
        "links",
        "generators",
        "stores",
        "storage_units",
    ]

    for table_name in table_names:
        if not hasattr(network, table_name):
            continue

        df = getattr(network, table_name)

        if df.empty:
            continue

        if "scn_name" not in df.columns:
            continue

        idx = df.index.astype(str)

        custom_mask = idx.str.startswith(prefixes)

        if custom_mask.any():
            df.loc[custom_mask, "scn_name"] = scn_name
            print(
                f"Set scn_name={scn_name!r} for "
                f"{custom_mask.sum()} custom {table_name}."
            )

        if fill_missing:
            scn = df["scn_name"]

            missing_mask = (
                scn.isna()
                | scn.astype(str).str.strip().isin(["", "nan", "None", "NaN"])
            )

            if missing_mask.any():
                df.loc[missing_mask, "scn_name"] = scn_name
                print(
                    f"Filled missing scn_name={scn_name!r} for "
                    f"{missing_mask.sum()} {table_name}."
                )

def diagnose_swfl_market_balance(
    network,
    swfl_ch4_bus: str = "biogas_sh_swfl_ch4_bus",
    swfl_heat_bus: str = "swfl_real_central_heat_bus",
    swfl_ac_bus: str = "33935",
) -> None:
    """
    Print SWFL load and supply balance before market optimisation.

    This is mainly for debugging market-model infeasibility.
    """
    n = network

    print("\n" + "=" * 100)
    print("SWFL market-balance diagnostic before optimisation")
    print("=" * 100)

    for bus in [swfl_ac_bus, swfl_heat_bus, swfl_ch4_bus]:
        print(f"\nBus {bus} exists:", bus in n.buses.index)
        if bus in n.buses.index:
            cols = [c for c in ["carrier", "x", "y"] if c in n.buses.columns]
            print(n.buses.loc[[bus], cols].to_string())

    print("\nSWFL loads")
    for load in ["swfl_real_ac_load", "swfl_real_heat_load"]:
        print(f"\n{load}")
        print("  exists:", load in n.loads.index)
        print("  time series:", load in n.loads_t.p_set.columns)

        if load in n.loads.index:
            print(n.loads.loc[[load]].to_string())

        if load in n.loads_t.p_set.columns:
            s = n.loads_t.p_set[load]
            try:
                w = n.snapshot_weightings.generators.reindex(s.index).fillna(1.0)
            except Exception:
                w = 1.0

            print("  snapshots:", len(s))
            print("  min MW:", float(s.min()))
            print("  mean MW:", float(s.mean()))
            print("  max MW:", float(s.max()))

            try:
                print("  weighted energy MWh:", float((s * w).sum()))
            except Exception:
                print("  unweighted energy MWh:", float(s.sum()))

    print("\nLinks connected to SWFL CH4 bus")
    mask_ch4 = (
        n.links.bus0.astype(str).eq(str(swfl_ch4_bus))
        | n.links.bus1.astype(str).eq(str(swfl_ch4_bus))
    )

    cols = [
        "carrier",
        "bus0",
        "bus1",
        "p_nom",
        "p_nom_extendable",
        "efficiency",
        "marginal_cost",
        "capital_cost",
    ]
    cols = [c for c in cols if c in n.links.columns]

    if mask_ch4.any():
        print(n.links.loc[mask_ch4, cols].to_string())
    else:
        print("No links connected to SWFL CH4 bus.")

    print("\nLinks connected to SWFL heat bus")
    mask_heat = (
        n.links.bus0.astype(str).eq(str(swfl_heat_bus))
        | n.links.bus1.astype(str).eq(str(swfl_heat_bus))
    )

    if mask_heat.any():
        print(n.links.loc[mask_heat, cols].to_string())
    else:
        print("No links connected to SWFL heat bus.")

    print("\nImportant SWFL links")
    important_links = [
        "biogas_sh_swfl_grid_supply_47538_to_swfl",
        "swfl_real_central_gas_CHP",
        "swfl_real_central_gas_CHP_heat",
        "swfl_real_reserve_gas_boiler",
    ]

    existing = [x for x in important_links if x in n.links.index]
    missing = [x for x in important_links if x not in n.links.index]

    print("Existing:", existing)
    print("Missing:", missing)

    if existing:
        print(n.links.loc[existing, cols].to_string())

    print("=" * 100)


def ensure_swfl_public_ch4_supply_link(
    network,
    link_name: str = "biogas_sh_swfl_grid_supply_47538_to_swfl",
    public_ch4_bus: str = "47538",
    swfl_ch4_bus: str = "biogas_sh_swfl_ch4_bus",
    p_nom: float = 1000.0,
    efficiency: float = 1.0,
    marginal_cost: float = 50.0,
    capital_cost: float = 0.0,
    scn_name: str = "eGon2035",
) -> None:
    """
    Ensure that SWFL has a sufficiently large public CH4 grid backup supply.

    This is needed because the old eGon SWFL consumer links are no longer
    redirected. SWFL demand now comes from real CHP/heat links, so the
    public-grid backup capacity must be set explicitly.
    """
    public_ch4_bus = str(public_ch4_bus)
    swfl_ch4_bus = str(swfl_ch4_bus)

    if public_ch4_bus not in network.buses.index.astype(str):
        raise KeyError(f"Public CH4 bus {public_ch4_bus!r} not found.")

    if swfl_ch4_bus not in network.buses.index.astype(str):
        raise KeyError(f"SWFL CH4 bus {swfl_ch4_bus!r} not found.")

    if link_name in network.links.index:
        network.links.loc[link_name, "bus0"] = public_ch4_bus
        network.links.loc[link_name, "bus1"] = swfl_ch4_bus
        network.links.loc[link_name, "carrier"] = "biogas_sh_swfl_grid_supply"
        network.links.loc[link_name, "p_nom"] = float(p_nom)
        network.links.loc[link_name, "efficiency"] = float(efficiency)
        network.links.loc[link_name, "marginal_cost"] = float(marginal_cost)
        network.links.loc[link_name, "capital_cost"] = float(capital_cost)

        if "p_nom_extendable" in network.links.columns:
            network.links.loc[link_name, "p_nom_extendable"] = False
        if "p_min_pu" in network.links.columns:
            network.links.loc[link_name, "p_min_pu"] = 0.0
        if "p_max_pu" in network.links.columns:
            network.links.loc[link_name, "p_max_pu"] = 1.0
        if "scn_name" in network.links.columns:
            network.links.loc[link_name, "scn_name"] = scn_name

        print(
            f"Updated SWFL public CH4 supply link {link_name}: "
            f"{public_ch4_bus} -> {swfl_ch4_bus}, p_nom={p_nom} MW."
        )
        return

    network.add(
        "Link",
        link_name,
        bus0=public_ch4_bus,
        bus1=swfl_ch4_bus,
        carrier="biogas_sh_swfl_grid_supply",
        p_nom=float(p_nom),
        p_nom_extendable=False,
        efficiency=float(efficiency),
        marginal_cost=float(marginal_cost),
        capital_cost=float(capital_cost),
        p_min_pu=0.0,
        p_max_pu=1.0,
        scn_name=scn_name,
    )

    print(
        f"Added SWFL public CH4 supply link {link_name}: "
        f"{public_ch4_bus} -> {swfl_ch4_bus}, p_nom={p_nom} MW."
    )

def ensure_swfl_ch4_import_generator(
    network,
    generator_name: str = "swfl_real_ch4_import_generator",
    swfl_ch4_bus: str = "biogas_sh_swfl_ch4_bus",
    p_nom: float = 1000.0,
    marginal_cost: float = 50.0,
    capital_cost: float = 0.0,
    scn_name: str = "eGon2035",
) -> None:
    """
    Add a dispatchable CH4 import/source at the SWFL CH4 bus.

    This is a targeted feasibility test:
    If the model becomes feasible with this generator, the problem was not
    SWFL heat demand itself, but missing/limited upstream CH4 supply.
    """
    swfl_ch4_bus = str(swfl_ch4_bus)

    if swfl_ch4_bus not in network.buses.index.astype(str):
        raise KeyError(f"SWFL CH4 bus {swfl_ch4_bus!r} not found.")

    if generator_name in network.generators.index:
        network.generators.loc[generator_name, "bus"] = swfl_ch4_bus
        network.generators.loc[generator_name, "carrier"] = "CH4_import"
        network.generators.loc[generator_name, "p_nom"] = float(p_nom)
        network.generators.loc[generator_name, "marginal_cost"] = float(marginal_cost)
        network.generators.loc[generator_name, "capital_cost"] = float(capital_cost)

        if "p_nom_extendable" in network.generators.columns:
            network.generators.loc[generator_name, "p_nom_extendable"] = False
        if "p_min_pu" in network.generators.columns:
            network.generators.loc[generator_name, "p_min_pu"] = 0.0
        if "p_max_pu" in network.generators.columns:
            network.generators.loc[generator_name, "p_max_pu"] = 1.0
        if "scn_name" in network.generators.columns:
            network.generators.loc[generator_name, "scn_name"] = scn_name

        print(
            f"Updated SWFL CH4 import generator {generator_name}: "
            f"bus={swfl_ch4_bus}, p_nom={p_nom} MW."
        )
        return

    network.add(
        "Generator",
        generator_name,
        bus=swfl_ch4_bus,
        carrier="CH4_import",
        p_nom=float(p_nom),
        p_nom_extendable=False,
        marginal_cost=float(marginal_cost),
        capital_cost=float(capital_cost),
        p_min_pu=0.0,
        p_max_pu=1.0,
        scn_name=scn_name,
    )

    print(
        f"Added SWFL CH4 import generator {generator_name}: "
        f"bus={swfl_ch4_bus}, p_nom={p_nom} MW."
    )
