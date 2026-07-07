"""
Stadtwerke Flensburg real-system replacement module for eTraGo/PyPSA.

What it does
------------
1. Removes generic eGon AC/heat loads and generators/links in the Flensburg/SWFL area.
2. Adds one real SWFL heat load from an hourly Stadtwerke heat time series.
3. Adds one scaled SWFL AC load using the existing eGon Flensburg AC profile shape.
4. Adds one central gas CHP electricity link and one central gas CHP heat link using SWFL capacities.
5. Removes old eGon central heat pumps and optionally adds future SWFL heat pumps.

Recommended call in appl.py
---------------------------
    from etrago.tools.swfl_real_system import apply_swfl_real_system

    etrago.adjust_network()
    apply_swfl_real_system(etrago.network, args.get("swfl_real_system", {}))
    etrago.ehv_clustering()
    etrago.spatial_clustering()
    etrago.spatial_clustering_gas()

Heat-pump flexibility
---------------------
You can add none, one, or both planned heat pumps:

    # none
    "future_heat_pumps": {"active": False, ...}

    # both
    "future_heat_pumps": {"active": True, "units": [{...}, {...}]}

    # only GWP 1, method A
    "future_heat_pumps": {
        "active": True,
        "units": [
            {"name": "swfl_gwp_1", "active": True, ...},
            {"name": "swfl_gwp_2", "active": False, ...},
        ],
    }

    # only GWP 1, method B
    "future_heat_pumps": {
        "active": True,
        "active_units": ["swfl_gwp_1"],
        "units": [{"name": "swfl_gwp_1", ...}, {"name": "swfl_gwp_2", ...}],
    }
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Main public function
# =============================================================================


def apply_swfl_real_system(network, settings: Optional[Dict[str, Any]] = None):
    """Apply the real SWFL replacement system to a PyPSA network."""
    settings = settings or {}

    if not _as_bool(settings.get("active", False), False):
        logger.info("SWFL real system inactive; network unchanged.")
        return network

    _ensure_timeseries_tables(network)
    snapshots = pd.Index(network.snapshots)

    # 1) Select Flensburg/SWFL area buses before removing anything.
    area_buses = get_swfl_area_buses(network, settings)

    # 2) Build old eGon AC profile shape before deleting old loads.
    ac_cfg = settings.get("ac_load", {}) or {}
    ac_shape = None
    if _as_bool(ac_cfg.get("active", True), True):
        ac_shape = build_existing_ac_profile_shape(network, area_buses, ac_cfg)

    # 3) Remove current eGon assets in the Flensburg/SWFL area.
    if _as_bool(settings.get("remove_existing_flensburg_assets", True), True):
        remove_existing_flensburg_assets(network, area_buses, settings)

    # 4) Remove old eGon central/rural heat pumps if requested.
    hp_cfg = settings.get("future_heat_pumps", {}) or {}
    if _as_bool(hp_cfg.get("remove_existing_central_heat_pumps", True), True):
        remove_existing_heat_pumps(network, area_buses, settings)

    # 5) Ensure new SWFL buses.
    swfl_ac_bus = str(settings.get("swfl_ac_bus", "swfl_ac_bus"))
    swfl_heat_bus = str(settings.get("swfl_heat_bus", "swfl_central_heat_bus"))
    swfl_ch4_bus = str(settings.get("swfl_ch4_bus", "biogas_sh_swfl_ch4_bus"))
    x, y = get_swfl_coordinates(settings)

    ensure_bus(network, swfl_ac_bus, carrier="AC", x=x, y=y)
    ensure_bus(network, swfl_heat_bus, carrier=settings.get("heat_carrier", "central_heat"), x=x, y=y)
    ensure_bus(network, swfl_ch4_bus, carrier="CH4", x=x, y=y)

    # 6) Add real heat load.
    heat_cfg = settings.get("heat_load", {}) or {}
    heat_profile = None
    if _as_bool(heat_cfg.get("active", True), True):
        heat_profile = read_heat_profile_for_snapshots(snapshots, heat_cfg)
        add_or_replace_load(
            network,
            name=str(heat_cfg.get("name", "swfl_real_heat_load")),
            bus=swfl_heat_bus,
            carrier=str(heat_cfg.get("carrier", settings.get("heat_carrier", "central_heat"))),
            p_set=heat_profile,
        )

    # 7) Add scaled AC load.
    if _as_bool(ac_cfg.get("active", True), True):
        ac_profile = scale_ac_profile_to_target(ac_shape, network, ac_cfg, snapshots)
        add_or_replace_load(
            network,
            name=str(ac_cfg.get("name", "swfl_real_ac_load")),
            bus=swfl_ac_bus,
            carrier=str(ac_cfg.get("carrier", "AC")),
            p_set=ac_profile,
        )

    # 8) Add central gas CHP links.
    chp_cfg = settings.get("central_gas_chp", {}) or {}
    if _as_bool(chp_cfg.get("active", True), True):
        add_central_gas_chp_links(
            network,
            chp_cfg,
            gas_bus=str(chp_cfg.get("gas_bus", swfl_ch4_bus)),
            ac_bus=str(chp_cfg.get("ac_bus", swfl_ac_bus)),
            heat_bus=str(chp_cfg.get("heat_bus", swfl_heat_bus)),
        )

    # 9) Optional reserve boiler.
    boiler_cfg = settings.get("reserve_gas_boiler", {}) or {}
    if _as_bool(boiler_cfg.get("active", False), False):
        add_reserve_gas_boiler(
            network,
            boiler_cfg,
            gas_bus=str(boiler_cfg.get("gas_bus", swfl_ch4_bus)),
            heat_bus=str(boiler_cfg.get("heat_bus", swfl_heat_bus)),
        )

    # 10) Optional future heat pumps.
    if _as_bool(hp_cfg.get("active", False), False):
        add_future_heat_pumps(network, hp_cfg, ac_bus=swfl_ac_bus, heat_bus=swfl_heat_bus)

    print_swfl_real_system_summary(network, settings, area_buses, heat_profile, ac_shape)
    return network


# =============================================================================
# Area selection
# =============================================================================


def get_swfl_coordinates(settings: Dict[str, Any]) -> Tuple[float, float]:
    """Default coordinates close to Stadtwerke Flensburg / Flensburg."""
    return (
        float(settings.get("swfl_ch4_bus_x", 9.436502119171873)),
        float(settings.get("swfl_ch4_bus_y", 54.79233181101448)),
    )


def get_swfl_area_buses(network, settings: Dict[str, Any]) -> Set[str]:
    """
    Identify buses in the Flensburg/SWFL replacement area.

    Preferred mode for this project:
        area_mode = "ding0_mv_grid_districts"

    This selects eTraGo buses located inside selected DING0 MV grid districts.
    For the current SWFL case:

        selected_mv_grid_district_ids = ["33935", "33543", "35906"]

    No radius-based selection is used.
    """
    area_mode = str(settings.get("area_mode", "ding0_mv_grid_districts"))

    if area_mode == "explicit_buses":
        explicit = settings.get("area_buses", [])
        existing = set(network.buses.index.astype(str))
        return {str(b) for b in explicit if str(b) in existing}

    if area_mode == "ding0_mv_grid_districts":
        return get_swfl_area_buses_from_ding0_mv_grid_districts(network, settings)

    raise ValueError(
        f"Unsupported swfl_real_system area_mode={area_mode!r}. "
        "Use 'ding0_mv_grid_districts' or 'explicit_buses'."
    )


def get_swfl_area_buses_from_ding0_mv_grid_districts(
    network,
    settings: Dict[str, Any],
) -> Set[str]:
    """
    Select network buses located inside selected DING0 MV grid districts.

    Required settings:
        mv_grid_districts_gpkg
        selected_mv_grid_district_ids

    Optional settings:
        mv_grid_layer
        mv_grid_id_column
        bus_crs
        district_crs
        expand_area_through_local_links
    """
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise ImportError(
            "geopandas is required for "
            "swfl_real_system.area_mode='ding0_mv_grid_districts'."
        ) from exc

    gpkg = settings.get("mv_grid_districts_gpkg")
    if not gpkg:
        raise ValueError(
            "swfl_real_system.area_mode='ding0_mv_grid_districts' requires "
            "'mv_grid_districts_gpkg'."
        )

    selected_ids = [
        str(x) for x in settings.get("selected_mv_grid_district_ids", [])
    ]
    if not selected_ids:
        raise ValueError(
            "swfl_real_system.area_mode='ding0_mv_grid_districts' requires "
            "'selected_mv_grid_district_ids'."
        )

    layer = settings.get("mv_grid_layer", None)
    id_col = str(settings.get("mv_grid_id_column", "name"))

    if layer is None:
        districts = gpd.read_file(gpkg)
    else:
        districts = gpd.read_file(gpkg, layer=layer)

    if id_col not in districts.columns:
        raise KeyError(
            f"Column {id_col!r} not found in DING0 MV grid district file. "
            f"Available columns: {list(districts.columns)}"
        )

    districts = districts.copy()
    districts[id_col] = districts[id_col].astype(str)

    selected_districts = districts[districts[id_col].isin(selected_ids)].copy()

    if selected_districts.empty:
        raise ValueError(
            f"No DING0 MV grid districts found for selected IDs {selected_ids} "
            f"using column {id_col!r}."
        )

    buses = network.buses.copy()

    if "x" not in buses.columns or "y" not in buses.columns:
        raise ValueError(
            "network.buses must contain x/y coordinates for DING0 spatial selection."
        )

    buses["x"] = pd.to_numeric(buses["x"], errors="coerce")
    buses["y"] = pd.to_numeric(buses["y"], errors="coerce")
    buses = buses.dropna(subset=["x", "y"])

    bus_gdf = gpd.GeoDataFrame(
        buses,
        geometry=gpd.points_from_xy(buses["x"], buses["y"]),
        crs=settings.get("bus_crs", "EPSG:4326"),
    )

    if selected_districts.crs is None:
        selected_districts = selected_districts.set_crs(
            settings.get("district_crs", "EPSG:4326")
        )

    if bus_gdf.crs != selected_districts.crs:
        bus_gdf = bus_gdf.to_crs(selected_districts.crs)

    joined = gpd.sjoin(
        bus_gdf,
        selected_districts[[id_col, "geometry"]],
        how="inner",
        predicate="within",
    )

    area_buses = set(joined.index.astype(str))

    # Also include the district IDs themselves if they are actual eTraGo buses.
    # In this case, 33935, 33543, 35906 are relevant AC buses.
    network_bus_ids = set(network.buses.index.astype(str))
    area_buses |= {x for x in selected_ids if x in network_bus_ids}

    # Expand from selected AC buses to directly connected local heat/CHP/boiler buses.
    # This catches the central/rural heat buses attached to these districts.
    if _as_bool(settings.get("expand_area_through_local_links", True), True):
        area_buses = expand_area_buses_through_local_links(
            network=network,
            area_buses=area_buses,
            settings=settings,
        )

    print("\nSWFL area selection using DING0 MV grid districts")
    print(f"  selected MV districts: {selected_ids}")
    print(f"  selected area buses:   {len(area_buses)}")

    return area_buses


def expand_area_buses_through_local_links(
    network,
    area_buses: Set[str],
    settings: Dict[str, Any],
) -> Set[str]:
    """
    Expand selected area buses through local heat/CHP/boiler links.

    This is needed because the selected DING0 MV district IDs are usually AC
    buses, while relevant heat buses may be connected through local conversion
    links such as central_gas_CHP_heat, central_gas_boiler, heat_pump, etc.
    """
    patterns = settings.get(
        "area_expansion_link_carrier_patterns",
        [
            "central_gas",
            "central_heat",
            "rural_heat",
            "heat_pump",
            "CHP",
            "boiler",
        ],
    )

    links = network.links.copy()
    if links.empty:
        return area_buses

    bus_cols = link_bus_columns(links)

    carrier_mask = pd.Series(False, index=links.index)
    if "carrier" in links.columns:
        carrier = links["carrier"].astype(str)
        for pat in patterns:
            carrier_mask |= carrier.str.contains(str(pat), case=False, na=False)
    else:
        carrier_mask[:] = True

    connected_mask = pd.Series(False, index=links.index)
    for col in bus_cols:
        if col in links.columns:
            connected_mask |= links[col].astype(str).isin(area_buses)

    local_links = links[carrier_mask & connected_mask]

    expanded = set(area_buses)
    for col in bus_cols:
        if col in local_links.columns:
            expanded |= set(local_links[col].astype(str))

    return expanded


# =============================================================================
# Removing old eGon assets
# =============================================================================


def remove_existing_flensburg_assets(network, area_buses: Set[str], settings: Dict[str, Any]) -> None:
    """Remove existing eGon loads, generators, and selected links in SWFL area."""
    protected_prefixes = tuple(settings.get("protected_prefixes", ["biogas_sh_", "swfl_real_"]))
    keep_components = {str(x) for x in settings.get("keep_components", [])}

    # Loads connected to area buses.
    load_ids = component_indices_connected_to_buses(
        network.loads,
        bus_columns=["bus"],
        area_buses=area_buses,
        protected_prefixes=protected_prefixes,
        keep_components=keep_components,
    )
    remove_components(network, "Load", load_ids)

    # Generators connected to area buses.
    gen_ids = component_indices_connected_to_buses(
        network.generators,
        bus_columns=["bus"],
        area_buses=area_buses,
        protected_prefixes=protected_prefixes,
        keep_components=keep_components,
    )
    remove_components(network, "Generator", gen_ids)

    # Conversion links connected to area buses. Carrier filter prevents deleting
    # unrelated transmission links just because one endpoint is in Flensburg.
    patterns = settings.get(
        "remove_link_carrier_patterns",
        ["central_gas", "central_heat", "rural_heat", "heat_pump", "CHP", "boiler"],
    )
    link_ids = component_indices_connected_to_buses(
        network.links,
        bus_columns=link_bus_columns(network.links),
        area_buses=area_buses,
        protected_prefixes=protected_prefixes,
        keep_components=keep_components,
        carrier_patterns=patterns,
    )
    remove_components(network, "Link", link_ids)


def remove_existing_heat_pumps(network, area_buses: Set[str], settings: Dict[str, Any]) -> None:
    """Remove old eGon heat-pump links in the SWFL area."""
    protected_prefixes = tuple(settings.get("protected_prefixes", ["biogas_sh_", "swfl_real_"]))
    keep_components = {str(x) for x in settings.get("keep_components", [])}
    patterns = settings.get(
        "remove_heat_pump_carrier_patterns",
        ["central_heat_pump", "rural_heat_pump", "heat_pump"],
    )
    ids = component_indices_connected_to_buses(
        network.links,
        bus_columns=link_bus_columns(network.links),
        area_buses=area_buses,
        protected_prefixes=protected_prefixes,
        keep_components=keep_components,
        carrier_patterns=patterns,
    )
    remove_components(network, "Link", ids)


def component_indices_connected_to_buses(
    df: pd.DataFrame,
    bus_columns: Sequence[str],
    area_buses: Set[str],
    protected_prefixes: Tuple[str, ...],
    keep_components: Set[str],
    carrier_patterns: Optional[Sequence[str]] = None,
) -> List[str]:
    if df is None or len(df) == 0:
        return []

    idx = df.index.astype(str)
    protected = pd.Series(False, index=df.index)
    for prefix in protected_prefixes:
        protected |= idx.str.startswith(prefix)
    if keep_components:
        protected |= idx.isin(keep_components)

    connected = pd.Series(False, index=df.index)
    for col in bus_columns:
        if col in df.columns:
            connected |= df[col].astype(str).isin(area_buses)

    if carrier_patterns is not None and "carrier" in df.columns:
        carrier = df["carrier"].astype(str)
        carrier_mask = pd.Series(False, index=df.index)
        for pat in carrier_patterns:
            carrier_mask |= carrier.str.contains(str(pat), case=False, na=False)
        connected &= carrier_mask

    return df.index[connected & ~protected].astype(str).tolist()


def link_bus_columns(links: pd.DataFrame) -> List[str]:
    return [c for c in links.columns if c.startswith("bus")]


def remove_components(network, component: str, names: Sequence[str]) -> None:
    """
    Remove PyPSA components safely.

    PyPSA network.mremove() raises a KeyError if one of the requested names
    does not exist. For this SWFL replacement module, we often call
    remove_components() before adding/replacing components, so missing names
    should simply be ignored.
    """
    names = [str(n) for n in names if str(n)]
    if not names:
        return

    component_table = {
        "Bus": "buses",
        "Load": "loads",
        "Generator": "generators",
        "Link": "links",
        "Store": "stores",
        "StorageUnit": "storage_units",
        "Line": "lines",
        "Transformer": "transformers",
    }

    table_name = component_table.get(component)
    if table_name is None or not hasattr(network, table_name):
        logger.warning("Unknown PyPSA component type: %s", component)
        return

    table = getattr(network, table_name)
    existing_index = set(table.index.astype(str))

    existing_names = [n for n in names if n in existing_index]
    missing_names = [n for n in names if n not in existing_index]

    if missing_names:
        logger.debug(
            "Skipping %d missing %s components: %s",
            len(missing_names),
            component,
            missing_names[:10],
        )

    if not existing_names:
        return

    logger.info("Removing %d %s components", len(existing_names), component)

    if hasattr(network, "mremove"):
        network.mremove(component, existing_names)
    else:
        for name in existing_names:
            try:
                network.remove(component, name)
            except Exception as exc:
                logger.warning("Could not remove %s %s: %s", component, name, exc)


# =============================================================================
# Load profiles
# =============================================================================


def build_existing_ac_profile_shape(network, area_buses: Set[str], cfg: Dict[str, Any]) -> pd.Series:
    """Use current eGon Flensburg AC loads as temporal profile shape."""
    source_ids = cfg.get("source_load_ids")
    if source_ids:
        load_ids = [str(i) for i in source_ids if str(i) in network.loads.index.astype(str)]
    else:
        carrier = str(cfg.get("source_carrier", cfg.get("carrier", "AC")))
        mask = network.loads["bus"].astype(str).isin(area_buses)
        if "carrier" in network.loads.columns:
            mask &= network.loads["carrier"].astype(str).str.contains(carrier, case=False, na=False)
        load_ids = network.loads.index[mask].astype(str).tolist()

    if not load_ids:
        raise ValueError(
            "No existing Flensburg AC loads found. Adjust area selection or set "
            "swfl_real_system.ac_load.source_load_ids."
        )

    p_set = get_load_p_set(network, load_ids)
    profile = p_set.sum(axis=1).astype(float).reindex(network.snapshots).fillna(0.0)
    if profile.sum() <= 0:
        raise ValueError("Selected eGon AC profile has zero sum.")
    return profile


def get_load_p_set(network, load_ids: Sequence[str]) -> pd.DataFrame:
    snapshots = pd.Index(network.snapshots)
    out = pd.DataFrame(index=snapshots)

    # Time-varying load profiles.
    if hasattr(network, "loads_t") and hasattr(network.loads_t, "p_set"):
        pset = network.loads_t.p_set
        if isinstance(pset, pd.DataFrame):
            for lid in load_ids:
                if lid in pset.columns:
                    out[lid] = pd.to_numeric(pset[lid], errors="coerce").reindex(snapshots)

    # Static p_set fallback.
    for lid in load_ids:
        if lid not in out.columns and lid in network.loads.index:
            val = network.loads.at[lid, "p_set"] if "p_set" in network.loads.columns else 0.0
            val = pd.to_numeric(pd.Series([val]), errors="coerce").fillna(0.0).iloc[0]
            out[lid] = float(val)

    return out.fillna(0.0)


def scale_ac_profile_to_target(profile, network, cfg, snapshots=None):
    """
    Scale an AC load profile to the target electricity demand.

    The function is called as:

        scale_ac_profile_to_target(ac_shape, network, ac_cfg, snapshots)

    For full-year runs:
        target_annual_demand_mwh is used directly.

    For short test runs:
        scale_annual_target_to_snapshot_hours=True prevents the full annual
        demand from being compressed into only a few snapshots.
    """
    profile = profile.copy()

    if profile.empty:
        raise ValueError("Cannot scale empty SWFL AC profile.")

    profile = profile.astype(float)

    if _as_bool(cfg.get("clip_negative", True), True):
        profile = profile.clip(lower=0.0)

    target = cfg.get("target_annual_demand_mwh", None)

    if target is None:
        raise ValueError(
            "ac_load.target_annual_demand_mwh is required for SWFL AC scaling."
        )

    target = float(target)

    if snapshots is None:
        snapshots = profile.index

    # Align profile to the selected snapshots
    profile = profile.reindex(snapshots)

    if profile.isna().any():
        profile = profile.interpolate(method="time").ffill().bfill()

    weights = snapshot_weights(network, profile.index)
    weights = weights.reindex(profile.index).fillna(1.0)

    represented_hours = float(weights.sum())

    if _as_bool(cfg.get("target_is_annual", True), True):
        if _as_bool(cfg.get("scale_annual_target_to_snapshot_hours", True), True):
            if represented_hours > 0 and represented_hours < 8760.0:
                original_target = target
                target = target * represented_hours / 8760.0

                print(
                    "\nSWFL AC annual-demand scaling for short run"
                    f"\n  original annual target: {original_target:.3f} MWh/a"
                    f"\n  represented hours:      {represented_hours:.3f} h"
                    f"\n  scaled target:          {target:.3f} MWh"
                )

    current = float((profile * weights).sum())

    if current <= 0:
        raise ValueError(
            "Existing eGon AC profile shape has zero weighted energy. "
            "Cannot scale SWFL AC load."
        )

    factor = target / current
    scaled = profile * factor

    print(
        "\nSWFL AC load scaling"
        f"\n  current weighted energy: {current:.3f} MWh"
        f"\n  target weighted energy:  {target:.3f} MWh"
        f"\n  scaling factor:          {factor:.6f}"
        f"\n  mean load:               {scaled.mean():.3f} MW"
        f"\n  peak load:               {scaled.max():.3f} MW"
    )

    return scaled


def read_heat_profile_for_snapshots(snapshots: pd.Index, cfg: Dict[str, Any]) -> pd.Series:
    """Read Stadtwerke hourly heat profile for one year and map to network snapshots."""
    year = int(cfg.get("year"))
    col_name = str(cfg.get("column", "HKW Wärmeleistung Gesamt"))
    unit = str(cfg.get("unit", "MW")).lower()

    if cfg.get("csv_path"):
        df = read_table_flexible(Path(cfg["csv_path"]))
    elif cfg.get("xlsx_path"):
        df = pd.read_excel(cfg["xlsx_path"], sheet_name=cfg.get("sheet_name", 0))
    else:
        raise ValueError("heat_load requires csv_path or xlsx_path.")

    df = normalise_columns(df)
    heat_col = find_column(df, col_name)
    dt = extract_datetime_index(df, cfg)

    values = pd.to_numeric(df[heat_col].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    s = pd.Series(values.values, index=dt, name="swfl_heat_mw").sort_index()
    s = s[~s.index.duplicated(keep="first")]
    s = s[s.index.year == year]

    if s.empty:
        raise ValueError(f"No heat data found for year {year}.")

    if unit in {"kw", "kwh/h"}:
        s = s / 1000.0
    elif unit in {"mw", "mwh/h"}:
        pass
    else:
        raise ValueError(f"Unsupported heat unit {unit!r}; use MW or kW.")

    s8760 = make_8760_hourly_year(s, year, fill_method=cfg.get("fill_method", "time_interpolate"))
    mapped = map_year_profile_to_network_snapshots(s8760, snapshots)
    if _as_bool(cfg.get("clip_negative", True), True):
        mapped = mapped.clip(lower=0.0)
    return mapped.astype(float)


# =============================================================================
# Adding components
# =============================================================================


def ensure_bus(network, name: str, carrier: str, x: float, y: float) -> None:
    name = str(name)
    if name in network.buses.index.astype(str):
        if "carrier" in network.buses.columns:
            network.buses.loc[name, "carrier"] = carrier
        if "x" in network.buses.columns:
            network.buses.loc[name, "x"] = x
        if "y" in network.buses.columns:
            network.buses.loc[name, "y"] = y
        return
    network.add("Bus", name, carrier=carrier, x=x, y=y)


def add_or_replace_load(network, name: str, bus: str, carrier: str, p_set: pd.Series) -> None:
    remove_components(network, "Load", [name])
    network.add("Load", name, bus=bus, carrier=carrier)
    _ensure_timeseries_tables(network)
    network.loads_t.p_set[name] = p_set.reindex(network.snapshots).astype(float).fillna(0.0)


def add_central_gas_chp_links(network, cfg: Dict[str, Any], gas_bus: str, ac_bus: str, heat_bus: str) -> None:
    """
    Add simple eGon-style central CHP links.

    Note: this is not yet a physically coupled CHP. Electricity and heat are
    separate links. A heat-to-power coupling constraint can be added later.
    """
    p_nom_is_output = _as_bool(cfg.get("p_nom_is_output_capacity", True), True)

    el_name = str(cfg.get("electric_link_name", "swfl_real_central_gas_CHP"))
    heat_name = str(cfg.get("heat_link_name", "swfl_real_central_gas_CHP_heat"))

    el_cap = float(cfg.get("electric_capacity_mw", 241.0))
    heat_cap = float(cfg.get("heat_capacity_mw", 370.0))
    el_eff = float(cfg.get("electric_efficiency", 1.0))
    heat_eff = float(cfg.get("heat_efficiency", 1.0))

    el_p_nom = el_cap / el_eff if p_nom_is_output and el_eff else el_cap
    heat_p_nom = heat_cap / heat_eff if p_nom_is_output and heat_eff else heat_cap

    remove_components(network, "Link", [el_name, heat_name])

    network.add(
        "Link",
        el_name,
        bus0=gas_bus,
        bus1=ac_bus,
        carrier=str(cfg.get("carrier_el", "central_gas_CHP")),
        p_nom=el_p_nom,
        p_nom_extendable=_as_bool(cfg.get("extendable", False), False),
        p_min_pu=float(cfg.get("p_min_pu", 0.0)),
        p_max_pu=float(cfg.get("p_max_pu", 1.0)),
        efficiency=el_eff,
        marginal_cost=float(cfg.get("electric_marginal_cost", cfg.get("marginal_cost", 0.0))),
        capital_cost=float(cfg.get("electric_capital_cost", cfg.get("capital_cost", 0.0))),
    )

    network.add(
        "Link",
        heat_name,
        bus0=gas_bus,
        bus1=heat_bus,
        carrier=str(cfg.get("carrier_heat", "central_gas_CHP_heat")),
        p_nom=heat_p_nom,
        p_nom_extendable=_as_bool(cfg.get("extendable", False), False),
        p_min_pu=float(cfg.get("p_min_pu", 0.0)),
        p_max_pu=float(cfg.get("p_max_pu", 1.0)),
        efficiency=heat_eff,
        marginal_cost=float(cfg.get("heat_marginal_cost", cfg.get("marginal_cost", 0.0))),
        capital_cost=float(cfg.get("heat_capital_cost", cfg.get("capital_cost", 0.0))),
    )


def add_reserve_gas_boiler(network, cfg: Dict[str, Any], gas_bus: str, heat_bus: str) -> None:
    name = str(cfg.get("name", "swfl_real_reserve_gas_boiler"))
    heat_cap = float(cfg.get("heat_capacity_mw", 203.0))
    eff = float(cfg.get("efficiency", 1.0))
    p_nom_is_output = _as_bool(cfg.get("p_nom_is_output_capacity", True), True)
    p_nom = heat_cap / eff if p_nom_is_output and eff else heat_cap

    remove_components(network, "Link", [name])
    network.add(
        "Link",
        name,
        bus0=gas_bus,
        bus1=heat_bus,
        carrier=str(cfg.get("carrier", "central_gas_boiler")),
        p_nom=p_nom,
        p_nom_extendable=_as_bool(cfg.get("extendable", False), False),
        p_min_pu=float(cfg.get("p_min_pu", 0.0)),
        p_max_pu=float(cfg.get("p_max_pu", 1.0)),
        efficiency=eff,
        marginal_cost=float(cfg.get("marginal_cost", 0.0)),
        capital_cost=float(cfg.get("capital_cost", 0.0)),
    )


def add_future_heat_pumps(network, cfg: Dict[str, Any], ac_bus: str, heat_bus: str) -> None:
    """Add future heat pumps; each unit can be activated individually."""
    units = cfg.get("units", []) or []
    active_units = cfg.get("active_units", None)
    active_units_set = {str(x) for x in active_units} if active_units is not None else None

    for unit in units:
        name = str(unit.get("name", "")).strip()
        if not name:
            raise ValueError("Each future heat pump unit needs a name.")

        if active_units_set is not None:
            unit_active = name in active_units_set
        else:
            unit_active = _as_bool(unit.get("active", True), True)

        if not unit_active:
            remove_components(network, "Link", [name])
            continue

        heat_capacity = float(unit.get("heat_capacity_mw", 60.0))
        cop = float(unit.get("cop", cfg.get("default_cop", 3.0)))
        if cop <= 0:
            raise ValueError(f"Heat pump {name} has invalid COP {cop}.")

        # PyPSA Link p_nom is input-side electric capacity; heat output = p0 * COP.
        electric_input_capacity = heat_capacity / cop

        remove_components(network, "Link", [name])
        network.add(
            "Link",
            name,
            bus0=str(unit.get("ac_bus", ac_bus)),
            bus1=str(unit.get("heat_bus", heat_bus)),
            carrier=str(unit.get("carrier", cfg.get("carrier", "central_heat_pump"))),
            p_nom=electric_input_capacity,
            p_nom_extendable=_as_bool(unit.get("extendable", cfg.get("extendable", False)), False),
            p_min_pu=float(unit.get("p_min_pu", cfg.get("p_min_pu", 0.0))),
            p_max_pu=float(unit.get("p_max_pu", cfg.get("p_max_pu", 1.0))),
            efficiency=cop,
            marginal_cost=float(unit.get("marginal_cost", cfg.get("marginal_cost", 0.0))),
            capital_cost=float(unit.get("capital_cost", cfg.get("capital_cost", 0.0))),
        )

        # Optional metadata for reporting.
        for col, val in {
            "heat_capacity_mw": heat_capacity,
            "cop": cop,
            "planned_year": unit.get("planned_year", np.nan),
        }.items():
            try:
                network.links.loc[name, col] = val
            except Exception:
                pass


# =============================================================================
# Time-series IO helpers
# =============================================================================


def read_table_flexible(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)

    attempts = [
        {"sep": None, "engine": "python"},
        {"sep": ";", "decimal": ","},
        {"sep": ";", "decimal": "."},
        {"sep": ",", "decimal": "."},
        {"sep": ",", "decimal": ","},
        {"sep": "\t", "decimal": ","},
        {"sep": "\t", "decimal": "."},
    ]
    last_error = None
    for kwargs in attempts:
        try:
            df = pd.read_csv(path, **kwargs)
            if df.shape[1] >= 2:
                return df
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Could not read {path}: {last_error}")


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().replace("\ufeff", "") for c in out.columns]
    return out


def normalise_col_name(s: str) -> str:
    return "".join(str(s).strip().lower().replace("_", " ").split())


def find_column(df: pd.DataFrame, requested: str) -> str:
    req = normalise_col_name(requested)
    exact = [c for c in df.columns if normalise_col_name(c) == req]
    if exact:
        return exact[0]
    fuzzy = [c for c in df.columns if req in normalise_col_name(c) or normalise_col_name(c) in req]
    if fuzzy:
        return fuzzy[0]
    raise KeyError(f"Could not find column {requested!r}. Available: {list(df.columns)}")


def extract_datetime_index(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DatetimeIndex:
    if cfg.get("datetime_column"):
        candidates = [cfg["datetime_column"]]
    else:
        candidates = ["timestamp", "datetime", "time", "date", "datum", "zeit", "DateTime", "Datum", "Zeit"]

    for c in candidates:
        if c in df.columns:
            dt = pd.to_datetime(df[c], errors="coerce", dayfirst=True)
            if dt.notna().sum() > 0:
                return pd.DatetimeIndex(dt)

    date_cols = [c for c in df.columns if normalise_col_name(c) in {"date", "datum"}]
    time_cols = [c for c in df.columns if normalise_col_name(c) in {"time", "zeit", "hour", "stunde"}]
    if date_cols and time_cols:
        dt = pd.to_datetime(df[date_cols[0]].astype(str) + " " + df[time_cols[0]].astype(str), errors="coerce", dayfirst=True)
        if dt.notna().sum() > 0:
            return pd.DatetimeIndex(dt)

    first = df.columns[0]
    dt = pd.to_datetime(df[first], errors="coerce", dayfirst=True)
    if dt.notna().sum() > 0:
        return pd.DatetimeIndex(dt)

    raise ValueError("Could not detect datetime column. Set datetime_column in heat_load config.")


def make_8760_hourly_year(s: pd.Series, year: int, fill_method: str = "time_interpolate") -> pd.Series:
    s = s.copy().sort_index()
    s.index = pd.DatetimeIndex(s.index)
    if s.index.tz is not None:
        s.index = s.index.tz_localize(None)

    # Remove leap day if present.
    s = s[~((s.index.month == 2) & (s.index.day == 29))]

    idx = pd.date_range(f"{year}-01-01 00:00:00", f"{year}-12-31 23:00:00", freq="h")
    idx = idx[~((idx.month == 2) & (idx.day == 29))]
    s = s.reindex(idx)

    if fill_method == "zero":
        s = s.fillna(0.0)
    elif fill_method == "ffill":
        s = s.ffill().bfill()
    elif fill_method == "time_interpolate":
        s = s.interpolate(method="time").ffill().bfill()
    else:
        raise ValueError(f"Unsupported fill_method {fill_method!r}")

    if len(s) != 8760:
        raise ValueError(f"Expected 8760 values, got {len(s)}")
    return s.astype(float)


def map_year_profile_to_network_snapshots(profile_8760: pd.Series, snapshots: pd.Index) -> pd.Series:
    profile_8760 = profile_8760.copy()
    profile_8760.index = pd.DatetimeIndex(profile_8760.index)

    lookup = {
        (ts.month, ts.day, ts.hour): float(val)
        for ts, val in profile_8760.items()
        if not (ts.month == 2 and ts.day == 29)
    }

    values = []
    for sn in pd.DatetimeIndex(snapshots):
        key = (sn.month, sn.day, sn.hour)
        values.append(lookup.get(key, np.nan))

    return pd.Series(values, index=snapshots, dtype=float).interpolate().ffill().bfill()


def snapshot_weights(network, snapshots: pd.Index) -> pd.Series:
    try:
        sw = network.snapshot_weightings
        if isinstance(sw, pd.DataFrame):
            if "generators" in sw.columns:
                w = sw["generators"].reindex(snapshots)
            elif "objective" in sw.columns:
                w = sw["objective"].reindex(snapshots)
            else:
                w = sw.iloc[:, 0].reindex(snapshots)
        else:
            w = pd.Series(sw, index=snapshots)
        return pd.to_numeric(w, errors="coerce").fillna(1.0).astype(float)
    except Exception:
        return pd.Series(1.0, index=snapshots, dtype=float)


def _ensure_timeseries_tables(network) -> None:
    if not hasattr(network, "loads_t"):
        return
    if not hasattr(network.loads_t, "p_set") or network.loads_t.p_set is None:
        network.loads_t.p_set = pd.DataFrame(index=network.snapshots)
    if not isinstance(network.loads_t.p_set, pd.DataFrame):
        network.loads_t.p_set = pd.DataFrame(network.loads_t.p_set, index=network.snapshots)


# =============================================================================
# Summary and helpers
# =============================================================================


def print_swfl_real_system_summary(
    network,
    settings: Dict[str, Any],
    area_buses: Set[str],
    heat_profile: Optional[pd.Series],
    ac_shape: Optional[pd.Series],
) -> None:
    hp_cfg = settings.get("future_heat_pumps", {}) or {}
    active_hps = []
    if _as_bool(hp_cfg.get("active", False), False):
        active_units = hp_cfg.get("active_units")
        active_set = {str(x) for x in active_units} if active_units is not None else None
        for u in hp_cfg.get("units", []) or []:
            name = str(u.get("name"))
            if active_set is not None:
                if name in active_set:
                    active_hps.append(name)
            elif _as_bool(u.get("active", True), True):
                active_hps.append(name)

    print("\nSWFL real system added")
    print(f"  area buses detected:       {len(area_buses)}")
    print(f"  swfl_ac_bus:              {settings.get('swfl_ac_bus', 'swfl_ac_bus')}")
    print(f"  swfl_heat_bus:            {settings.get('swfl_heat_bus', 'swfl_central_heat_bus')}")
    print(f"  swfl_ch4_bus:             {settings.get('swfl_ch4_bus', 'biogas_sh_swfl_ch4_bus')}")
    if heat_profile is not None:
        print(f"  heat load max MW:         {float(heat_profile.max()):.3f}")
        print(f"  heat load mean MW:        {float(heat_profile.mean()):.3f}")
    if ac_shape is not None:
        print(f"  AC source profile points: {len(ac_shape)}")
    print(f"  future heat pumps active: {active_hps if active_hps else 'none'}")


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    s = str(value).strip().lower()
    if s in {"true", "1", "yes", "y", "on"}:
        return True
    if s in {"false", "0", "no", "n", "off"}:
        return False
    return default


# =============================================================================
# Example config block for appl.py
# =============================================================================


SWFL_REAL_SYSTEM_EXAMPLE = {
    "active": True,

    # ------------------------------------------------------------------
    # Area definition: DING0 MV grid districts, not radius
    # ------------------------------------------------------------------
    # Flensburg/SWFL assets are selected from these DING0 MV grid districts:
    #   33935, 33543, 35906
    #
    # No radius-based area selection is used.
    "area_mode": "ding0_mv_grid_districts",

    # In appl.py, use:
    # "mv_grid_districts_gpkg": str(DING0_MV_GPKG),
    #
    # If this example stays inside swfl_real_system.py, keep it as a string path.
    "mv_grid_districts_gpkg": "/path/to/ding0_mv_grid_districts.gpkg",
    "mv_grid_layer": None,
    "mv_grid_id_column": "name",

    "selected_mv_grid_district_ids": [
        "33935",
        "33543",
        "35906",
    ],

    # Also include directly connected local heat/CHP/boiler buses.
    # This is important because the MV district IDs are mainly AC buses,
    # while the heat buses are often connected via conversion links.
    "expand_area_through_local_links": True,

    # Remove old generic eGon assets inside the selected MV districts.
    "remove_existing_flensburg_assets": True,

    # Do not delete Biogas-SH or newly created SWFL components.
    "protected_prefixes": [
        "biogas_sh_",
        "swfl_real_",
        "swfl_gwp_",
    ],

    # Old eGon conversion links to remove in the selected MV districts.
    "remove_link_carrier_patterns": [
        "central_gas",
        "central_heat",
        "rural_heat",
        "heat_pump",
        "CHP",
        "boiler",
    ],

    # Old eGon heat pumps to remove in the selected MV districts.
    "remove_heat_pump_carrier_patterns": [
        "central_heat_pump",
        "rural_heat_pump",
        "heat_pump",
    ],

    # ------------------------------------------------------------------
    # Main replacement buses
    # ------------------------------------------------------------------
    "swfl_ac_bus": "33935",
    "swfl_heat_bus": "swfl_real_central_heat_bus",
    "swfl_ch4_bus": "biogas_sh_swfl_ch4_bus",

    "swfl_ch4_bus_x": 9.436502119171873,
    "swfl_ch4_bus_y": 54.79233181101448,

    # ------------------------------------------------------------------
    # Real SWFL heat load
    # ------------------------------------------------------------------
    "heat_load": {
        "active": True,

        # In appl.py, use:
        # "csv_path": str(SWFL_HEAT_CSV),
        #
        # If this example stays inside swfl_real_system.py, keep it as a string path.
        "csv_path": "/path/to/stadtwerke_hourly_heat.csv",

        "year": 2023,
        "column": "HKW Wärmeleistung Gesamt",
        "unit": "MW",
        "carrier": "central_heat",
        "name": "swfl_real_heat_load",

        # Optional if automatic detection fails:
        # "datetime_column": "Zeitstempel",

        "fill_method": "time_interpolate",
        "clip_negative": True,
    },

    # ------------------------------------------------------------------
    # Real-scaled Flensburg/SWFL AC load
    # ------------------------------------------------------------------
    # The temporal profile shape comes from existing eGon AC loads in the
    # selected MV districts. The annual sum is scaled to real electricity demand.
    "ac_load": {
        "active": True,
        "use_existing_egon_profile_shape": True,

        # Real annual electricity demand:
        # 381.516 GWh/a = 381,516 MWh/a
        "target_annual_demand_mwh": 381516.0,

        "carrier": "AC",
        "source_carrier": "AC",
        "name": "swfl_real_ac_load",
        "clip_negative": True,

        # Optional if automatic MV-district selection misses the correct loads:
        # "source_load_ids": ["load_id_1", "load_id_2"],
    },

    # ------------------------------------------------------------------
    # Real SWFL central gas CHP
    # ------------------------------------------------------------------
    # Represented as two simple links:
    #   CH4 bus -> AC bus
    #   CH4 bus -> central heat bus
    #
    # This version does not yet force a fixed heat-to-power coupling.
    "central_gas_chp": {
        "active": True,

        "electric_link_name": "swfl_real_central_gas_CHP",
        "heat_link_name": "swfl_real_central_gas_CHP_heat",

        "electric_capacity_mw": 241.0,
        "heat_capacity_mw": 370.0,

        "gas_bus": "biogas_sh_swfl_ch4_bus",
        "ac_bus": "33935",
        "heat_bus": "swfl_real_central_heat_bus",

        "carrier_el": "central_gas_CHP",
        "carrier_heat": "central_gas_CHP_heat",

        "p_nom_is_output_capacity": True,
        "electric_efficiency": 1.0,
        "heat_efficiency": 1.0,

        "extendable": False,
        "p_min_pu": 0.0,
        "p_max_pu": 1.0,

        "marginal_cost": 0.0,
        "capital_cost": 0.0,
    },

    # ------------------------------------------------------------------
    # Optional reserve gas boiler
    # ------------------------------------------------------------------
    "reserve_gas_boiler": {
        "active": False,

        "name": "swfl_real_reserve_gas_boiler",
        "heat_capacity_mw": 203.0,

        "gas_bus": "biogas_sh_swfl_ch4_bus",
        "heat_bus": "swfl_real_central_heat_bus",

        "carrier": "central_gas_boiler",

        "p_nom_is_output_capacity": True,
        "efficiency": 1.0,

        "extendable": False,
        "p_min_pu": 0.0,
        "p_max_pu": 1.0,

        "marginal_cost": 0.0,
        "capital_cost": 0.0,
    },

    # ------------------------------------------------------------------
    # Future SWFL large heat pumps
    # ------------------------------------------------------------------
    "future_heat_pumps": {
        # First remove existing eGon heat pumps inside the selected MV districts.
        "remove_existing_central_heat_pumps": True,

        # False = remove old eGon heat pumps but add no new SWFL heat pumps.
        # True  = add selected units below.
        "active": False,

        # Flexible options:
        #
        # none:
        # "active": False
        #
        # only GWP 1:
        # "active": True,
        # "active_units": ["swfl_gwp_1"]
        #
        # only GWP 2:
        # "active": True,
        # "active_units": ["swfl_gwp_2"]
        #
        # both:
        # "active": True,
        # "active_units": ["swfl_gwp_1", "swfl_gwp_2"]

        "carrier": "central_heat_pump",
        "default_cop": 3.0,

        "extendable": False,
        "p_min_pu": 0.0,
        "p_max_pu": 1.0,

        "marginal_cost": 0.0,
        "capital_cost": 0.0,

        "units": [
            {
                "name": "swfl_gwp_1",
                "active": True,
                "heat_capacity_mw": 60.0,
                "cop": 3.0,
                "planned_year": 2028,
                "ac_bus": "33935",
                "heat_bus": "swfl_real_central_heat_bus",
            },
            {
                "name": "swfl_gwp_2",
                "active": True,
                "heat_capacity_mw": 60.0,
                "cop": 3.0,
                "planned_year": None,
                "ac_bus": "33935",
                "heat_bus": "swfl_real_central_heat_bus",
            },
        ],
    },
}