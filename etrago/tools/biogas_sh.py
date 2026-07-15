# -*- coding: utf-8 -*-
"""
Biogas-SH integration for eTraGo.

This module adds project-specific Biogas-SH plants to an eTraGo/PyPSA network.
It follows the modelling decisions agreed for the Biogas-SH project:

Scenario dimensions
-------------------
1. Gas connection target:
   - "swfl"
   - "gas_grid"
   In the current project setup both can connect to target CH4 bus 47538.
   The distinction is kept as scenario metadata and for future cost/demand
   extensions.

2. Local generation switch:
   - add_local_generation = True/False

3. Gas generation switch:
   - add_gas_generation = True/False

Plant treatment
---------------
- AC-only plants are represented as industrial_biomass_CHP.
- AC + heat plants are represented as two separate fixed-capacity generators:
    central_biomass_CHP       -> electricity
    central_biomass_CHP_heat  -> heat
- Gas/SWFL option is represented as CH4_biogas production.

Demand-side connection
----------------------
- AC generators are connected to the AC load bus in the same ding0 MV grid
  district. If no AC load bus exists there, the nearest/neighbouring district
  with AC load is used.
- Heat generators are connected to the rural_heat bus/load in the same ding0
  MV grid district. In eTraGo this is usually the bus1 side of a
  rural_heat_pump link. If no rural_heat bus exists there, the nearest/
  neighbouring district with rural_heat is used.
- Multiple CHP_heat generators in the same MV district are connected to the
  same rural_heat bus. No additional heat link is required; PyPSA nodal
  balance lets generators on the same bus supply the load.

Gas topology
------------
Two gas topologies are supported:

1. direct_at_target
   Add CH4_biogas generator directly at target_ch4_bus.

2. producer_bus_link  [recommended]
   Add a plant-specific CH4 bus at the plant location, add the CH4_biogas
   generator there, and connect it to target_ch4_bus by a CH4 link. This is
   closer to the existing eGon CH4_biogas producer-bus topology.

Resource constraint
-------------------
This module only adds assets. The shared annual raw-biogas resource constraint
must be added in etrago/tools/constraints.py and activated through
args["extra_functionality"]["biogas_sh_resource"].

Expected minimum CSV columns
----------------------------
plant_id
plant_name
raw_biogas_mwh_hs_a
biomethane_mwh_hs_a_at_96pct
biomethane_price_eur_per_mwh_hs_at_96pct
lat
lon

Recommended CSV columns
-----------------------
hbl_95_mwel
installed_electric_capacity_mwel
useful_heat_mwh_a
plant_category            # optional: ac_only or ac_heat
ac_bus_for_model          # optional, can be auto-mapped
heat_bus_for_model        # optional, can be auto-mapped

Author: Biogas-SH project-specific eTraGo extension
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Generic helpers
# =============================================================================


def _is_missing(value) -> bool:
    """Return True for NaN/None/empty-string values."""
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except TypeError:
        pass
    return str(value).strip() == ""



def _clean_id(value) -> Optional[str]:
    """Convert numeric ids such as 30872.0 to '30872'."""
    if _is_missing(value):
        return None
    try:
        f = float(value)
        if f.is_integer():
            return str(int(f))
    except (TypeError, ValueError):
        pass
    return str(value).strip()



def _safe_float(row, column: str, default=0.0) -> float:
    """Read a float from a dataframe row; return default if missing."""
    if column not in row.index or _is_missing(row[column]):
        return float(default)
    try:
        return float(row[column])
    except (TypeError, ValueError):
        return float(default)



def _safe_str(row, column: str, default=None) -> Optional[str]:
    """Read a string from a dataframe row; return default if missing."""
    if column not in row.index or _is_missing(row[column]):
        return default
    return str(row[column]).strip()



def _as_bool(value, default=False) -> bool:
    """Robust bool parser for args values."""
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)



def _ensure_carrier(network, carrier: str) -> None:
    """Add carrier if it does not yet exist."""
    if carrier not in network.carriers.index:
        network.add("Carrier", carrier)



def _remove_component_if_exists(network, component: str, name: str) -> None:
    """Remove a PyPSA component if it already exists."""
    table = getattr(network, component.lower() + "s")
    if name in table.index:
        network.mremove(component, [name])



def _read_biogas_sh_csv(csv_path: str | Path) -> pd.DataFrame:
    """Read and validate the Biogas-SH plant CSV."""
    csv_path = Path(csv_path).expanduser()
    if not csv_path.exists():
        raise FileNotFoundError(f"Biogas-SH CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required = [
        "plant_id",
        "plant_name",
        "raw_biogas_mwh_hs_a",
        "biomethane_mwh_hs_a_at_96pct",
        "biomethane_price_eur_per_mwh_hs_at_96pct",
        "lat",
        "lon",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "Biogas-SH CSV is missing required columns: " + ", ".join(missing)
        )

    df = df[df["raw_biogas_mwh_hs_a"].fillna(0) > 0].copy()
    df["plant_id"] = df["plant_id"].astype(int)
    return df



def _distance_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Approximate haversine distance in km."""
    r = 6371.0
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return float(r * c)


# =============================================================================
# MV grid district and demand-bus mapping
# =============================================================================


def _read_mv_grid_districts(settings):
    """Read ding0 MV grid district polygons from GeoPackage."""
    try:
        import geopandas as gpd  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "geopandas is required for Biogas-SH auto_map_demand_buses=True. "
            "Install geopandas or provide ac_bus_for_model and heat_bus_for_model "
            "directly in the CSV."
        ) from exc

    gpkg = settings.get("mv_grid_districts_gpkg")
    if gpkg is None:
        raise ValueError(
            "args['biogas_sh']['mv_grid_districts_gpkg'] is required when "
            "auto_map_demand_buses=True."
        )

    gpkg = Path(gpkg).expanduser()
    if not gpkg.exists():
        raise FileNotFoundError(f"MV grid districts GeoPackage not found: {gpkg}")

    import geopandas as gpd

    layer = settings.get("mv_grid_layer")
    districts = gpd.read_file(gpkg) if not layer else gpd.read_file(gpkg, layer=layer)

    if districts.empty:
        raise ValueError(f"MV grid districts are empty: {gpkg}")

    districts = districts.copy()
    districts["geometry"] = districts.geometry.buffer(0)
    return districts



def _detect_column(df: pd.DataFrame, preferred: Optional[str], candidates: List[str]) -> str:
    """Return preferred column if present, otherwise first candidate present."""
    if preferred and preferred in df.columns:
        return preferred
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        "Could not detect required column. Tried: " + ", ".join([str(preferred)] + candidates)
    )



def _district_id_column(districts: pd.DataFrame, settings) -> tuple[pd.DataFrame, str]:
    """Return districts with a usable district id column."""
    district_id_col = settings.get("mv_grid_id_column")
    if district_id_col and district_id_col in districts.columns:
        return districts, district_id_col

    districts = districts.reset_index().rename(columns={"index": "_mv_grid_id"})
    return districts, "_mv_grid_id"



def _plant_points_gdf(df: pd.DataFrame, target_crs):
    """Return plant points as GeoDataFrame in target CRS."""
    import geopandas as gpd

    plants = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326",
    )
    if target_crs is not None:
        plants = plants.to_crs(target_crs)
    return plants



def _map_plants_to_districts(df, districts, settings):
    """
    Map Biogas-SH plant points to ding0 MV grid district polygons.

    The GeoPackage may have only one identifier column, e.g. 'name'.
    This function avoids duplicate column labels by renaming district
    columns before the spatial join.
    """

    import geopandas as gpd

    def _find_col(dataframe, candidates, label):
        """Return first matching column name from candidates."""
        for col in candidates:
            if col and col in dataframe.columns:
                return col
        raise ValueError(
            f"Could not detect {label}. Tried: {candidates}. "
            f"Available columns are: {list(dataframe.columns)}"
        )

    df = df.copy()
    districts = districts.copy()

    # Detect plant coordinate columns
    lon_col = _find_col(
        df,
        ["lon", "longitude", "x", "plant_lon", "bus_x"],
        "plant longitude column",
    )

    lat_col = _find_col(
        df,
        ["lat", "latitude", "y", "plant_lat", "bus_y"],
        "plant latitude column",
    )

    # Use configured MV grid ID column, normally 'name'
    id_col = settings.get("mv_grid_id_column") or "name"

    if id_col not in districts.columns:
        raise ValueError(
            f"Configured mv_grid_id_column='{id_col}' not found in "
            f"MV grid districts. Available columns are: "
            f"{list(districts.columns)}"
        )

    # Bus column may be the same as ID column, e.g. both are 'name'
    bus_col = settings.get("mv_grid_bus_column")

    if bus_col is not None and bus_col not in districts.columns:
        raise ValueError(
            f"Configured mv_grid_bus_column='{bus_col}' not found in "
            f"MV grid districts. Available columns are: "
            f"{list(districts.columns)}"
        )

    # Build clean district table with unique internal names
    districts_join = districts[[id_col, "geometry"]].copy()
    districts_join = districts_join.rename(
        columns={id_col: "__mv_grid_id__"}
    )

    if bus_col is not None:
        districts_join["__mv_grid_bus__"] = districts[bus_col].astype(str)
    else:
        districts_join["__mv_grid_bus__"] = districts_join[
            "__mv_grid_id__"
        ].astype(str)

    districts_join["__mv_grid_id__"] = districts_join[
        "__mv_grid_id__"
    ].astype(str)

    # Build plant GeoDataFrame
    plants_gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs="EPSG:4326",
    )

    # Make CRS consistent
    if districts_join.crs is None:
        districts_join = districts_join.set_crs("EPSG:4326")
    else:
        districts_join = districts_join.to_crs("EPSG:4326")

    # Spatial join: plant inside MV district
    joined = gpd.sjoin(
        plants_gdf,
        districts_join,
        how="left",
        predicate="within",
    )

    # If a plant lies exactly on a border or outside due to geometry precision,
    # use nearest MV district as fallback.
    missing = joined["__mv_grid_id__"].isna()

    if missing.any():
        nearest = gpd.sjoin_nearest(
            plants_gdf.loc[missing],
            districts_join,
            how="left",
            distance_col="__mv_grid_distance__",
        )

        joined.loc[missing, "__mv_grid_id__"] = nearest[
            "__mv_grid_id__"
        ].values
        joined.loc[missing, "__mv_grid_bus__"] = nearest[
            "__mv_grid_bus__"
        ].values

    # Store results back into the normal DataFrame
    df["ding0_mv_grid_id"] = joined["__mv_grid_id__"].astype(str).values
    df["ding0_mv_grid_bus_id"] = joined["__mv_grid_bus__"].astype(str).values

    return df



def _network_bus_points(network, bus_ids: Iterable[str], target_crs):
    """Return selected network buses as GeoDataFrame."""
    import geopandas as gpd

    ids = [str(b) for b in bus_ids if str(b) in network.buses.index]
    if not ids:
        return gpd.GeoDataFrame(columns=["bus", "geometry"], geometry="geometry", crs=target_crs)

    buses = network.buses.loc[ids].copy()
    buses["bus"] = buses.index.astype(str)
    gdf = gpd.GeoDataFrame(
        buses[["bus", "x", "y"]],
        geometry=gpd.points_from_xy(buses["x"], buses["y"]),
        crs="EPSG:4326",
    )
    if target_crs is not None:
        gdf = gdf.to_crs(target_crs)
    return gdf



def _candidate_ac_load_buses(network, settings) -> List[str]:
    """Return buses with AC load."""
    ac_carrier = settings.get("ac_load_carrier", "AC")
    candidates = set()

    if not network.loads.empty:
        loads = network.loads.copy()
        if "carrier" in loads.columns:
            candidates.update(loads.loc[loads.carrier == ac_carrier, "bus"].astype(str))

        ac_buses = network.buses.index[network.buses.carrier == "AC"].astype(str)
        candidates.update(loads.loc[loads.bus.astype(str).isin(ac_buses), "bus"].astype(str))

    return sorted([b for b in candidates if b in network.buses.index])



def _candidate_rural_heat_buses(network, settings) -> List[str]:
    """
    Return buses associated with rural heat demand.

    Primary source: bus1 of rural_heat_pump links.
    Fallbacks: loads/carrier or bus carrier containing rural_heat.
    """
    heat_carrier = settings.get("heat_demand_carrier", "rural_heat")
    heat_pump_carrier = settings.get("rural_heat_pump_carrier", "rural_heat_pump")

    candidates = set()

    if hasattr(network, "links") and not network.links.empty:
        hp = network.links[network.links.carrier.astype(str) == heat_pump_carrier]
        if not hp.empty and "bus1" in hp.columns:
            candidates.update(hp.bus1.astype(str))

    if not network.loads.empty:
        loads = network.loads.copy()
        if "carrier" in loads.columns:
            candidates.update(
                loads.loc[
                    loads.carrier.astype(str).str.contains(heat_carrier, na=False),
                    "bus",
                ].astype(str)
            )

        rural_heat_buses = network.buses.index[
            network.buses.carrier.astype(str).str.contains(heat_carrier, na=False)
        ].astype(str)
        candidates.update(loads.loc[loads.bus.astype(str).isin(rural_heat_buses), "bus"].astype(str))

    return sorted([b for b in candidates if b in network.buses.index])



def _load_energy_by_bus(network, buses: Iterable[str], carrier_contains: Optional[str] = None) -> Dict[str, float]:
    """Approximate load energy by bus; used to choose among multiple demand buses."""
    buses = [str(b) for b in buses]
    energy = {b: 0.0 for b in buses}

    if network.loads.empty:
        return energy

    loads = network.loads.copy()
    if carrier_contains and "carrier" in loads.columns:
        loads = loads[loads.carrier.astype(str).str.contains(carrier_contains, na=False)]

    for bus in buses:
        load_ids = loads.index[loads.bus.astype(str) == bus]
        if len(load_ids) == 0:
            continue

        if hasattr(network, "loads_t") and hasattr(network.loads_t, "p_set") and not network.loads_t.p_set.empty:
            existing = [l for l in load_ids if l in network.loads_t.p_set.columns]
            if existing:
                try:
                    weights = network.snapshot_weightings.generators.loc[network.snapshots]
                    energy[bus] = float(network.loads_t.p_set[existing].sum(axis=1).mul(weights, axis=0).sum())
                except Exception:
                    energy[bus] = float(network.loads_t.p_set[existing].sum().sum())
            else:
                energy[bus] = float(len(load_ids))
        else:
            energy[bus] = float(len(load_ids))

    return energy



def _map_candidate_buses_to_districts(network, districts, candidate_buses: List[str], settings) -> pd.DataFrame:
    """
    Assign candidate demand buses to MV districts.

    This version avoids duplicate column names like 'name' / 'name_right'
    by renaming the MV district id column before spatial joins.
    """
    import geopandas as gpd
    import pandas as pd

    if not candidate_buses:
        return pd.DataFrame(columns=["bus", "ding0_mv_grid_id"])

    districts, district_id_col = _district_id_column(districts, settings)

    if district_id_col not in districts.columns:
        raise ValueError(
            f"District id column '{district_id_col}' not found. "
            f"Available columns: {list(districts.columns)}"
        )

    # Rename district id to an internal unique name before joining.
    districts_join = districts[[district_id_col, "geometry"]].copy()
    districts_join = districts_join.rename(
        columns={district_id_col: "__mv_grid_id__"}
    )
    districts_join["__mv_grid_id__"] = districts_join[
        "__mv_grid_id__"
    ].astype(str)

    bus_points = _network_bus_points(network, candidate_buses, districts_join.crs)

    if bus_points.empty:
        return pd.DataFrame(columns=["bus", "ding0_mv_grid_id"])

    # First try: bus point inside MV district polygon.
    joined = gpd.sjoin(
        bus_points,
        districts_join[["__mv_grid_id__", "geometry"]],
        how="left",
        predicate="within",
    )

    # Fallback for buses on boundaries or just outside polygons.
    missing_mask = joined["__mv_grid_id__"].isna()

    if missing_mask.any():
        missing_index = joined.index[missing_mask]

        # Important: use the original bus_points, not the already joined frame.
        # Otherwise columns like 'name' get duplicated/renamed by GeoPandas.
        nearest = gpd.sjoin_nearest(
            bus_points.loc[missing_index],
            districts_join[["__mv_grid_id__", "geometry"]],
            how="left",
            distance_col="_bus_join_distance",
        )

        joined.loc[missing_index, "__mv_grid_id__"] = nearest[
            "__mv_grid_id__"
        ].values

    out = pd.DataFrame(
        {
            "bus": joined["bus"].astype(str).values,
            "ding0_mv_grid_id": joined["__mv_grid_id__"].astype(str).values,
        }
    )

    # Avoid duplicate bus rows if a point hits multiple polygons.
    out = out.dropna(subset=["ding0_mv_grid_id"])
    out = out.drop_duplicates(subset=["bus"])

    return out



def _choose_bus_for_district(
    plant_district_id,
    candidate_map: pd.DataFrame,
    districts,
    demand_energy: Dict[str, float],
    settings,
) -> Optional[str]:
    """Choose same-district demand bus; fallback to nearest candidate district."""
    if candidate_map.empty:
        return None

    same = candidate_map[candidate_map["ding0_mv_grid_id"] == plant_district_id]
    if not same.empty:
        return max(same.bus.astype(str), key=lambda b: demand_energy.get(str(b), 0.0))

    if not _as_bool(settings.get("fallback_to_neighbor_area", True), True):
        return None

    # Fallback: choose largest-load candidate bus if geometry lookup fails.
    districts, district_id_col = _district_id_column(districts, settings)
    plant_poly = districts.loc[districts[district_id_col] == plant_district_id]
    if plant_poly.empty:
        return max(candidate_map.bus.astype(str), key=lambda b: demand_energy.get(str(b), 0.0))

    plant_centroid = plant_poly.geometry.iloc[0].centroid
    candidate_districts = candidate_map.drop_duplicates("ding0_mv_grid_id").merge(
        districts[[district_id_col, "geometry"]],
        left_on="ding0_mv_grid_id",
        right_on=district_id_col,
        how="left",
    )
    candidate_districts = candidate_districts.dropna(subset=["geometry"])
    if candidate_districts.empty:
        return max(candidate_map.bus.astype(str), key=lambda b: demand_energy.get(str(b), 0.0))

    candidate_districts["_distance"] = candidate_districts.geometry.centroid.distance(plant_centroid)
    nearest_district = candidate_districts.sort_values("_distance").iloc[0]["ding0_mv_grid_id"]
    nearest_buses = candidate_map[candidate_map["ding0_mv_grid_id"] == nearest_district].bus.astype(str)
    return max(nearest_buses, key=lambda b: demand_energy.get(str(b), 0.0))



def _auto_map_demand_buses(self, df: pd.DataFrame, settings) -> pd.DataFrame:
    """Add ac_bus_for_model and heat_bus_for_model from MV districts and demand buses."""
    districts = _read_mv_grid_districts(settings)
    df = _map_plants_to_districts(df, districts, settings)

    ac_candidates = _candidate_ac_load_buses(self.network, settings)
    heat_candidates = _candidate_rural_heat_buses(self.network, settings)

    ac_map = _map_candidate_buses_to_districts(self.network, districts, ac_candidates, settings)
    heat_map = _map_candidate_buses_to_districts(self.network, districts, heat_candidates, settings)

    ac_energy = _load_energy_by_bus(self.network, ac_candidates, carrier_contains=settings.get("ac_load_carrier", "AC"))
    heat_energy = _load_energy_by_bus(self.network, heat_candidates, carrier_contains=settings.get("heat_demand_carrier", "rural_heat"))

    ac_buses = []
    heat_buses = []
    for _, row in df.iterrows():
        district_id = row.get("ding0_mv_grid_id")
        ac_buses.append(_choose_bus_for_district(district_id, ac_map, districts, ac_energy, settings))
        heat_buses.append(_choose_bus_for_district(district_id, heat_map, districts, heat_energy, settings))

    if "ac_bus_for_model" not in df.columns:
        df["ac_bus_for_model"] = ac_buses
    else:
        df["ac_bus_for_model"] = [
            _clean_id(existing) if not _is_missing(existing) else auto
            for existing, auto in zip(df["ac_bus_for_model"], ac_buses)
        ]

    if "heat_bus_for_model" not in df.columns:
        df["heat_bus_for_model"] = heat_buses
    else:
        df["heat_bus_for_model"] = [
            _clean_id(existing) if not _is_missing(existing) else auto
            for existing, auto in zip(df["heat_bus_for_model"], heat_buses)
        ]

    return df


# =============================================================================
# Capacity and plant category helpers
# =============================================================================


def _plant_has_heat(row, settings) -> bool:
    """Return True for AC+heat plants."""
    if "plant_category" in row.index and not _is_missing(row["plant_category"]):
        return str(row["plant_category"]).strip().lower() in {
            "ac_heat",
            "heat",
            "chp",
            "central_biomass_chp",
        }
    return _safe_float(row, "useful_heat_mwh_a", 0.0) > float(settings.get("heat_threshold_mwh_a", 0.0))



def _get_electric_p_nom(row, settings) -> float:
    """Return electricity p_nom in MW_el."""
    p_nom_column = settings.get("electric_capacity_column", "hbl_95_mwel")
    p_nom = _safe_float(row, p_nom_column, default=np.nan)
    if not np.isfinite(p_nom) or p_nom <= 0:
        p_nom = _safe_float(row, "installed_electric_capacity_mwel", default=0.0)
    return max(float(p_nom), 0.0)



def _get_heat_p_nom(row, settings) -> float:
    """Return heat p_nom in MW_th."""
    if "p_nom_heat_mw" in row.index and not _is_missing(row["p_nom_heat_mw"]):
        return max(_safe_float(row, "p_nom_heat_mw"), 0.0)

    method = settings.get("heat_capacity_method", "annual_heat_div_8760")
    if method != "annual_heat_div_8760":
        raise ValueError(f"Unsupported heat_capacity_method: {method}")

    heat_mwh_a = _safe_float(row, "useful_heat_mwh_a", default=0.0)
    hours = float(settings.get("hours_for_heat_capacity", 8760.0))
    return max(heat_mwh_a / hours, 0.0) if hours > 0 else 0.0



def _get_ch4_p_nom(row, settings) -> float:
    """Return CH4_biogas p_nom in MW_CH4."""
    if "p_nom_ch4_mw" in row.index and not _is_missing(row["p_nom_ch4_mw"]):
        return max(_safe_float(row, "p_nom_ch4_mw"), 0.0)

    biomethane_mwh_a = _safe_float(row, "biomethane_mwh_hs_a_at_96pct", default=0.0)
    hours = float(settings.get("hours_for_gas_capacity", 8760.0))
    return max(biomethane_mwh_a / hours, 0.0) if hours > 0 else 0.0


# =============================================================================
# Asset creation
# =============================================================================


def _add_electricity_generator(network, plant_id: int, ac_bus: str, carrier: str, p_nom: float, mc: float) -> None:
    """Add plant electricity generator."""
    name = f"biogas_sh_el_{plant_id}"
    _remove_component_if_exists(network, "Generator", name)
    network.add(
        "Generator",
        name,
        bus=ac_bus,
        carrier=carrier,
        p_nom=p_nom,
        p_nom_extendable=False,
        p_nom_min=0.0,
        p_min_pu=0.0,
        p_max_pu=1.0,
        marginal_cost=mc,
        capital_cost=0.0,
        efficiency=1.0,
        build_year=0,
        lifetime=np.inf,
        committable=False,
    )



def _add_heat_generator(network, plant_id: int, heat_bus: str, carrier: str, p_nom: float, mc: float) -> None:
    """Add plant heat generator."""
    name = f"biogas_sh_heat_{plant_id}"
    _remove_component_if_exists(network, "Generator", name)
    network.add(
        "Generator",
        name,
        bus=heat_bus,
        carrier=carrier,
        p_nom=p_nom,
        p_nom_extendable=False,
        p_nom_min=0.0,
        p_min_pu=0.0,
        p_max_pu=1.0,
        marginal_cost=mc,
        capital_cost=0.0,
        efficiency=1.0,
        build_year=0,
        lifetime=np.inf,
        committable=False,
    )



def _add_ch4_generator_direct(network, plant_id: int, ch4_bus: str, p_nom: float, mc: float) -> None:
    """Add CH4_biogas generator directly at target CH4 bus."""
    name = f"biogas_sh_ch4_{plant_id}"
    _remove_component_if_exists(network, "Generator", name)
    network.add(
        "Generator",
        name,
        bus=ch4_bus,
        carrier="CH4_biogas",
        p_nom=p_nom,
        p_nom_extendable=False,
        p_nom_min=0.0,
        p_min_pu=0.0,
        p_max_pu=1.0,
        marginal_cost=mc,
        capital_cost=0.0,
        efficiency=1.0,
        build_year=0,
        lifetime=np.inf,
        committable=False,
    )



def _add_ch4_generator_with_link(network, row, plant_id: int, target_ch4_bus: str, p_nom: float, mc: float, settings) -> None:
    """Add plant CH4 bus, CH4_biogas generator, and CH4 link to target bus."""
    plant_bus = f"biogas_sh_ch4_bus_{plant_id}"
    gen_name = f"biogas_sh_ch4_{plant_id}"
    link_name = f"biogas_sh_ch4_link_{plant_id}_to_{target_ch4_bus}"

    lon = _safe_float(row, "lon")
    lat = _safe_float(row, "lat")

    _remove_component_if_exists(network, "Generator", gen_name)
    _remove_component_if_exists(network, "Link", link_name)

    if plant_bus in network.buses.index:
        # Keep bus but update coordinates/carrier if needed.
        network.buses.loc[plant_bus, "carrier"] = "CH4"
        network.buses.loc[plant_bus, "country"] = "DE"
        network.buses.loc[plant_bus, "x"] = lon
        network.buses.loc[plant_bus, "y"] = lat
    else:
        network.add(
            "Bus",
            plant_bus,
            carrier="CH4",
            country="DE",
            x=lon,
            y=lat,
        )

    network.add(
        "Generator",
        gen_name,
        bus=plant_bus,
        carrier="CH4_biogas",
        p_nom=p_nom,
        p_nom_extendable=False,
        p_nom_min=0.0,
        p_min_pu=0.0,
        p_max_pu=1.0,
        marginal_cost=mc,
        capital_cost=0.0,
        efficiency=1.0,
        build_year=0,
        lifetime=np.inf,
        committable=False,
    )

    link_factor = float(settings.get("ch4_link_p_nom_factor", 1.0))
    link_p_nom = max(p_nom * link_factor, 0.0)

    p_min_pu = float(settings.get("ch4_link_p_min_pu", 0.0))  # one-way injection by default
    efficiency = float(settings.get("ch4_link_efficiency", 1.0))
    link_mc = float(settings.get("ch4_link_marginal_cost", 0.0))
    link_capital_cost = float(settings.get("ch4_link_capital_cost", 0.0))

    network.add(
        "Link",
        link_name,
        bus0=plant_bus,
        bus1=target_ch4_bus,
        carrier=settings.get("ch4_link_carrier", "biogas_sh_gas_grid_injection"),
        p_nom=link_p_nom,
        p_nom_extendable=_as_bool(settings.get("ch4_link_extendable", False), False),
        p_nom_min=0.0,
        p_min_pu=p_min_pu,
        p_max_pu=1.0,
        efficiency=efficiency,
        marginal_cost=link_mc,
        capital_cost=link_capital_cost,
    )

    # Optional extra metadata if the target bus has coordinates.
    try:
        tx = float(network.buses.loc[target_ch4_bus, "x"])
        ty = float(network.buses.loc[target_ch4_bus, "y"])
        network.links.loc[link_name, "length"] = _distance_km(lon, lat, tx, ty)
    except Exception:
        pass



# =============================================================================
# Flexible gas route and SWFL-direct helpers
# =============================================================================


def _resolve_biogas_sh_route_switches(settings):
    """Resolve flexible scenario switches."""
    mode = str(settings.get("scenario_mode", "custom")).strip().lower()
    if mode == "onsite":
        return mode, True, False, False
    if mode in {"gas_grid", "grid"}:
        return mode, False, True, False
    if mode in {"swfl", "swfl_direct"}:
        return mode, False, False, True
    if mode in {"hybrid", "all", "competition"}:
        return mode, True, True, True
    if mode != "custom":
        raise ValueError("Unsupported biogas_sh.scenario_mode. Use custom, onsite, gas_grid, swfl, or hybrid.")
    add_local = _as_bool(settings.get("add_local_generation", True), True)
    add_grid = _as_bool(settings.get("add_gas_grid_generation", settings.get("add_gas_generation", True)), True)
    swfl_cfg = settings.get("swfl_direct", {}) or {}
    add_swfl = _as_bool(settings.get("add_swfl_direct_supply", swfl_cfg.get("active", False)), False)
    return mode, add_local, add_grid, add_swfl


def _swfl_direct_settings(settings) -> dict:
    cfg = settings.get("swfl_direct", {})
    if cfg is None:
        return {}
    if not isinstance(cfg, dict):
        raise TypeError("biogas_sh.swfl_direct must be a dictionary if provided.")
    return cfg


def _consumer_link_ids_from_settings(swfl_cfg: dict) -> List[str]:
    ids = swfl_cfg.get("consumer_link_ids", []) or []
    return [str(_clean_id(i)) for i in ids if _clean_id(i) is not None]


def _infer_swfl_bus_coordinates(network, swfl_cfg: dict, public_ch4_bus: str, consumer_link_ids: List[str]):
    if "swfl_ch4_bus_x" in swfl_cfg and "swfl_ch4_bus_y" in swfl_cfg:
        return float(swfl_cfg["swfl_ch4_bus_x"]), float(swfl_cfg["swfl_ch4_bus_y"])
    xs, ys = [], []
    for link_id in consumer_link_ids:
        if link_id not in network.links.index:
            continue
        bus1 = str(network.links.loc[link_id, "bus1"])
        if bus1 in network.buses.index:
            xs.append(float(network.buses.loc[bus1, "x"]))
            ys.append(float(network.buses.loc[bus1, "y"]))
    if xs and ys:
        return float(np.mean(xs)), float(np.mean(ys))
    if public_ch4_bus in network.buses.index:
        return float(network.buses.loc[public_ch4_bus, "x"]), float(network.buses.loc[public_ch4_bus, "y"])
    return 9.436502119171873, 54.79233181101448


def _ensure_swfl_ch4_bus_and_access(network, settings):
    """Create artificial SWFL CH4 bus and optional public-grid supply link."""
    swfl_cfg = _swfl_direct_settings(settings)
    public_ch4_bus = _clean_id(swfl_cfg.get("public_ch4_bus", settings.get("target_ch4_bus", "47538")))
    swfl_bus = _clean_id(swfl_cfg.get("swfl_ch4_bus", "biogas_sh_swfl_ch4_bus")) or "biogas_sh_swfl_ch4_bus"
    if public_ch4_bus not in network.buses.index:
        raise ValueError(f"SWFL public_ch4_bus {public_ch4_bus} not found in network.buses.")
    consumer_link_ids = _consumer_link_ids_from_settings(swfl_cfg)
    x, y = _infer_swfl_bus_coordinates(network, swfl_cfg, public_ch4_bus, consumer_link_ids)
    _ensure_carrier(network, "CH4")
    _ensure_carrier(network, "biogas_sh_swfl_direct")
    _ensure_carrier(network, "biogas_sh_swfl_grid_supply")
    if swfl_bus in network.buses.index:
        network.buses.loc[swfl_bus, "carrier"] = "CH4"
        network.buses.loc[swfl_bus, "country"] = "DE"
        network.buses.loc[swfl_bus, "x"] = x
        network.buses.loc[swfl_bus, "y"] = y
    else:
        network.add("Bus", swfl_bus, carrier="CH4", country="DE", x=x, y=y)
    redirected = []
    if _as_bool(swfl_cfg.get("redirect_consumer_links", True), True):
        for link_id in consumer_link_ids:
            if link_id not in network.links.index:
                logger.warning("SWFL consumer link %s not found; cannot redirect to %s.", link_id, swfl_bus)
                continue
            old_bus0 = str(network.links.loc[link_id, "bus0"])
            if old_bus0 == public_ch4_bus or _as_bool(swfl_cfg.get("redirect_even_if_bus0_differs", False), False):
                network.links.loc[link_id, "bus0"] = swfl_bus
                redirected.append(link_id)
    grid_supply_added = False
    if _as_bool(swfl_cfg.get("grid_supply_active", True), True):
        grid_supply_name = _clean_id(swfl_cfg.get("grid_supply_link", f"biogas_sh_swfl_grid_supply_{public_ch4_bus}_to_{swfl_bus}"))
        _remove_component_if_exists(network, "Link", grid_supply_name)
        if "grid_supply_p_nom" in swfl_cfg and not _is_missing(swfl_cfg["grid_supply_p_nom"]):
            grid_p_nom = float(swfl_cfg["grid_supply_p_nom"])
        else:
            p_nom_sum = 0.0
            for link_id in consumer_link_ids:
                if link_id in network.links.index and "p_nom" in network.links.columns:
                    try:
                        p_nom_sum += float(network.links.loc[link_id, "p_nom"])
                    except Exception:
                        pass
            grid_p_nom = p_nom_sum if p_nom_sum > 0 else float(swfl_cfg.get("grid_supply_p_nom_default", 1e6))
            grid_p_nom *= float(swfl_cfg.get("grid_supply_p_nom_factor", 1.0))
        network.add(
            "Link", grid_supply_name, bus0=public_ch4_bus, bus1=swfl_bus,
            carrier="biogas_sh_swfl_grid_supply", p_nom=grid_p_nom,
            p_nom_extendable=_as_bool(swfl_cfg.get("grid_supply_extendable", False), False),
            p_nom_min=0.0, p_min_pu=0.0, p_max_pu=1.0,
            efficiency=float(swfl_cfg.get("grid_supply_efficiency", 1.0)),
            marginal_cost=float(swfl_cfg.get("grid_supply_marginal_cost", 0.0)),
            capital_cost=float(swfl_cfg.get("grid_supply_capital_cost", 0.0)),
        )
        grid_supply_added = True
    if _as_bool(swfl_cfg.get("add_swfl_gas_load", False), False):
        load_name = _clean_id(swfl_cfg.get("swfl_gas_load", "biogas_sh_swfl_ch4_load"))
        _remove_component_if_exists(network, "Load", load_name)
        mwh_a = float(swfl_cfg.get("swfl_demand_mwh_a", 0.0))
        flat_mw = mwh_a / 8760.0 if mwh_a > 0 else 0.0
        network.add("Load", load_name, bus=swfl_bus, carrier="CH4", p_set=flat_mw)
    return swfl_bus, public_ch4_bus, redirected, grid_supply_added


def _ensure_plant_ch4_bus_and_generator(network, row, plant_id: int, p_nom: float, mc: float):
    plant_bus = f"biogas_sh_ch4_bus_{plant_id}"
    gen_name = f"biogas_sh_ch4_{plant_id}"
    lon = _safe_float(row, "lon")
    lat = _safe_float(row, "lat")
    _remove_component_if_exists(network, "Generator", gen_name)
    if plant_bus in network.buses.index:
        network.buses.loc[plant_bus, "carrier"] = "CH4"
        network.buses.loc[plant_bus, "country"] = "DE"
        network.buses.loc[plant_bus, "x"] = lon
        network.buses.loc[plant_bus, "y"] = lat
    else:
        network.add("Bus", plant_bus, carrier="CH4", country="DE", x=lon, y=lat)
    network.add(
        "Generator", gen_name, bus=plant_bus, carrier="CH4_biogas", p_nom=p_nom,
        p_nom_extendable=False, p_nom_min=0.0, p_min_pu=0.0, p_max_pu=1.0,
        marginal_cost=mc, capital_cost=0.0, efficiency=1.0, build_year=0,
        lifetime=np.inf, committable=False,
    )
    return plant_bus, gen_name


def _add_ch4_grid_link_from_plant(
    network,
    plant_id: int,
    plant_bus: str,
    target_ch4_bus: str,
    p_nom: float,
    settings,
) -> None:
    """Add one-way Biogas-SH plant CH4 bus -> public CH4 grid injection link."""
    link_name = f"biogas_sh_ch4_grid_link_{plant_id}_to_{target_ch4_bus}"

    _remove_component_if_exists(network, "Link", link_name)

    link_factor = float(settings.get("ch4_link_p_nom_factor", 1.0))
    link_p_nom = max(float(p_nom) * link_factor, 0.0)

    ch4_link_carrier = settings.get(
        "ch4_link_carrier",
        "biogas_sh_gas_grid_injection",
    )

    network.add(
        "Link",
        link_name,
        bus0=plant_bus,
        bus1=target_ch4_bus,
        carrier=ch4_link_carrier,
        p_nom=link_p_nom,
        p_nom_extendable=_as_bool(settings.get("ch4_link_extendable", False), False),
        p_nom_min=0.0,
        p_min_pu=float(settings.get("ch4_link_p_min_pu", 0.0)),
        p_max_pu=1.0,
        efficiency=float(settings.get("ch4_link_efficiency", 1.0)),
        marginal_cost=float(settings.get("ch4_link_marginal_cost", 0.0)),
        capital_cost=float(settings.get("ch4_link_capital_cost", 0.0)),
    )


def _add_swfl_direct_link_from_plant(network, plant_id: int, plant_bus: str, swfl_bus: str, p_nom: float, settings) -> None:
    swfl_cfg = _swfl_direct_settings(settings)
    link_name = f"biogas_sh_swfl_direct_link_{plant_id}_to_{swfl_bus}"
    _remove_component_if_exists(network, "Link", link_name)
    link_p_nom = max(p_nom * float(swfl_cfg.get("direct_link_p_nom_factor", 1.0)), 0.0)
    network.add(
        "Link", link_name, bus0=plant_bus, bus1=swfl_bus,
        carrier="biogas_sh_swfl_direct", p_nom=link_p_nom,
        p_nom_extendable=_as_bool(swfl_cfg.get("direct_link_extendable", False), False),
        p_nom_min=0.0, p_min_pu=0.0, p_max_pu=1.0,
        efficiency=float(swfl_cfg.get("direct_link_efficiency", 1.0)),
        marginal_cost=float(swfl_cfg.get("direct_link_marginal_cost", swfl_cfg.get("swfl_direct_transport_cost", 0.0))),
        capital_cost=float(swfl_cfg.get("direct_link_capital_cost", 0.0)),
    )


def _biogas_sh_storage_settings(settings) -> dict:
    """Read optional Biogas.SH gas-storage settings."""
    cfg = settings.get("gas_storage", {})
    if cfg is None:
        return {}
    if not isinstance(cfg, dict):
        raise TypeError("biogas_sh.gas_storage must be a dictionary.")
    return cfg


def _ensure_biogas_sh_single_storage(network, settings) -> tuple[str, bool]:
    """
    Add one central Biogas.SH CH4 storage.

    Topology:
        Biogas.SH plant CH4 buses
              -> biogas_sh_storage_ch4_bus
              <-> biogas_sh_ch4_store
              -> SWFL CH4 bus
              -> public CH4 grid
    """
    cfg = _biogas_sh_storage_settings(settings)

    if not _as_bool(cfg.get("active", False), False):
        return None, False

    storage_bus = str(cfg.get("bus", "biogas_sh_storage_ch4_bus"))
    storage_name = str(cfg.get("store", "biogas_sh_ch4_store"))

    # approximate location near Flensburg / SWFL
    x = float(cfg.get("x", 9.436502119171873))
    y = float(cfg.get("y", 54.79233181101448))

    if storage_bus not in network.buses.index:
        network.add(
            "Bus",
            storage_bus,
            carrier="CH4",
            x=x,
            y=y,
        )

    _remove_component_if_exists(network, "Store", storage_name)

    network.add(
        "Store",
        storage_name,
        bus=storage_bus,
        carrier="CH4",
        e_nom=float(cfg.get("e_nom_mwh", 500.0)),
        e_nom_extendable=_as_bool(cfg.get("e_nom_extendable", False), False),
        e_nom_min=float(cfg.get("e_nom_min", 0.0)),
        e_initial=float(cfg.get("e_initial", 0.0)),
        e_cyclic=_as_bool(cfg.get("e_cyclic", True), True),
        standing_loss=float(cfg.get("standing_loss", 0.0)),
        marginal_cost=float(cfg.get("marginal_cost", 0.0)),
        capital_cost=float(cfg.get("capital_cost", 0.0)),
    )

    return storage_bus, True


def _add_biogas_sh_storage_output_links(
    network,
    storage_bus: str,
    settings,
    add_gas_grid_generation: bool,
    add_swfl_direct_supply: bool,
    target_ch4_bus: str = None,
    swfl_bus: str = None,
) -> None:
    """
    Add output links from the single Biogas.SH storage bus to:
    1) public CH4 grid
    2) SWFL CH4 bus
    """
    cfg = _biogas_sh_storage_settings(settings)

    if add_gas_grid_generation:
        if target_ch4_bus is None:
            raise ValueError("target_ch4_bus is required for storage-to-grid link.")

        grid_link = str(
            cfg.get(
                "grid_link",
                f"biogas_sh_storage_to_grid_{target_ch4_bus}",
            )
        )

        _remove_component_if_exists(network, "Link", grid_link)

        network.add(
            "Link",
            grid_link,
            bus0=storage_bus,
            bus1=target_ch4_bus,
            carrier=str(cfg.get("grid_link_carrier", "biogas_sh_storage_to_grid")),
            p_nom=float(cfg.get("grid_link_p_nom_mw", 50.0)),
            p_nom_extendable=_as_bool(cfg.get("grid_link_extendable", False), False),
            p_nom_min=float(cfg.get("grid_link_p_nom_min", 0.0)),
            p_min_pu=float(cfg.get("grid_link_p_min_pu", 0.0)),
            p_max_pu=float(cfg.get("grid_link_p_max_pu", 1.0)),
            efficiency=float(cfg.get("grid_link_efficiency", 1.0)),
            marginal_cost=float(cfg.get("grid_link_marginal_cost", 0.0)),
            capital_cost=float(cfg.get("grid_link_capital_cost", 0.0)),
        )

    if add_swfl_direct_supply:
        if swfl_bus is None:
            raise ValueError("swfl_bus is required for storage-to-SWFL link.")

        swfl_link = str(
            cfg.get(
                "swfl_link",
                f"biogas_sh_storage_to_swfl_{swfl_bus}",
            )
        )

        _remove_component_if_exists(network, "Link", swfl_link)

        network.add(
            "Link",
            swfl_link,
            bus0=storage_bus,
            bus1=swfl_bus,
            carrier=str(cfg.get("swfl_link_carrier", "biogas_sh_storage_to_swfl")),
            p_nom=float(cfg.get("swfl_link_p_nom_mw", 50.0)),
            p_nom_extendable=_as_bool(cfg.get("swfl_link_extendable", False), False),
            p_nom_min=float(cfg.get("swfl_link_p_nom_min", 0.0)),
            p_min_pu=float(cfg.get("swfl_link_p_min_pu", 0.0)),
            p_max_pu=float(cfg.get("swfl_link_p_max_pu", 1.0)),
            efficiency=float(cfg.get("swfl_link_efficiency", 1.0)),
            marginal_cost=float(cfg.get("swfl_link_marginal_cost", 0.0)),
            capital_cost=float(cfg.get("swfl_link_capital_cost", 0.0)),
        )


# =============================================================================
# Public function attached to Etrago in network.py
# =============================================================================


def apply_biogas_sh_assets(self) -> None:
    """
    Add Biogas.SH assets with flexible onsite, public gas-grid, and SWFL-direct routes.

    Routes
    ------
    add_local_generation:
        Adds onsite electricity and heat generators at mapped AC / heat buses.

    add_gas_grid_generation:
        Adds plant CH4 buses, CH4_biogas generators, and links to the public CH4 grid.

    add_swfl_direct_supply:
        Adds plant CH4 buses, CH4_biogas generators, and direct links to the SWFL CH4 bus.

    Important
    ---------
    Onsite generators use custom Biogas.SH carriers by default so they remain
    identifiable after clustering/export:
        - biogas_sh_onsite_el
        - biogas_sh_onsite_chp_el
        - biogas_sh_onsite_chp_heat
    """

    # -------------------------------------------------------------------------
    # 1. Read and validate settings
    # -------------------------------------------------------------------------
    settings = self.args.get("biogas_sh", {})

    if not settings or not _as_bool(settings.get("active", False), False):
        logger.info("Biogas.SH assets not active.")
        return

    csv_path = settings.get("csv_path")
    if csv_path is None:
        raise ValueError("args['biogas_sh']['csv_path'] must be set.")

    route_mode, add_local_generation, add_gas_grid_generation, add_swfl_direct_supply = (
        _resolve_biogas_sh_route_switches(settings)
    )

    network = self.network

    # -------------------------------------------------------------------------
    # 2. Read Biogas.SH plant data and optionally map demand buses
    # -------------------------------------------------------------------------
    df = _read_biogas_sh_csv(csv_path)

    if _as_bool(settings.get("auto_map_demand_buses", True), True):
        df = _auto_map_demand_buses(self, df, settings)

    mapped_csv = settings.get("write_mapped_csv")
    if mapped_csv:
        mapped_csv = Path(mapped_csv).expanduser()
        mapped_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(mapped_csv, index=False)

    # -------------------------------------------------------------------------
    # 3. Carrier setup
    # -------------------------------------------------------------------------
    # Use custom carriers by default. This avoids mixing custom Biogas.SH onsite
    # generators with generic eGon biomass / CHP generators during clustering.
    ac_only_carrier = settings.get("ac_only_carrier", "biogas_sh_onsite_el")
    chp_el_carrier = settings.get("chp_el_carrier", "biogas_sh_onsite_chp_el")
    chp_heat_carrier = settings.get("chp_heat_carrier", "biogas_sh_onsite_chp_heat")

    ch4_link_carrier = settings.get(
        "ch4_link_carrier",
        "biogas_sh_gas_grid_injection",
    )

    required_carriers = [
        ac_only_carrier,
        chp_el_carrier,
        chp_heat_carrier,
        "CH4",
        "CH4_biogas",
        ch4_link_carrier,
        "biogas_sh_swfl_direct",
        "biogas_sh_swfl_grid_supply",
        "biogas_sh_storage_to_grid",
        "biogas_sh_storage_to_swfl",
    ]

    for carrier in required_carriers:
        _ensure_carrier(network, carrier)

    # -------------------------------------------------------------------------
    # 4. Cost and topology settings
    # -------------------------------------------------------------------------
    el_mc = float(settings.get("electricity_marginal_cost", 42.1))
    heat_mc = float(settings.get("heat_marginal_cost", 0.0))
    default_biomethane_cost = float(settings.get("default_biomethane_cost", 75.0))

    # Optional override from appl.py.
    # If None: use plant-specific CSV value.
    # If number: use this value for all Biogas.SH CH4_biogas generators.
    biomethane_price_override = settings.get(
        "biomethane_price_override_eur_per_mwh",
        None,
    )

    if biomethane_price_override is not None and not _is_missing(biomethane_price_override):
        biomethane_price_override = float(biomethane_price_override)
    else:
        biomethane_price_override = None

    target_ch4_bus = _clean_id(settings.get("target_ch4_bus", "47538"))
    gas_topology = str(settings.get("gas_topology", "producer_bus_link")).strip().lower()

    if gas_topology not in {"direct_at_target", "producer_bus_link"}:
        raise ValueError(
            "biogas_sh.gas_topology must be 'direct_at_target' or "
            "'producer_bus_link'."
        )

    if add_gas_grid_generation:
        if target_ch4_bus is None:
            raise ValueError("biogas_sh.target_ch4_bus is missing.")
        if target_ch4_bus not in network.buses.index:
            raise ValueError(
                f"Biogas.SH target_ch4_bus {target_ch4_bus} not found in network.buses."
            )

    # -------------------------------------------------------------------------
    # 5. Optional SWFL CH4 bus and public-grid supply access
    # -------------------------------------------------------------------------
    swfl_bus = None
    swfl_public_ch4_bus = None
    swfl_redirected_links = []
    swfl_grid_supply_added = False

    if add_swfl_direct_supply:
        (
            swfl_bus,
            swfl_public_ch4_bus,
            swfl_redirected_links,
            swfl_grid_supply_added,
        ) = _ensure_swfl_ch4_bus_and_access(network, settings)
        
        
    # -------------------------------------------------------------------------
    # Optional single Biogas.SH CH4 storage
    # -------------------------------------------------------------------------
    storage_cfg = _biogas_sh_storage_settings(settings)
    use_biogas_sh_storage = _as_bool(storage_cfg.get("active", False), False)

    storage_bus = None
    storage_added = False

    if use_biogas_sh_storage:
        storage_bus, storage_added = _ensure_biogas_sh_single_storage(
            network=network,
            settings=settings,
        )

        _add_biogas_sh_storage_output_links(
            network=network,
            storage_bus=storage_bus,
            settings=settings,
            add_gas_grid_generation=add_gas_grid_generation,
            add_swfl_direct_supply=add_swfl_direct_supply,
            target_ch4_bus=target_ch4_bus,
            swfl_bus=swfl_bus,
        )

    # -------------------------------------------------------------------------
    # 6. Counters and debug bookkeeping
    # -------------------------------------------------------------------------
    added_el = 0
    added_heat = 0
    added_ch4 = 0
    added_grid_links = 0
    added_swfl_links = 0

    added_el_capacity = 0.0
    added_heat_capacity = 0.0
    added_ch4_capacity = 0.0
    added_grid_link_capacity = 0.0
    added_swfl_link_capacity = 0.0

    potential_el_capacity = 0.0
    potential_heat_capacity = 0.0
    potential_ch4_capacity = 0.0

    debug_local = {
        "rows_total": 0,
        "local_block_entered": 0,
        "has_heat_true": 0,
        "ac_bus_missing": 0,
        "ac_bus_not_in_network": 0,
        "p_nom_el_zero": 0,
        "heat_bus_missing": 0,
        "heat_bus_not_in_network": 0,
        "p_nom_heat_zero": 0,
        "ch4_p_nom_zero": 0,
    }

    skipped = []

    # -------------------------------------------------------------------------
    # 7. Add plant assets
    # -------------------------------------------------------------------------
    for _, row in df.iterrows():
        debug_local["rows_total"] += 1

        plant_id = int(row["plant_id"])
        plant_name = str(row["plant_name"])
        has_heat = _plant_has_heat(row, settings)

        # ---------------------------------------------------------------------
        # 7a. Local onsite electricity / heat route
        # ---------------------------------------------------------------------
        if add_local_generation:
            debug_local["local_block_entered"] += 1

            # Electricity generator
            ac_bus = _clean_id(_safe_str(row, "ac_bus_for_model"))
            p_nom_el = _get_electric_p_nom(row, settings)
            potential_el_capacity += float(max(p_nom_el, 0.0))

            if ac_bus is None:
                debug_local["ac_bus_missing"] += 1
                skipped.append((plant_id, plant_name, "electricity_bus_missing", ac_bus))

            elif ac_bus not in network.buses.index:
                debug_local["ac_bus_not_in_network"] += 1
                skipped.append((plant_id, plant_name, "electricity_bus_not_in_network", ac_bus))

            elif p_nom_el <= 0:
                debug_local["p_nom_el_zero"] += 1
                skipped.append((plant_id, plant_name, "electricity_p_nom_zero", p_nom_el))

            else:
                carrier = chp_el_carrier if has_heat else ac_only_carrier
                _add_electricity_generator(
                    network=network,
                    plant_id=plant_id,
                    ac_bus=ac_bus,
                    carrier=carrier,
                    p_nom=p_nom_el,
                    mc=el_mc,
                )
                added_el += 1
                added_el_capacity += float(p_nom_el)

            # Heat generator
            if has_heat:
                debug_local["has_heat_true"] += 1

                heat_bus = _clean_id(_safe_str(row, "heat_bus_for_model"))
                p_nom_heat = _get_heat_p_nom(row, settings)
                potential_heat_capacity += float(max(p_nom_heat, 0.0))

                if heat_bus is None:
                    debug_local["heat_bus_missing"] += 1
                    skipped.append((plant_id, plant_name, "heat_bus_missing", heat_bus))

                elif heat_bus not in network.buses.index:
                    debug_local["heat_bus_not_in_network"] += 1
                    skipped.append((plant_id, plant_name, "heat_bus_not_in_network", heat_bus))

                elif p_nom_heat <= 0:
                    debug_local["p_nom_heat_zero"] += 1
                    skipped.append((plant_id, plant_name, "heat_p_nom_zero", p_nom_heat))

                else:
                    _add_heat_generator(
                        network=network,
                        plant_id=plant_id,
                        heat_bus=heat_bus,
                        carrier=chp_heat_carrier,
                        p_nom=p_nom_heat,
                        mc=heat_mc,
                    )
                    added_heat += 1
                    added_heat_capacity += float(p_nom_heat)

        # ---------------------------------------------------------------------
        # 7b. Gas-grid and/or SWFL-direct CH4 route
        # ---------------------------------------------------------------------
        
        # ---------------------------------------------------------------------
        # 7b. Gas-grid and/or SWFL-direct CH4 route
        # ---------------------------------------------------------------------
        if add_gas_grid_generation or add_swfl_direct_supply:
            p_nom_ch4 = _get_ch4_p_nom(row, settings)
            potential_ch4_capacity += float(max(p_nom_ch4, 0.0))

            if biomethane_price_override is not None:
                biomethane_cost = biomethane_price_override
            else:
                biomethane_cost = _safe_float(
                    row,
                    "biomethane_price_eur_per_mwh_hs_at_96pct",
                    default=default_biomethane_cost,
                )

            if p_nom_ch4 <= 0:
                debug_local["ch4_p_nom_zero"] += 1
                skipped.append((plant_id, plant_name, "ch4_p_nom_zero", p_nom_ch4))
                continue

            # Direct-at-target topology should only be used when no Biogas.SH storage is active
            if (
                add_gas_grid_generation
                and not add_swfl_direct_supply
                and gas_topology == "direct_at_target"
                and not use_biogas_sh_storage
            ):
                _add_ch4_generator_direct(
                    network=network,
                    plant_id=plant_id,
                    ch4_bus=target_ch4_bus,
                    p_nom=p_nom_ch4,
                    mc=biomethane_cost,
                )
                added_ch4 += 1
                added_ch4_capacity += float(p_nom_ch4)
                continue
            
            # Plant-specific CH4 bus + CH4_biogas generator
            plant_bus, _ = _ensure_plant_ch4_bus_and_generator(
                network=network,
                row=row,
                plant_id=plant_id,
                p_nom=p_nom_ch4,
                mc=biomethane_cost,
            )
            added_ch4 += 1
            added_ch4_capacity += float(p_nom_ch4)

        if use_biogas_sh_storage:
            # New topology:
            # plant CH4 bus -> one central Biogas.SH storage bus
            #
            # Downstream routes are already added once above:
            # storage bus -> SWFL
            # storage bus -> public CH4 grid
            _add_ch4_grid_link_from_plant(
                network=network,
                plant_id=plant_id,
                plant_bus=plant_bus,
                target_ch4_bus=storage_bus,
                p_nom=p_nom_ch4,
                settings=settings,
            )

            added_grid_links += 1
            added_grid_link_capacity += float(
            p_nom_ch4 * float(settings.get("ch4_link_p_nom_factor", 1.0))
            )

        else:
            # Original topology:
            # plant CH4 bus -> public CH4 grid
            if add_gas_grid_generation:
                _add_ch4_grid_link_from_plant(
                    network=network,
                    plant_id=plant_id,
                    plant_bus=plant_bus,
                    target_ch4_bus=target_ch4_bus,
                    p_nom=p_nom_ch4,
                    settings=settings,
                )
                added_grid_links += 1
                added_grid_link_capacity += float(
                    p_nom_ch4 * float(settings.get("ch4_link_p_nom_factor", 1.0))
                )

            # Original topology:
            # plant CH4 bus -> SWFL CH4 bus
            if add_swfl_direct_supply:
                swfl_cfg = _swfl_direct_settings(settings)
                _add_swfl_direct_link_from_plant(
                    network=network,
                    plant_id=plant_id,
                    plant_bus=plant_bus,
                    swfl_bus=swfl_bus,
                    p_nom=p_nom_ch4,
                    settings=settings,
                )
                added_swfl_links += 1
                added_swfl_link_capacity += float(
                    p_nom_ch4 * float(swfl_cfg.get("direct_link_p_nom_factor", 1.0))
                )
        
        
    # -------------------------------------------------------------------------
    # 8. Summary output
    # -------------------------------------------------------------------------
    print("Biogas-SH assets added")
    print(f"  scenario_mode:              {route_mode}")
    print(f"  local generation:           {add_local_generation}")
    print(f"  gas-grid injection:         {add_gas_grid_generation}")
    print(f"  SWFL direct supply:         {add_swfl_direct_supply}")
    print(f"  gas_topology:               {gas_topology}")
    print(f"  target_ch4_bus:             {target_ch4_bus}")
    
    if biomethane_price_override is not None:
        print(f"  biomethane price override:  {biomethane_price_override:.2f} €/MWh_Hs")
    else:
        print("  biomethane price source:    CSV plant-specific values")

    print(f"  onsite AC-only carrier:     {ac_only_carrier}")
    print(f"  onsite CHP-el carrier:      {chp_el_carrier}")
    print(f"  onsite CHP-heat carrier:    {chp_heat_carrier}")
    print(f"  gas-grid link carrier:      {ch4_link_carrier}")

    if add_swfl_direct_supply:
        print(f"  swfl_ch4_bus:               {swfl_bus}")
        print(f"  swfl_public_ch4_bus:        {swfl_public_ch4_bus}")
        print(f"  redirected SWFL links:      {len(swfl_redirected_links)}")
        print(f"  SWFL grid supply link:      {swfl_grid_supply_added}")

    print(f"  electricity generators:     {added_el}")
    print(f"  heat generators:            {added_heat}")
    print(f"  CH4_biogas generators:      {added_ch4}")
    print(f"  gas-grid CH4 links:         {added_grid_links}")
    print(f"  SWFL direct links:          {added_swfl_links}")

    print(f"  electricity capacity MW:    {added_el_capacity:.6f}")
    print(f"  heat capacity MW:           {added_heat_capacity:.6f}")
    print(f"  CH4 capacity MW:            {added_ch4_capacity:.6f}")
    print(f"  gas-grid link capacity MW:  {added_grid_link_capacity:.6f}")
    print(f"  SWFL direct capacity MW:    {added_swfl_link_capacity:.6f}")

    print(f"  potential el capacity MW:   {potential_el_capacity:.6f}")
    print(f"  potential heat capacity MW: {potential_heat_capacity:.6f}")
    print(f"  potential CH4 capacity MW:  {potential_ch4_capacity:.6f}")
    
    print(f"  Biogas.SH storage active:   {use_biogas_sh_storage}")
    if use_biogas_sh_storage:
        print(f"  Biogas.SH storage bus:      {storage_bus}")
        print(f"  Biogas.SH storage added:    {storage_added}")
    

    print(f"  skipped components:         {len(skipped)}")

    print("  local debug:")
    for key, value in debug_local.items():
        print(f"    {key}: {value}")

    if skipped and _as_bool(settings.get("print_skipped_components", True), True):
        print("  skipped details:")
        for plant_id, plant_name, kind, value in skipped[:30]:
            print(f"    plant {plant_id} ({plant_name}), {kind}, value={value}")
        if len(skipped) > 30:
            print(f"    ... {len(skipped) - 30} more skipped components")