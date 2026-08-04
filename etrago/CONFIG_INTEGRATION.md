# Biogas.SH / SWFL scenario configuration v2

This package moves frequently changed scenario assumptions out of the large
`args` dictionary while preserving static topology, component names, carriers
and file paths in `appl.py`.

## Files

- `config.yaml` — selections, sensitivity cases and technical assumptions.
- `scenario_config.py` — validation, resolution, updates to `args`, summaries
  and matrix generation.
- `scenario_matrix.csv` — generated factorial sensitivity table.

## Place the files

Copy `config.yaml` and `scenario_config.py` into the same directory as
`appl.py`.

## Integrate into `appl.py`

Add these imports near the other imports:

```python
from pathlib import Path

from scenario_config import (
    load_and_apply_config,
    scenario_summary,
    write_resolved_config,
)
```

After the complete `args` dictionary is defined, but before constructing
`Etrago`, apply the YAML configuration:

```python
CONFIG_PATH = Path(__file__).with_name("config.yaml")

args, resolved_scenario = load_and_apply_config(
    args,
    CONFIG_PATH,
)

print(scenario_summary(resolved_scenario))
```

Then continue normally:

```python
etrago = Etrago(
    args,
    json_path=json_path,
)
```

## Required heat-pump cleanup sequence

The known legacy eGon heat pump must still be removed before electricity
spatial clustering:

```python
etrago.ehv_clustering()

remove_known_legacy_swfl_heat_pump_before_clustering(
    etrago.network,
)

etrago.spatial_clustering()
```

The post-clustering purge must be conditional, because the `none` case has no
planned GWP Links:

```python
future_hp_cfg = (
    args.get("swfl_real_system", {})
    .get("future_heat_pumps", {})
)

if future_hp_cfg.get("active", False):
    purge_legacy_swfl_heat_pumps(
        etrago.network,
        stage="after spatial clustering",
    )
```

After that, continue with gas clustering and temporal preprocessing:

```python
etrago.spatial_clustering_gas()
etrago.snapshot_clustering()
etrago.skip_snapshots()
```

## Save the effective scenario beside the results

After the run has created the CSV export directory:

```python
result_directory = Path(args["csv_export"])

write_resolved_config(
    resolved_scenario,
    result_directory / "resolved_config.yaml",
)
```

Skip this call when `args["csv_export"]` is false.

## Normal use

Normally only edit:

```yaml
selection:
  natural_gas_price_case: "upstream_only"
  biomethane_price_case: "low_25"
  heat_pump_case: "two"
  swfl_unit_case: "all_operational"
  biomethane_use_case: "k12_k13_only"
  biogas_route_case: "hybrid"
```

Available heat-pump cases:

```text
none
one_gwp1
one_gwp2
two
```

Available SWFL unit cases:

```text
all_operational
gas_units_only
k12_k13_plus_gas_to_power
k12_k13_only
electric_heat_only
```

Available storage-aware Biogas.SH route cases:

```text
onsite
storage_to_grid
storage_to_swfl
hybrid
```

## Validate and inspect

```bash
python scenario_config.py config.yaml validate
python scenario_config.py config.yaml show
```

## Generate the factorial matrix

```bash
python scenario_config.py config.yaml matrix \
  --output scenario_matrix.csv
```

The supplied matrix contains:

- 3 natural-gas price cases
- 3 biomethane price cases
- 2 heat-pump cases
- 2 SWFL unit cases
- 1 biomethane-use case
- 1 Biogas.SH route case

This produces 36 scenarios.

## Environment-variable overrides

```bash
ETRAGO_NATURAL_GAS_PRICE_CASE=adder_50 \
ETRAGO_BIOMETHANE_PRICE_CASE=cost_based_75 \
ETRAGO_HEAT_PUMP_CASE=two \
ETRAGO_SWFL_UNIT_CASE=k12_k13_plus_gas_to_power \
ETRAGO_BIOMETHANE_USE_CASE=k12_k13_only \
ETRAGO_BIOGAS_ROUTE_CASE=hybrid \
python appl.py
```

Supported variables:

```text
ETRAGO_NATURAL_GAS_PRICE_CASE
ETRAGO_BIOMETHANE_PRICE_CASE
ETRAGO_HEAT_PUMP_CASE
ETRAGO_SWFL_UNIT_CASE
ETRAGO_BIOMETHANE_USE_CASE
ETRAGO_BIOGAS_ROUTE_CASE
```

## Important mappings

The loader now reads the current top-level structure:

```python
args["swfl_real_system"]
args["biogas_sh"]
args["extra_functionality"]["biogas_sh_resource"]
```

`run.ac_clusters` updates only:

```python
args["network_clustering_ehv"]["cluster"]["n_clusters"]
```

It does not overwrite the complete clustering dictionary.

The result-name template updates `args["csv_export"]` using the effective
scenario name, represented hours and AC cluster count.

## Gas-price interpretation

`swfl_import_adder_eur_per_mwh_hs` is applied to:

```python
args["biogas_sh"]["swfl_direct"]["grid_supply_marginal_cost"]
```

It is an adder on the public-grid-to-SWFL natural-gas route. Use
`upstream_only` when the public CH4 network already includes the natural-gas
commodity cost.

## Route-name interpretation

The model keeps the existing internal booleans for compatibility, but the
user-facing route names describe the actual storage-aware topology:

- `storage_to_grid`: plant buses → central storage → public CH4 grid
- `storage_to_swfl`: plant buses → central storage → SWFL biomethane bus
- `hybrid`: onsite options plus both storage output routes
