# Biogas.SH scenario configuration v3

This package moves frequently changed scenario assumptions out of the large
`args` dictionary while preserving static topology, component names, carriers,
and file paths in `appl.py`.

The current version supports fossil-gas and biomethane price sensitivities,
planned heat-pump cases, SWFL unit availability, biomethane eligibility,
storage-aware Biogas.SH routes, and optional run settings.

## Files

- `config.yaml` — selections, price cases, technical assumptions, and run
  settings.
- `scenario_config.py` — validation, scenario resolution, updates to `args`,
  network-price application, summaries, and matrix generation.
- `scenario_matrix.csv` — generated factorial sensitivity table.
- `resolved_config.yaml` — effective configuration written beside a completed
  result export.

## Place the files

Copy `config.yaml` and `scenario_config.py` into the same directory as
`appl.py`.

## Integrate into `appl.py`

Add these imports near the other imports:

```python
from pathlib import Path

from scenario_config import (
    apply_network_price_scenario,
    load_and_apply_config,
    scenario_summary,
    write_resolved_config,
)
```

After the complete `args` dictionary is defined, but before constructing
`Etrago`, load and apply the YAML configuration:

```python
CONFIG_PATH = Path(__file__).resolve().with_name("config.yaml")

args, resolved_scenario = load_and_apply_config(
    args,
    CONFIG_PATH,
)

print(scenario_summary(resolved_scenario))
```

Then construct the model and apply the selected fossil-gas and biomethane
prices before any electricity or gas clustering:

```python
etrago = Etrago(
    args,
    json_path=json_path,
)

apply_network_price_scenario(
    etrago.network,
    resolved_scenario,
)
```

At this point, `adjust_CH4_gen_carriers()` and the creation of the custom
Biogas.SH assets must already be complete. A successful hybrid run reports both
the updated `CH4_NG` generators and 21 updated custom biomethane generators.

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

The post-clustering purge must be conditional because the `none` case has no
planned GWP links:

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
if args.get("csv_export"):
    result_directory = Path(args["csv_export"])

    write_resolved_config(
        resolved_scenario,
        result_directory / "resolved_config.yaml",
    )
```

The resolved file records the effective selection, prices, run settings, and
output directory used by that result.

## Normal use

Normally, edit only `selection` and, when needed, `run`:

```yaml
selection:
  fossil_gas_price_case: "legacy_egon"
  biomethane_price_case: "low_25"
  heat_pump_case: "two"
  swfl_unit_case: "all_operational"
  biomethane_use_case: "k12_k13_only"
  biogas_route_case: "hybrid"

run:
  start_snapshot: 1
  end_snapshot: 24
  ac_clusters: 50
  result_name_template: "{scenario}_{hours}h_{ac_clusters}ac"
```

Use `null` for a run value when the corresponding value already defined in
`appl.py` should be preserved.

Available fossil-gas cases:

```text
low_2035
legacy_egon
high_2035
crisis_2035
```

Available biomethane-price cases:

```text
low_25
medium_50
cost_based_75
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

Run both checks before starting eTraGo:

```bash
python scenario_config.py config.yaml validate
python scenario_config.py config.yaml show
```

For the baseline 24-hour test, `show` should report:

```text
Fossil-gas case: legacy_egon
Final CH4_NG marginal cost: 40.9765 EUR/MWh_fuel
Start snapshot: 1
End snapshot: 24
Represented hours: 24
AC clusters: 50
```

## Verified 24-hour test

For a genuine consecutive 24-hour test, keep snapshot clustering disabled and
set the following in `appl.py`:

```python
"skip_snapshots": False,
```

The following command has been tested successfully:

```bash
python -u appl.py 2>&1 | tee \
  "scenario_run_$(date +%Y%m%d_%H%M%S).log"
```

The verified selection was:

```text
legacy_egon + low_25 + two + all_operational
+ k12_k13_only + hybrid + 24 hours + 50 AC clusters
```

It produced the following result directory under `etrago`:

```text
legacy_egon__low_25__two__all_operational__k12_k13_only__hybrid_24h_50ac
```

This confirms the end-to-end loading of the YAML selection, scenario-based
result naming, model execution, and CSV export for the 24-hour/50-cluster test.

## Generate the factorial matrix

```bash
python scenario_config.py config.yaml matrix \
  --output scenario_matrix.csv
```

With all four fossil-gas cases enabled, the supplied matrix contains:

- 4 fossil-gas price cases
- 3 biomethane price cases
- 2 heat-pump cases
- 2 SWFL unit cases
- 1 biomethane-use case
- 1 Biogas.SH route case

This produces 48 scenario rows. Matrix generation writes the combinations to
CSV; it does not execute eTraGo runs.

## Environment-variable overrides

```bash
ETRAGO_FOSSIL_GAS_PRICE_CASE=high_2035 \
ETRAGO_BIOMETHANE_PRICE_CASE=cost_based_75 \
ETRAGO_HEAT_PUMP_CASE=two \
ETRAGO_SWFL_UNIT_CASE=k12_k13_plus_gas_to_power \
ETRAGO_BIOMETHANE_USE_CASE=k12_k13_only \
ETRAGO_BIOGAS_ROUTE_CASE=hybrid \
python -u appl.py
```

Supported variables:

```text
ETRAGO_FOSSIL_GAS_PRICE_CASE
ETRAGO_BIOMETHANE_PRICE_CASE
ETRAGO_HEAT_PUMP_CASE
ETRAGO_SWFL_UNIT_CASE
ETRAGO_BIOMETHANE_USE_CASE
ETRAGO_BIOGAS_ROUTE_CASE
```

## Important mappings

The loader reads and updates the current top-level structure:

```python
args["swfl_real_system"]
args["biogas_sh"]
args["extra_functionality"]["biogas_sh_resource"]
```

`run.ac_clusters` updates only:

```python
args["network_clustering"]["electricity_grid"]["n_clusters"]
```

It does not overwrite the complete clustering dictionary. The result-name
template updates `args["csv_export"]` using the effective scenario name,
represented hours, and AC cluster count.

The current `scenario_config.py` also requires the onsite price section in
`config.yaml`, even when the selected route is not `onsite`:

```yaml
price_cases:
  onsite:
    selected_electricity_case: "full_eeg"
    electricity_marginal_cost_eur_per_mwh:
      full_eeg: 102.4
      half_eeg: 146.0
      no_eeg: 189.5
    heat_marginal_cost_eur_per_mwh: 0.0
```

## Gas-price interpretation

The selected fossil-gas marginal cost is calculated as:

```text
CH4_NG marginal cost
= gas commodity price
+ CO2 price × 0.201 tCO2/MWh_fuel
```

The four current values are:

| Case | Final `CH4_NG` cost [EUR/MWh_fuel] |
| --- | ---: |
| `low_2035` | 23.0500 |
| `legacy_egon` | 40.9765 |
| `high_2035` | 63.0690 |
| `crisis_2035` | 88.9440 |

`apply_network_price_scenario()` assigns the selected value to the original
`CH4_NG` generators and assigns the selected biomethane cost to the 21 custom
Biogas.SH `CH4_biogas` generators.

Keep the additional SWFL import charge at zero unless a separate transport or
network charge is intentionally modelled:

```yaml
swfl_import_adder_eur_per_mwh_fuel: 0.0
```

The downstream SWFL gas-to-power, boiler, reserve-boiler, and routing-link
marginal costs should not repeat the gas commodity or CO2 cost. Otherwise, fuel
costs would be counted twice.

## Route-name interpretation

The model keeps the existing internal booleans for compatibility, but the
user-facing route names describe the actual storage-aware topology:

- `onsite`: onsite electricity and heat options only
- `storage_to_grid`: plant buses → central storage → public CH4 grid
- `storage_to_swfl`: plant buses → central storage → SWFL biomethane bus
- `hybrid`: onsite options plus both storage output routes
