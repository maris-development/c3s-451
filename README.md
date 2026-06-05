
# C3S Event Attribution Tools

Python library for the [C3S2-451: Operational Extreme Event Monitoring and Attribution Service](https://climate.copernicus.eu/operational-extreme-event-monitoring-and-attribution-service), implemented by MARIS B.V. as part of the [Copernicus Climate Change Service](https://climate.copernicus.eu/).

Tools for analysing and attributing extreme weather events to climate change. Covers statistical analyses for the Interactive Web Application, data processing for the Extreme Event Attribution Office, and counter-factual weather data generation.

## Installation

```bash
pip install c3s-event-attribution-tools
```

Or install directly from source:

```bash
pip install git+https://github.com/maris-development/c3s-451
```

## Modules

| Module | Description |
|---|---|
| `analogues` | Circulation analogue analysis: find historical weather patterns similar to a target event using ERA reanalysis, Euclidean distance metrics, and composite anomaly methods |
| `process` | Data processing functions for preparing and transforming climate datasets |
| `plot` | Plotting and visualisation functions for extreme event analysis |
| `utils` | General utility functions |
| `data.cds_client` | Client for the Copernicus Climate Data Store (CDS) |
| `data.mars_client` | Client for the ECMWF MARS archive |
| `data.beacon_client` | Client for the Beacon data service |
| `data.cordex_client` | Client for CORDEX regional climate model data |
| `data.variable` | Variable definitions and metadata |
| `data.conversions` | Unit and format conversions |

## Usage

Start with the [notebook templates](https://github.com/maris-development/c3s-451/tree/main/notebook_templates), which cover a full attribution workflow:

1. **Event definition** - define the extreme event of interest
2. **Trend analysis** - assess observed trends in the relevant climate variable
3. **Circulation analogues** - identify historical circulation analogues
4. **Model validation & Probabilistic attribution** - validate model data and compute attribution probabilities
5. **Synthesis** - summarise results and produce communication outputs

An interactive [Virtual Research Environment](https://event-attribution.copernicus-climate.eu/jupyterlab) with pre-configured notebooks is available on the project platform.

## Resources

| | |
|---|---|
| Documentation | <https://maris-development.github.io/c3s-451/latest/> |
| PyPI | <https://pypi.org/project/c3s-event-attribution-tools/> |
| Source code | <https://github.com/maris-development/c3s-451> |
| Service platform | <https://event-attribution.copernicus-climate.eu/> |
| C3S project page | <https://climate.copernicus.eu/operational-extreme-event-monitoring-and-attribution-service> |
| Hazard monitoring tool | <https://event-attribution.copernicus-climate.eu/pinto> |

## License

[Apache 2.0](LICENSE.md)

