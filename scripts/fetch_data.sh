#!/usr/bin/env bash
# Download the public raw data into data/ (about 80 MB), then run scripts/prepare_data.py.
set -euo pipefail
cd "$(dirname "$0")/../data"
for y in 2023 2024 2025; do
  curl -fsSL -o "EnergieUebersichtCH-$y.xlsx" "https://www.swissgrid.ch/dam/dataimport/energy-statistic/EnergieUebersichtCH-$y.xlsx"
done
curl -fsSL -o ogd-smn_bus_h_historical_2020-2029.csv \
  https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn/bus/ogd-smn_bus_h_historical_2020-2029.csv
