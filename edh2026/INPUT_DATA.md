# Input data for the challenge

The input data can be found in `edh.input`. It includes historical data of forecasts and realizations. The forecasts are given until 27 August 2026 and realizations until 20 August 2026. Some tables might have missing data.

## Generation forecast
18 Tables: ``generation_forecast_<country>_<energy_source_type>``

For each country (CH, DE, FR, IT, AT), there is the historical data of the generated energy. The time buckets are given in MTU (Market Time Unit), which is the smallest time bucket on which European electricity markets clear and settle, every bid, activation, price and imbalance. Across the EU the MTU has been 15 minutes since 1 October 2025, reduced from one hour. The time is in UTC.

The data includes the day-ahead forecast as well as realization (actual) for solar, wind (onshore and offshore, if present) and total generation. The units are Megawatts (MW).

## Net positions
Table: ``net_positions``

The net position of a country is the netted sum of all its electricity exports and imports across all borders for a specific market time unit. A positive net position means the country is a net exporter, while a negative net position indicates it is a net importer. The time is local time.

## Net transfer capacity
Table: ``ntc_month``

An estimation of the maximum amount of energy Switzerland could sell on the month-ahead market, taking into account outage situations to ensure the grid remains stable and compliant with safety standards on both sides of the border. The time is local time.

## Cross border exchanges
Table: ``cross_border_exchanges``

The commercial flows between countries. The flow direction is given by the column header: if e.g. the header is CH-IT, positive values mean a flow from CH to IT and negative values from IT to CH. The time is local time.
