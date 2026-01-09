import subprocess
import pandas as pd
import xarray as xr
import geopandas as gpd
import beacon_api
from datetime import datetime, timedelta
from cdsapi import Client
import tempfile
import numpy as np

# doesnt work on windows
try:
    import iris
except ImportError:
    print("cant import iris, are you on windows?")

try:
    # Available in Python 3.12+ as per PEP 702
    from warnings import deprecated
except ImportError:
    # Backport available in typing_extensions for older versions
    from typing_extensions import deprecated


class DataClient():
    """
    DataClient is a class that provides methods to fetch and process climate data
    from various sources including the Climate Data Store (CDS), Beacon Cache, and MARS.
    
    Parameters:
        cds_key (str): The API key for the Climate Data Store (CDS).
        beacon_cache_url (str | None): Optional URL for the Beacon Cache service.
        beacon_token (str | None): Optional token for accessing the Beacon Cache.
        mars_key (str | None): Optional API key for accessing MARS data.
        
    
    """
    def __init__(self, cds_key: str, beacon_cache_url: str | None = None, beacon_token: str | None = None, mars_key: str | None = None) -> None:
        self.cds_client = CDSClient(cds_key)
        self.beacon_cache = None
        self.mars_client = None
        if beacon_cache_url:
            self.beacon_cache = BeaconDataClient(beacon_cache_url=beacon_cache_url, beacon_token=beacon_token)
        if mars_key:
            self.mars_client = MarsClient(key=mars_key)

    def _bbox_to_0_360(self, bbox: tuple[float,float,float,float], eps=1e-9) -> list[tuple[float,float,float,float]]:
        """
        Convert a bounding box from [-180, 180] longitude to [0, 360] longitude.

        If the converted box crosses 0°, return two wrapped bounding boxes.

        Parameters:
            bbox (tuple):
                (min_lon, min_lat, max_lon, max_lat) with longitude in [-180, 180].
            eps (float):
                Small tolerance value.
            as_shapely (bool):
                If True, return shapely boxes (requires shapely).

        Returns:
            list: List of bounding boxes in
            (min_lon_360, min_lat, max_lon_360, max_lat) format or shapely boxes.
        """
        min_lon, min_lat, max_lon, max_lat = bbox

        # Handle antimeridian-style input like (170, ..., -170, ...) meaning it crosses -180/180 already
        if min_lon > max_lon:
            max_lon += 360.0

        # Full-world guard
        if (max_lon - min_lon) >= 360.0 - eps:
            out = [(0.0, min_lat, 360.0, max_lat)]
        else:
            to360 = lambda x: x % 360.0
            lo = to360(min_lon)
            hi = to360(max_lon)

            if lo <= hi:  # no wrap in 0–360 space
                out = [(lo, min_lat, hi, max_lat)]
            else:
                # wraps across 0° in 0–360; split into two
                out = [
                    (lo,  min_lat, 360.0, max_lat),
                    (0.0, min_lat, hi,    max_lat),
                ]
        return out

    def _convert_temp(self, df: gpd.GeoDataFrame, from_unit="k", to_unit="c") -> gpd.GeoDataFrame:
        """
        Convert temperature values in the 't2m' column from one unit to another.

        This method performs in-place temperature conversion within a GeoDataFrame and
        supports Kelvin (k), Celsius (c), and Fahrenheit (f).

        Parameters:
            df (gpd.GeoDataFrame):
                GeoDataFrame containing a 't2m' column with temperature values.
            from_unit (str, optional):
                Source temperature unit ('k', 'c', or 'f'). Default is 'k'.
            to_unit (str, optional):
                Target temperature unit ('k', 'c', or 'f'). Default is 'c'.

        Returns:
            gpd.GeoDataFrame: The GeoDataFrame with the 't2m' column converted to the
            target unit.

        Raises:
            ValueError: If from_unit is not one of 'k', 'c', or 'f'.

        Notes:
            - If from_unit and to_unit are identical, no conversion is performed.
            - Conversion modifies the 't2m' column in place.
            - Conversion formulas:
                - Kelvin to Celsius: C = K - 273.15
                - Kelvin to Fahrenheit: F = (K - 273.15) × 9/5 + 32
                - Celsius to Kelvin: K = C + 273.15
                - Celsius to Fahrenheit: F = C × 9/5 + 32
                - Fahrenheit to Kelvin: K = (F - 32) × 5/9 + 273.15
                - Fahrenheit to Celsius: C = (F - 32) × 5/9
        """

        if from_unit not in ["k", "c", "f"]:
            raise ValueError(f"Invalid from_unit: {from_unit}. Must be 'k', 'c', or 'f'.")
            return df

        if from_unit == "k" and to_unit == "k":
            return df
        if from_unit == "c" and to_unit == "c":
            return df
        if from_unit == "f" and to_unit == "f":
            return df

        if from_unit == "k" and to_unit == "c":
            df['t2m'] = df['t2m'] - 273.15
        elif from_unit == "k" and to_unit == "f":
            df['t2m'] = (df['t2m'] - 273.15) * 9/5 + 32
        elif from_unit == "c" and to_unit == "k":
            df['t2m'] = df['t2m'] + 273.15
        elif from_unit == "c" and to_unit == "f":
            df['t2m'] = (df['t2m'] * 9/5) + 32
        elif from_unit == "f" and to_unit == "k":
            df['t2m'] = (df['t2m'] - 32) * 5/9 + 273.15
        elif from_unit == "f" and to_unit == "c":
            df['t2m'] = (df['t2m'] - 32) * 5/9

        return df

    def _convert_precipitation(self, df: gpd.GeoDataFrame, from_unit, to_unit) -> gpd.GeoDataFrame:
        """
        Convert precipitation values in the 'tp' column between supported units.

        Parameters:
            df (gpd.GeoDataFrame):
                GeoDataFrame containing a 'tp' column with precipitation values.
            from_unit (str):
                Source unit. Must be 'm' or 'm/h'.
            to_unit (str):
                Target unit. Supported target unit is 'mm'.

        Returns:
            gpd.GeoDataFrame: The GeoDataFrame with converted precipitation values.

        Raises:
            ValueError: If from_unit is not 'm' or 'm/h'.
        """

        if from_unit not in ["m", "m/h"]:
            raise ValueError(f"Invalid from_unit: {from_unit}. Must be 'm' or 'm/h'.")
            return df

        if from_unit == "m/h" and to_unit == "mm":
            df['tp'] = df['tp'] * 24000
            print("Converted from m/h to mm.")
        elif from_unit == "m" and to_unit == "mm":
            df['tp'] = df['tp'] * 1000
            print("Converted from m to mm.")

        return df

    def last_day_of_month(self, dt: datetime) -> datetime:
        """
        Return the last day of the month for the given datetime.

        Parameters:
            dt (datetime):
                The input datetime.

        Returns:
            datetime: The datetime corresponding to the last day of the same month.
        """
        next_month = dt.replace(day=28) + timedelta(days=4)  # always moves to the next month
        return next_month.replace(day=1) - timedelta(days=1)    # always returns back to the current month

    def get_beacon_cache(self, gdfs:list[gpd.GeoDataFrame],
                         bbox:tuple[float, float, float, float],
                         time_range:tuple[datetime, datetime],
                         var:str, 
                         var_to:str, 
                         table:str, 
                         months:list[int]|None=None,
                         min_retrieved_time=None, 
                         max_retrieved_time=None):
        """
        Attempts to retrieve data from the beacon cache for the specified parameters.

        This method fetches ERA5 daily data from a zarr-based beacon cache and handles
        bounding boxes that cross the antimeridian by converting longitudes to the
        0–360 degree range. It also updates the minimum and maximum retrieved timestamps
        based on accumulated results.

        Parameters:
            gdfs (list[gpd.GeoDataFrame]):
                List to which retrieved GeoDataFrames will be appended.
            bbox (tuple[float, float, float, float]):
                Bounding box as (min_lon, min_lat, max_lon, max_lat).
            time_range (tuple[datetime, datetime]):
                Start and end datetime for data retrieval.
            var (str):
                Source variable name to retrieve from the cache.
            var_to (str):
                Target variable name to rename the retrieved variable to.
            table (str):
                Table identifier used for the beacon cache query.
            months (list[int] | None, optional):
                List of months to filter by. Default is None.
            min_retrieved_time (datetime | None, optional):
                Minimum previously retrieved timestamp. Default is None.
            max_retrieved_time (datetime | None, optional):
                Maximum previously retrieved timestamp. Default is None.

        Returns:
            tuple: A tuple containing:
                - list[gpd.GeoDataFrame]: The updated list with appended GeoDataFrames.
                - datetime: The minimum valid_time across all retrieved data.
                - datetime: The maximum valid_time across all retrieved data.

        Note:
            - Retrieved data is sorted by valid_time, longitude, and latitude.
            - The source variable is renamed to the target variable name.
            - Antimeridian crossing is handled by converting the bounding box to 0–360° longitude.
        """

        print(f"Fetching data from beacon cache for range: {time_range[0]} - {time_range[1]}")

        adjusted_bboxes = self._bbox_to_0_360(bbox)

        for beacon_bbox in adjusted_bboxes:
            # print("Beacon Bbox: "+  str(beacon_bbox))
            gdf = self.beacon_cache._fetch_from_era5_daily_zarr(bbox=beacon_bbox, time_range=time_range, variable=var, months=months, table=table)
            if not gdf.empty:
                gdf = gdf.sort_values(['valid_time', 'longitude', 'latitude']).reset_index(drop=True)
                # Get the min and max valid time from the DF and validate it covers the requested time range or else fill with era5 cds request
                min_retrieved_time = min(gdf['valid_time'].min(), min_retrieved_time) if min_retrieved_time is not None else gdf['valid_time'].min()
                max_retrieved_time = max(gdf['valid_time'].max(), max_retrieved_time) if max_retrieved_time is not None else gdf['valid_time'].max()

                # Rename t2m_min to t2m for consistency
                gdf = gdf.rename(columns={var: var_to})

                gdfs.append(gdf)

        return gdfs, min_retrieved_time, max_retrieved_time

    def get_cds_data(self, gdfs:list[gpd.GeoDataFrame],
                     bbox:tuple[float, float, float, float],
                     time_range:tuple[datetime, datetime],
                     var:str, 
                     statistic:str, 
                     months:list[int]|None=None,
                     min_retrieved_time:datetime|None=None, 
                     max_retrieved_time:datetime|None=None
            ):
        """
        Fetch climate data from the CDS for a specified time range and location.

        This method retrieves ERA5 single-level daily statistics from the CDS API and
        appends the results to a list of GeoDataFrames. It also tracks the minimum and
        maximum retrieved timestamps across multiple calls.

        Parameters:
            gdfs (list[gpd.GeoDataFrame]):
                List to which newly fetched GeoDataFrames will be appended.
            bbox (tuple[float, float, float, float]):
                Bounding box as (west, south, east, north).
            time_range (tuple[datetime, datetime]):
                Start and end datetime for the data request.
            var (str):
                The daily statistic variable to retrieve (e.g., 'maximum_2m_temperature_in_last_24h').
            statistic (str):
                Statistical measure to apply to the variable.
            months (list[int] | None, optional):
                List of months (1–12) to filter by. Defaults to None (all months).
            min_retrieved_time (datetime | None, optional):
                Current minimum timestamp from previous retrievals. Defaults to None.
            max_retrieved_time (datetime | None, optional):
                Current maximum timestamp from previous retrievals. Defaults to None.

        Returns:
            tuple: A tuple containing:
                - gdfs (list[gpd.GeoDataFrame]): Updated list with newly fetched GeoDataFrames appended.
                - min_retrieved_time (datetime): Minimum valid_time across all retrieved data.
                - max_retrieved_time (datetime): Maximum valid_time across all retrieved data.

        Side Effects:
            - Prints a message indicating the time range being fetched from CDS.
            - Modifies the input gdfs list by appending new data.
        """

        print(f"Fetching missing data from CDS for range: {time_range[0]} - {time_range[1]}")

        gdf = self.cds_client._fetch_data_single_levels("derived-era5-single-levels-daily-statistics", [statistic], bbox, time_range=time_range, daily_statistic=var, months=months)
        min_retrieved_time = gdf['valid_time'].min() if min_retrieved_time is None else min(min_retrieved_time, gdf['valid_time'].min())
        max_retrieved_time = gdf['valid_time'].max() if max_retrieved_time is None else max(max_retrieved_time, gdf['valid_time'].max())

        gdfs.append(gdf)

        return gdfs, min_retrieved_time, max_retrieved_time

    def get_mars_data(
        self,
        gdfs: list[gpd.GeoDataFrame],
        bbox: tuple[float, float, float, float],
        time_range: tuple[datetime, datetime],
        var: str,
        min_retrieved_time: datetime | None = None,
        max_retrieved_time: datetime | None = None,
    ):
        """
        Fetch operational data from MARS for a specified variable and time range.

        This method retrieves missing meteorological data from the MARS Operational
        archive to fill gaps in a time series. Supported variables include temperature
        (min, max, mean) and total precipitation. It also tracks the minimum and maximum
        retrieved timestamps across multiple calls.

        Parameters:
            gdfs (list[gpd.GeoDataFrame]):
                List to which fetched GeoDataFrames will be appended.
            bbox (tuple[float, float, float, float]):
                Bounding box as (min_lon, min_lat, max_lon, max_lat).
            time_range (tuple[datetime, datetime]):
                Start and end datetime for the data request.
            var (str):
                Variable to fetch. Supported values: 't2m_min', 't2m_max', 't2m_mean', 'tp'.
            min_retrieved_time (datetime | None, optional):
                Minimum datetime already retrieved. Defaults to None.
            max_retrieved_time (datetime | None, optional):
                Maximum datetime already retrieved; used as start date for fetching.
                Defaults to None, which uses time_range[0].

        Returns:
            tuple[list[gpd.GeoDataFrame], datetime | None, datetime | None]:
                - Updated list of GeoDataFrames with newly fetched data appended.
                - Updated minimum retrieved time across all fetches.
                - Updated maximum retrieved time across all fetches.

        Note:
            - If mars_client is not configured, a warning is printed and inputs are returned unchanged.
            - min_retrieved_time and max_retrieved_time are updated based on the 'valid_time' column of fetched data.
        """

        print(f"Fetching missing data at the end of the time range from MARS Operational for range: {time_range[0]} - {time_range[1]}")

        if self.mars_client is not None:

            gdf = None

            # this looks bad but is currently how the api works
            if var == 't2m_min':
                gdf = self.mars_client.fetch_t2m_min_operational_data(min_date=max_retrieved_time if max_retrieved_time is not None else time_range[0],
                                                                        max_date=time_range[1], min_lon=bbox[0], max_lon=bbox[2], min_lat=bbox[1], max_lat=bbox[3])
            if var == 't2m_max':
                gdf = self.mars_client.fetch_t2m_max_operational_data(min_date=max_retrieved_time if max_retrieved_time is not None else time_range[0],
                                                                        max_date=time_range[1], min_lon=bbox[0], max_lon=bbox[2], min_lat=bbox[1], max_lat=bbox[3])
            if var == 't2m_mean':
                gdf = self.mars_client.fetch_t2m_mean_operational_data(min_date=max_retrieved_time if max_retrieved_time is not None else time_range[0],
                                                                        max_date=time_range[1], min_lon=bbox[0], max_lon=bbox[2], min_lat=bbox[1], max_lat=bbox[3])
            if var == 'tp':
                gdf = self.mars_client.fetch_total_precipitation_operational_data(min_date=max_retrieved_time if max_retrieved_time is not None else time_range[0],
                                                                        max_date=time_range[1], min_lon=bbox[0], max_lon=bbox[2], min_lat=bbox[1], max_lat=bbox[3])

            # Update min and max retrieved time
            min_retrieved_time = gdf['valid_time'].min() if min_retrieved_time is None else min(min_retrieved_time, gdf['valid_time'].min())
            max_retrieved_time = gdf['valid_time'].max() if max_retrieved_time is None else max(max_retrieved_time, gdf['valid_time'].max())

            gdfs.append(gdf)
        else:
            print("MARS client not configured, cannot fetch missing data.")

        return gdfs, min_retrieved_time, max_retrieved_time

    def temperature_2m_min(
        self,
        bbox: tuple[float, float, float, float],
        time_range: tuple[datetime, datetime],
        months: list[str] | list[int] | None = None,
        from_unit: str = "k",
        to_unit: str = "c",
    ) -> gpd.GeoDataFrame:
        """
        Fetch minimum 2-meter temperature data for a specified bounding box and time range.

        This method retrieves data from multiple sources in the following order:
        1. Beacon cache (if available)
        2. CDS API (for missing data)
        3. MARS data (for remaining gaps)
        4. MARS Forecast data 

        Parameters:
            bbox (tuple[float, float, float, float]):
                Bounding box as (min_longitude, min_latitude, max_longitude, max_latitude).
            time_range (tuple[datetime, datetime]):
                Tuple of (start_time, end_time) as datetime objects, inclusive.
            months (list[str] | list[int] | None, optional):
                Filter by month names or numbers (1–12). Defaults to None (all months).
            from_unit (str, optional):
                Source temperature unit. Default is "k" (Kelvin).
            to_unit (str, optional):
                Target temperature unit. Default is "c" (Celsius).

        Returns:
            gpd.GeoDataFrame: GeoDataFrame containing minimum 2-meter temperature data
            in EPSG:4326, with values converted to the specified unit.

        Notes:
            - Data gaps are handled automatically by fetching from multiple sources.
            - A warning is printed if data is still missing after all attempts.
            - MARS Forecast data retrieval is not fully implemented yet.
        """

        # Implementation will go here
        min_retrieved_time = None
        max_retrieved_time = None
        gdfs = []

        # Fetch the data from the beacon client if has been defined. Then check the min and max date and compare. Whatever is missing should be requested via the cds api
        if self.beacon_cache:
            gdfs, min_retrieved_time, max_retrieved_time = self.get_beacon_cache(gdfs, bbox, time_range, months=months, var='t2m_min', var_to='t2m', table='daily')

        if min_retrieved_time is None or max_retrieved_time is None or min_retrieved_time > time_range[0] or max_retrieved_time < time_range[1]:
            # No valid data found in beacon cache or we are missing data for the requested time range
            print("Missing data in beacon cache, fetching missing data from CDS...")

            if min_retrieved_time is None or min_retrieved_time > time_range[0]:
                # Request missing data from CDS
                fetch_start = time_range[0]
                fetch_end = min_retrieved_time if min_retrieved_time is not None else time_range[1]
                gdfs, min_retrieved_time, max_retrieved_time = self.get_cds_data(gdfs, bbox, (fetch_start, fetch_end), months=months, var='daily_min', statistic='2m_temperature',
                                                                                 min_retrieved_time=min_retrieved_time, max_retrieved_time=max_retrieved_time)

            if max_retrieved_time is None or max_retrieved_time < time_range[1]:
                # Request missing data from CDS
                fetch_start = max_retrieved_time if max_retrieved_time is not None else time_range[0]
                fetch_end = time_range[1]
                gdfs, min_retrieved_time, max_retrieved_time = self.get_cds_data(gdfs, bbox, (fetch_start, fetch_end), months=months, var='daily_min', statistic='2m_temperature',
                                                                                 min_retrieved_time=min_retrieved_time, max_retrieved_time=max_retrieved_time)

        if max_retrieved_time is None or max_retrieved_time < time_range[1]:
            # We are still missing data at the end of the range, try to fetch from MARS if available
            gdfs, min_retrieved_time, max_retrieved_time = self.get_mars_data(gdfs, bbox, time_range, var='t2m_min', min_retrieved_time=min_retrieved_time, max_retrieved_time=max_retrieved_time)

        if max_retrieved_time is None or max_retrieved_time < time_range[1]:
            # TODO: implement MARS Forecast data fetching or remove if block
            print("Still missing data. Fetching from MARS Forecast data...")
            if self.mars_client is not None:
                # Write warning that it is not implemented yet
                print("MARS Forecast data fetching not implemented yet.")

        # Concatenate all GeoDataFrames
        final_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs='EPSG:4326')

        df = self._convert_temp(final_gdf, from_unit, to_unit)

        return df

    def temperature_2m_max(
        self,
        bbox: tuple[float, float, float, float],
        time_range: tuple[datetime, datetime],
        months: list[str] | list[int] | None = None,
        from_unit: str = "k",
        to_unit: str = "c",
    ) -> gpd.GeoDataFrame:
        """
        Fetch maximum 2-meter temperature data for a specified bounding box and time range.

        This method retrieves data from multiple sources in the following order:
        1. Beacon cache (if available)
        2. CDS API (for missing data)
        3. MARS data (for remaining gaps)
        4. MARS Forecast data 

        Parameters:
            bbox (tuple[float, float, float, float]):
                Bounding box as (min_longitude, min_latitude, max_longitude, max_latitude).
            time_range (tuple[datetime, datetime]):
                Tuple of (start_time, end_time) as datetime objects, inclusive.
            months (list[str] | list[int] | None, optional):
                Filter by month names or numbers (1–12). Defaults to None (all months).
            from_unit (str, optional):
                Source temperature unit. Default is "k" (Kelvin).
            to_unit (str, optional):
                Target temperature unit. Default is "c" (Celsius).

        Returns:
            gpd.GeoDataFrame: GeoDataFrame containing maximum 2-meter temperature data
            in EPSG:4326, with values converted to the specified unit.

        Notes:
            - Data gaps are handled automatically by fetching from multiple sources.
            - A warning is printed if data is still missing after all attempts.
            - MARS Forecast data retrieval is not fully implemented yet.
        """

        # Implementation will go here
        min_retrieved_time = None
        max_retrieved_time = None
        gdfs = []

        # Fetch the data from the beacon client if has been defined. Then check the min and max date and compare. Whatever is missing should be requested via the cds api
        if self.beacon_cache:
            gdfs, min_retrieved_time, max_retrieved_time = self.get_beacon_cache(gdfs, bbox, time_range, months=months, var='t2m_max', var_to='t2m', table='daily')

        if min_retrieved_time is None or max_retrieved_time is None or min_retrieved_time > time_range[0] or max_retrieved_time < time_range[1]:
            # No valid data found in beacon cache or we are missing data for the requested time range
            print("Missing data in beacon cache, fetching missing data from CDS...")

            if min_retrieved_time is None or min_retrieved_time > time_range[0]:
                # Request missing data from CDS
                fetch_start = time_range[0]
                fetch_end = min_retrieved_time if min_retrieved_time is not None else time_range[1]
                gdfs, min_retrieved_time, max_retrieved_time = self.get_cds_data(gdfs, bbox, (fetch_start, fetch_end), months=months, var='daily_max', statistic='2m_temperature',
                                                                                 min_retrieved_time=min_retrieved_time, max_retrieved_time=max_retrieved_time)

            if max_retrieved_time is None or max_retrieved_time < time_range[1]:
                # Request missing data from CDS
                fetch_start = max_retrieved_time if max_retrieved_time is not None else time_range[0]
                fetch_end = time_range[1]
                gdfs, min_retrieved_time, max_retrieved_time = self.get_cds_data(gdfs, bbox, (fetch_start, fetch_end), months=months, var='daily_max', statistic='2m_temperature',
                                                                                 min_retrieved_time=min_retrieved_time, max_retrieved_time=max_retrieved_time)

        if max_retrieved_time is None or max_retrieved_time < time_range[1]:
            # We are still missing data at the end of the range, try to fetch from MARS if available
            gdfs, min_retrieved_time, max_retrieved_time = self.get_mars_data(gdfs, bbox, time_range, var='t2m_max', min_retrieved_time=min_retrieved_time, max_retrieved_time=max_retrieved_time)

        if max_retrieved_time is None or max_retrieved_time < time_range[1]:
            print("Still missing data. Fetching from MARS Forecast data...")
            if self.mars_client is not None:
                # Write warning that it is not implemented yet
                print("MARS Forecast data fetching not implemented yet.")

        # Concatenate all GeoDataFrames
        final_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs='EPSG:4326')

        df = self._convert_temp(final_gdf, from_unit, to_unit)

        return df

    def mean_sea_level_pressure(self, bbox: tuple[float,float,float,float], time_range: tuple[datetime,datetime]) -> gpd.GeoDataFrame:
        """
        Fetch mean sea level pressure data for a specified bounding box and time range.

        This method retrieves daily mean sea level pressure statistics from the ERA5
        single levels daily statistics dataset via the CDS client.

        Parameters:
            bbox (tuple[float, float, float, float]):
                Bounding box coordinates as (min_longitude, min_latitude, max_longitude, max_latitude).
            time_range (tuple[datetime, datetime]):
                Time range as (start_date, end_date) defining the period for which to fetch data.

        Returns:
            gpd.GeoDataFrame: A GeoDataFrame containing mean sea level pressure data with
            spatial and temporal information.
        """
        return self.cds_client._fetch_data_single_levels("derived-era5-single-levels-daily-statistics", ['mean_sea_level_pressure'], bbox, time_range, daily_statistic="daily_mean")

    def z500_geopotential_mean(self, bbox: tuple[float,float,float,float], time_range: tuple[datetime,datetime]) -> gpd.GeoDataFrame:
        """
        Fetch daily mean geopotential data at 500 hPa pressure level from ERA5.

        This method retrieves geopotential height data at the 500 hPa pressure level,
        which is commonly used in meteorological analysis to identify atmospheric patterns
        and pressure systems.

        Parameters:
            bbox (tuple[float, float, float, float]): 
                Bounding box coordinates in the format (min_longitude, min_latitude, max_longitude, max_latitude) defining the geographical area of interest.
            time_range (tuple[datetime, datetime]): 
                Time range as a tuple of two datetime objects (start_date, end_date) defining the temporal extent of the data.

        Returns:
            gpd.GeoDataFrame: 
                A GeoDataFrame containing the daily mean geopotential data
                at 500 hPa pressure level for the specified spatial and temporal extent.
        """
        return self.cds_client._fetch_data_pressure_levels("derived-era5-pressure-levels-daily-statistics", ['geopotential'], bbox, time_range, levels=[500], daily_statistic="daily_mean")

    def temperature_2m_mean(self, bbox: tuple[float,float,float,float], time_range: tuple[datetime,datetime], months:list[str]|list[int]|None=None, from_unit:str = "k", to_unit:str = "c") -> gpd.GeoDataFrame:
        """
        Fetches mean 2-meter temperature data for a specified bounding box and time range.
        This method attempts to retrieve data from multiple sources in the following order:
        1. Beacon cache (if available)
        2. CDS API (for missing data)
        3. MARS data (for any remaining gaps)
        4. MARS Forecast data 

        Parameters:
            bbox (tuple[float, float, float, float]):
                Bounding box defined as (min_longitude, min_latitude, max_longitude, max_latitude).
            time_range (tuple[datetime, datetime]):
                Tuple of (start_time, end_time) as datetime objects, inclusive.
            months (list[str] | list[int] | None, optional):
                Filter by month names or month numbers (1–12). If None, all months are included.
            from_unit (str, optional):
                Unit of the source temperature data. Default is "k".
            to_unit (str, optional):
                Desired output temperature unit. Default is "c".

        Returns:
            gpd.GeoDataFrame: 
                GeoDataFrame containing mean 2-meter temperature data
                in EPSG:4326 with temperature converted to the target unit.

        Notes:
            - Data gaps are automatically backfilled using multiple sources.
            - Remaining gaps trigger a warning.
            - MARS Forecast data retrieval is not fully implemented yet.
        """
        min_retrieved_time = None
        max_retrieved_time = None
        gdfs = []

        # Fetch the data from the beacon client if has been defined. Then check the min and max date and compare. Whatever is missing should be requested via the cds api
        if self.beacon_cache:
            gdfs, min_retrieved_time, max_retrieved_time = self.get_beacon_cache(gdfs, bbox, time_range, months=months, var='t2m', var_to='t2m', table='daily')

        if min_retrieved_time is None or max_retrieved_time is None or min_retrieved_time > time_range[0] or max_retrieved_time < time_range[1]:
            # No valid data found in beacon cache or we are missing data for the requested time range
            print("Missing data in beacon cache, fetching missing data from CDS...")

            if min_retrieved_time is None or min_retrieved_time > time_range[0]:
                # Request missing data from CDS
                fetch_start = time_range[0]
                fetch_end = min_retrieved_time if min_retrieved_time is not None else time_range[1]
                gdfs, min_retrieved_time, max_retrieved_time = self.get_cds_data(gdfs, bbox, (fetch_start, fetch_end), months=months, var='daily_mean', statistic='2m_temperature',
                                                                                 min_retrieved_time=min_retrieved_time, max_retrieved_time=max_retrieved_time)

            if max_retrieved_time is None or max_retrieved_time < time_range[1]:
                # Request missing data from CDS
                fetch_start = max_retrieved_time if max_retrieved_time is not None else time_range[0]
                fetch_end = time_range[1]
                gdfs, min_retrieved_time, max_retrieved_time = self.get_cds_data(gdfs, bbox, (fetch_start, fetch_end), months=months, var='daily_mean', statistic='2m_temperature',
                                                                                 min_retrieved_time=min_retrieved_time, max_retrieved_time=max_retrieved_time)

        if max_retrieved_time is None or max_retrieved_time < time_range[1]:
            # We are still missing data at the end of the range, try to fetch from MARS if available
            gdfs, min_retrieved_time, max_retrieved_time = self.get_mars_data(gdfs, bbox, time_range, var='t2m_mean', min_retrieved_time=min_retrieved_time, max_retrieved_time=max_retrieved_time)

        if max_retrieved_time is None or max_retrieved_time < time_range[1]:
            print("Still missing data. Fetching from MARS Forecast data...")
            if self.mars_client is not None:
                # Write warning that it is not implemented yet
                print("MARS Forecast data fetching not implemented yet.")

        # Concatenate all GeoDataFrames
        final_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs='EPSG:4326')

        # Implementation will go here
        df = self._convert_temp(final_gdf, from_unit, to_unit)

        return df

    def total_precipitation(self, bbox: tuple[float,float,float,float], time_range: tuple[datetime,datetime], months:list[str]|list[int]|None=None, from_unit:str = "m", to_unit:str = "mm") -> gpd.GeoDataFrame:
        """
        Fetch precipitation data for a given bounding box and time range.

        Parameters:
            bbox (tuple[float, float, float, float]):
                Bounding box as (min_longitude, min_latitude, max_longitude, max_latitude).
            time_range (tuple[datetime, datetime]):
                Tuple of (start_time, end_time) as datetime objects, inclusive.
            months (list[str] | list[int] | None, optional):
                Filter by specific months (names or numbers 1–12). Defaults to None (all months).
            from_unit (str, optional):
                Source unit of precipitation. Default is "m".
            to_unit (str, optional):
                Target unit of precipitation. Default is "mm".

        Returns:
            gpd.GeoDataFrame: GeoDataFrame containing precipitation data in the specified unit.
            Precipitation values are expressed in L/m²/day.
        """

        # Implementation will go here
        min_retrieved_time = None
        max_retrieved_time = None
        gdfs = []

        # Fetch the data from the beacon client if has been defined. Then check the min and max date and compare. Whatever is missing should be requested via the cds api
        if self.beacon_cache:
            gdfs, min_retrieved_time, max_retrieved_time = self.get_beacon_cache(gdfs, bbox, time_range, months=months, var='total_precipitation', var_to='tp', table='daily')

        if min_retrieved_time is None or max_retrieved_time is None or min_retrieved_time > time_range[0] or max_retrieved_time < time_range[1]:
            # No valid data found in beacon cache or we are missing data for the requested time range
            print("Missing data in beacon cache, fetching missing data from CDS...")

            if min_retrieved_time is None or min_retrieved_time > time_range[0]:
                # Request missing data from CDS
                fetch_start = time_range[0]
                fetch_end = min_retrieved_time if min_retrieved_time is not None else time_range[1]
                gdfs, min_retrieved_time, max_retrieved_time = self.get_cds_data(gdfs, bbox, (fetch_start, fetch_end), months=months, var='daily_sum', statistic='tp',
                                                                                 min_retrieved_time=min_retrieved_time, max_retrieved_time=max_retrieved_time)

            if max_retrieved_time is None or max_retrieved_time < time_range[1]:
                # Request missing data from CDS
                fetch_start = max_retrieved_time if max_retrieved_time is not None else time_range[0]
                fetch_end = time_range[1]
                gdfs, min_retrieved_time, max_retrieved_time = self.get_cds_data(gdfs, bbox, (fetch_start, fetch_end), months=months, var='daily_sum', statistic='tp',
                                                                                 min_retrieved_time=min_retrieved_time, max_retrieved_time=max_retrieved_time)

        if max_retrieved_time is None or max_retrieved_time < time_range[1]:
            # We are still missing data at the end of the range, try to fetch from MARS if available
            gdfs, min_retrieved_time, max_retrieved_time = self.get_mars_data(gdfs, bbox, time_range, var='tp', min_retrieved_time=min_retrieved_time, max_retrieved_time=max_retrieved_time)

        if max_retrieved_time is None or max_retrieved_time < time_range[1]:
            print("Still missing data. Fetching from MARS Forecast data...")
            if self.mars_client is not None:
                # Write warning that it is not implemented yet
                print("MARS Forecast data fetching not implemented yet.")

        # Concatenate all GeoDataFrames
        final_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs='EPSG:4326')

        # Implementation will go here
        df = self._convert_precipitation(final_gdf, from_unit, to_unit)

        return df

    def gmst(self, bbox: tuple[float,float,float,float], time_range: tuple[datetime,datetime], from_unit:str = "k", to_unit:str = "c") -> gpd.GeoDataFrame:
        """
        Fetch global mean surface temperature data for a given bounding box and time range.

        Parameters:
            bbox (tuple[float, float, float, float]):
                Bounding box as (min_longitude, min_latitude, max_longitude, max_latitude).
            time_range (tuple[datetime, datetime]):
                Tuple of (start_time, end_time) as datetime objects, inclusive.
            from_unit (str, optional):
                Source temperature unit. Default is "k" (Kelvin).
            to_unit (str, optional):
                Target temperature unit. Default is "c" (Celsius).

        Returns:
            gpd.GeoDataFrame: GeoDataFrame containing global mean surface temperature data
            in the specified unit.
        """
        # Implementation will go here
        min_retrieved_time = None
        max_retrieved_time = None
        gdfs = []

        # Fetch the data from the beacon client if has been defined. Then check the min and max date and compare. Whatever is missing should be requested via the cds api
        if self.beacon_cache:
            print("Fetching gmst data from beacon cache...")
            gdf = self.beacon_cache._fetch_from_era5_monthly_zarr_gpd(bbox=bbox, time_range=time_range, variable='t2m')
            if not gdf.empty:
                gdf = gdf.sort_values(['valid_time', 'longitude', 'latitude']).reset_index(drop=True)
                # Get the min and max valid time from the DF and validate it covers the requested time range or else fill with era5 cds request
                min_retrieved_time = min(gdf['valid_time'].min(), min_retrieved_time) if min_retrieved_time is not None else gdf['valid_time'].min()
                max_retrieved_time = max(gdf['valid_time'].max(), max_retrieved_time) if max_retrieved_time is not None else gdf['valid_time'].max()
                gdfs.append(gdf)

        if min_retrieved_time is None or max_retrieved_time is None or min_retrieved_time > time_range[0] or max_retrieved_time < time_range[1]:
            # No valid data found in beacon cache or we are missing data for the requested time range
            print("Missing gmst data in beacon cache, fetching missing gmst data from CDS...")

            if min_retrieved_time is None or min_retrieved_time > time_range[0]:
                # Request missing data from CDS
                fetch_start = time_range[0]
                fetch_end = min_retrieved_time if min_retrieved_time is not None else time_range[1]
                print(f"Requesting missing data from CDS for range: {fetch_start} - {fetch_end}")
                gdf = self.cds_client._fetch_data_monthly_averaged('2m_temperature', bbox, (fetch_start, fetch_end))
                min_retrieved_time = gdf['valid_time'].min() if min_retrieved_time is None else min(min_retrieved_time, gdf['valid_time'].min())
                max_retrieved_time = gdf['valid_time'].max() if max_retrieved_time is None else max(max_retrieved_time, gdf['valid_time'].max())
                print(f"Fetched {len(gdf)} new records from CDS.")
                gdfs.append(gdf)

            if max_retrieved_time is None or max_retrieved_time < time_range[1]:
                # Request missing data from CDS
                fetch_start = max_retrieved_time if max_retrieved_time is not None else time_range[0]
                fetch_end = time_range[1]
                print(f"Re-Requesting missing data from CDS for range: {fetch_start} - {fetch_end}")
                gdf = self.cds_client._fetch_data_monthly_averaged('2m_temperature', bbox, (fetch_start, fetch_end))

                # Only retain data that is newer than max_retrieved_time
                gdf = gdf[gdf['valid_time'] > max_retrieved_time] if max_retrieved_time is not None else gdf
                min_retrieved_time = gdf['valid_time'].min() if min_retrieved_time is None else min(min_retrieved_time, gdf['valid_time'].min())
                max_retrieved_time = gdf['valid_time'].max() if max_retrieved_time is None else max(max_retrieved_time, gdf['valid_time'].max())
                print(f"Fetched {len(gdf)} new records from re-requesting CDS.")
                gdfs.append(gdf)

        # Concatenate all GeoDataFrames
        final_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs='EPSG:4326')

        final_gdf = self._convert_temp(final_gdf, from_unit, to_unit)

        return final_gdf

    def fetch_data(self, parameter:str, bbox: tuple[float,float,float,float], time_range: tuple[datetime,datetime], months:list[str]|list[int]|None=None, from_unit:str|None = None, to_unit:str|None = None) -> gpd.GeoDataFrame:
        """
        Retrieve climate data for a specified parameter, bounding box, and time range.

        Parameters:
            parameter (str):
                Climate parameter to retrieve. Supported values:
                * "tmean": Mean 2m temperature
                * "tmin": Minimum 2m temperature
                * "tmax": Maximum 2m temperature
                * "precipitation": Total precipitation
                * "z500": 500 hPa geopotential mean
                * "slp": Mean sea level pressure
            bbox (tuple[float, float, float, float]):
                Bounding box coordinates as (min_lon, min_lat, max_lon, max_lat).
            time_range (tuple[datetime, datetime]):
                Start and end datetime for the data request.
            months (list[str] | list[int] | None, optional):
                Months to filter data. Can be month names (full or abbreviated) or integers (1–12).
                Defaults to None (all months included).
            from_unit (str | None, optional):
                Unit to convert from. If None, no conversion is applied.
            to_unit (str | None, optional):
                Unit to convert to. If None, no conversion is applied.

        Returns:
            gpd.GeoDataFrame:
                GeoDataFrame containing the requested climate data.

        Raises:
            ValueError:
                Raised if an invalid `parameter` is provided, or if month values are invalid
                (integers not in 1–12, or unrecognized month names).

        Notes:
            * When `months` is specified, the time range is adjusted automatically from
                the first day of the minimum month to the last day of the maximum month
                within the given year range.
        """

        return self.GET(parameter, bbox, time_range, months, from_unit, to_unit)

    @deprecated("Use `self.fetch_data()` instead.")
    def GET(self, parameter:str, bbox: tuple[float,float,float,float], time_range: tuple[datetime,datetime], months:list[str]|list[int]|None=None, from_unit:str|None = None, to_unit:str|None = None) -> gpd.GeoDataFrame:
        """
        Deprecated:
            Use `self.fetch_data()` instead.
        """
        parameter = parameter.lower()

        month_map = {
            "jan": 1,
            "january": 1,
            "feb": 2,
            "february": 2,
            "mar": 3,
            "march": 3,
            "apr": 4,
            "april": 4,
            "may": 5,
            "jun": 6,
            "june": 6,
            "jul": 7,
            "july": 7,
            "aug": 8,
            "august": 8,
            "sep": 9,
            "sept": 9,
            "september": 9,
            "oct": 10,
            "october": 10,
            "nov": 11,
            "november": 11,
            "dec": 12,
            "december": 12,
        }
        
        # Build kwargs only if explicitly set
        unit_kwargs = {}
        if from_unit is not None:
            unit_kwargs["from_unit"] = from_unit
        if to_unit is not None:
            unit_kwargs["to_unit"] = to_unit

        allowed_months: set[int] = set()
        if months is not None:

            # Normalize months → integers 1–12
            for m in months:
                if isinstance(m, int):
                    if 1 <= m <= 12:
                        allowed_months.add(m)
                    else:
                        raise ValueError(f"Invalid month integer: {m}")
                elif isinstance(m, str):
                    m_lower = m.lower()
                    if m_lower in month_map:
                        allowed_months.add(month_map[m_lower])
                    else:
                        raise ValueError(f"Invalid month string: {m}")
                else:
                    raise ValueError(f"Month must be int or str, got: {type(m)}")

            # add months variable to kwargs
            unit_kwargs["months"] = allowed_months

            # adjust time range for checking beacon cache
            start = max(time_range[0], datetime(year=time_range[0].year, month=min(allowed_months), day=1))
            end = min(time_range[1], self.last_day_of_month(dt=datetime(year=time_range[1].year, month=max(allowed_months), day=1)))
            time_range = (start, end)

        match parameter:
            case "tmean":
                return self.temperature_2m_mean(bbox, time_range, **unit_kwargs)
            case 'tmin':
                return self.temperature_2m_min(bbox, time_range, **unit_kwargs)
            case 'tmax':
                return self.temperature_2m_max(bbox, time_range, **unit_kwargs)
            case 'precipitation':
                return self.total_precipitation(bbox, time_range, **unit_kwargs)
            case 'z500': # 'z500_geopotential_mean':
                return self.z500_geopotential_mean(bbox, time_range, **unit_kwargs)
            case 'slp': # 'mean_sea_level_pressure':
                return self.mean_sea_level_pressure(bbox, time_range, **unit_kwargs)
            case 'gmst':
                return self.gmst(bbox, time_range, **unit_kwargs)
            case _:
                return ValueError(f"Unsupported parameter: {parameter}")

class CDSClient():
    CDS_API_URL = "https://cds.climate.copernicus.eu/api"
    
    def __init__(self, cds_key: str):
        self.cds_client = Client(self.CDS_API_URL, key=cds_key)
        
    def _split_time_range_by_year(
        self,
        start: datetime,
        end: datetime
    ) -> list[tuple[datetime, datetime]]:
        """
        Split a time range into sub-ranges, each within a single calendar year.
        """
        ranges = []
        current_start = start
        # iterate until within same year as end
        while current_start.year < end.year:
            year_end = datetime(year=current_start.year, month=12, day=31)
            ranges.append((current_start, year_end))
            current_start = datetime(year=current_start.year + 1, month=1, day=1)
        ranges.append((current_start, end))
        return ranges
    
    def _split_time_rangbe_by_year_and_months(
        self,
        start: datetime,
        end: datetime,
        months: list[str]|list[int]
    ) -> list[tuple[datetime, datetime]]:
    
        result = []

        def last_day_of_month(dt: datetime) -> datetime:
            next_month = dt.replace(day=28) + timedelta(days=4)  # always moves to the next month
            return next_month.replace(day=1) - timedelta(days=1)    # always returns back to the current month
        
        current = datetime(start.year, start.month, start.day)

        while current <= end:
            if current.month in months:
                month_start = current
                month_end = last_day_of_month(current)

                actual_start = max(month_start, start)
                actual_end = min(month_end, end)

                if actual_start <= actual_end:
                    result.append((actual_start, actual_end))
                
            if current.month == 12:
                current = datetime(current.year + 1, 1, 1)
            else:
                current = datetime(current.year, current.month + 1, 1)
        
        return result
    
    def _build_request_monthly_averaged(self, variable: str, time_range: tuple[datetime, datetime], bbox: tuple[float, float, float, float]) -> dict:
        start_dt, end_dt = time_range
        # convert to pandas Timestamp for range generation
        start = pd.Timestamp(start_dt)
        end = pd.Timestamp(end_dt)
        dates = pd.date_range(start=start, end=end, freq='MS')

        years = sorted({d.strftime('%Y') for d in dates})
        months = sorted({d.strftime('%m') for d in dates})

        request = {
            "product_type": "monthly_averaged_reanalysis",
            "variable": [variable],
            "year": years,
            "month": months,
            "time": ["00:00"],
            "data_format": "netcdf",
            "download_format": "unarchived",
            "area": [bbox[3], bbox[0], bbox[1], bbox[2]],  # CDS expects [north, west, south, east]
        }
        return request
    
    def _fetch_data_monthly_averaged(self, variable: str, bbox: tuple[float, float, float, float], time_range: tuple[datetime, datetime]) -> gpd.GeoDataFrame:
        dataset = "reanalysis-era5-single-levels-monthly-means"
        
        req = self._build_request_monthly_averaged(variable, time_range, bbox)
        with tempfile.NamedTemporaryFile(suffix='.nc') as tmp:
            self.cds_client.retrieve(
                dataset,
                req
            ).download(target=tmp.name)
            # open dataset
            ds = xr.open_dataset(tmp.name)
            # convert entire dataset to DataFrame
            df = ds.to_dataframe().reset_index()
            # create geometry
            df['geometry'] = gpd.points_from_xy(df['longitude'], df['latitude'])
            gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
            return gdf
    
    def _build_request_pressure_levels(self, variables: list[str], bbox: tuple[float, float, float, float], time_range: tuple[datetime, datetime], levels: list[int], daily_statistic: str = "daily_mean") -> dict:
        start_dt, end_dt = time_range
        # convert to pandas Timestamp for range generation
        start = pd.Timestamp(start_dt)
        end = pd.Timestamp(end_dt)
        dates = pd.date_range(start=start, end=end, freq='D')

        year = str(start.year)
        months = sorted({d.strftime('%m') for d in dates})
        days = sorted({d.strftime('%d') for d in dates})

        request = {
            "product_type": "reanalysis",
            "variable": variables,
            "year": year,
            "month": months,
            "day": days,
            "pressure_level": levels,
            "daily_statistic": daily_statistic,
            "time_zone": "utc+00:00",
            "frequency": "1_hourly",
            # CDS expects [north, west, south, east]
            "area": [bbox[3], bbox[0], bbox[1], bbox[2]],
        }
        return request

    def _build_request_single_levels(
        self,
        variables: list[str],
        bbox: tuple[float, float, float, float],
        time_range: tuple[datetime, datetime],
        daily_statistic: str = "daily_mean"
    ) -> dict:
        """
        Build the CDS API request dictionary for ERA5 daily statistics, limited to one year.

        variables: list of variable names for CDS (e.g. ['2m_temperature'])
        bbox: (min_lon, min_lat, max_lon, max_lat)
        time_range: (start_datetime, end_datetime) within a single year
        """
        start_dt, end_dt = time_range
        # convert to pandas Timestamp for range generation
        start = pd.Timestamp(start_dt)
        end = pd.Timestamp(end_dt)
        dates = pd.date_range(start=start, end=end, freq='D')

        year = str(start.year)
        months = sorted({d.strftime('%m') for d in dates})
        days = sorted({d.strftime('%d') for d in dates})

        request = {
            "product_type": "reanalysis",
            "variable": variables,
            "year": year,
            "month": months,
            "day": days,
            "daily_statistic": daily_statistic,
            "time_zone": "utc+00:00",
            "frequency": "1_hourly",
            # CDS expects [north, west, south, east]
            "area": [bbox[3], bbox[0], bbox[1], bbox[2]],
        }
        return request
    
    def _fetch_data_pressure_levels(
        self,
        dataset: str,
        variables: list[str],
        bbox: tuple[float, float, float, float],
        time_range: tuple[datetime, datetime],
        levels: list[int],
        daily_statistic: str = "daily_mean"
    ) -> gpd.GeoDataFrame:
        """
        Fetch data from CDS API for given variables and bbox, splitting by year.
        Saves each year's netCDF to a temp file, converts entire xarray dataset
        straight to a pandas DataFrame, then to a GeoDataFrame.
        Returns a list of GeoDataFrames, one per year.
        """
        ranges = self._split_time_range_by_year(*time_range)
        gdfs = []
        for start_dt, end_dt in ranges:
            req = self._build_request_pressure_levels(variables, bbox, (start_dt, end_dt), levels, daily_statistic=daily_statistic)
            with tempfile.NamedTemporaryFile(suffix='.nc') as tmp:
                self.cds_client.retrieve(
                    dataset,
                    req
                ).download(target=tmp.name)
                # open dataset
                ds = xr.open_dataset(tmp.name)
                # convert entire dataset to DataFrame
                df = ds.to_dataframe().reset_index()
                # create geometry
                df['geometry'] = gpd.points_from_xy(df['longitude'], df['latitude'])
                gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
                gdfs.append(gdf)
                
        combined_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs='EPSG:4326')

        # filter on given time range instead of taking the entire month
        combined_gdf = combined_gdf[(combined_gdf['valid_time'] >= start_dt) & (combined_gdf['valid_time'] <= end_dt)]

        return combined_gdf

    def _fetch_data_single_levels(
        self,
        dataset: str,
        variables: list[str],
        bbox: tuple[float, float, float, float],
        time_range: tuple[datetime, datetime],
        daily_statistic: str = "daily_mean",
        months: list[int]|None = None
    ) -> gpd.GeoDataFrame:
        """
        Fetch data from CDS API for given variables and bbox, splitting by year.
        Saves each year's netCDF to a temp file, converts entire xarray dataset
        straight to a pandas DataFrame, then to a GeoDataFrame.
        Returns a list of GeoDataFrames, one per year.
        """
        ranges = self._split_time_rangbe_by_year_and_months(time_range[0], time_range[1], months) if months is not None else self._split_time_range_by_year(*time_range)
        min_start = min(t[0] for t in ranges)
        max_end = max(t[1] for t in ranges)

        gdfs = []
        for start_dt, end_dt in ranges:
            req = self._build_request_single_levels(variables, bbox, (start_dt, end_dt), daily_statistic=daily_statistic)
            with tempfile.NamedTemporaryFile(suffix='.nc') as tmp:
                self.cds_client.retrieve(
                    dataset,
                    req
                ).download(target=tmp.name)
                # open dataset
                ds = xr.open_dataset(tmp.name)
                # convert entire dataset to DataFrame
                df = ds.to_dataframe().reset_index()
                # create geometry
                col_to_drop = 'number'
                if col_to_drop in df.columns:
                    df = df.drop(columns=[col_to_drop])
                df['geometry'] = gpd.points_from_xy(df['longitude'], df['latitude'])
                gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
                gdfs.append(gdf)
                
        combined_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs='EPSG:4326')

        # filter on given time range instead of taking the entire month
        combined_gdf = combined_gdf[(combined_gdf['valid_time'] >= min_start) & (combined_gdf['valid_time'] <= max_end)]

        return combined_gdf


class BeaconDataClient():
    def __init__(self, beacon_cache_url: str, beacon_token: str | None = None) -> None:
        self.beacon_cache_url = beacon_cache_url
        self.beacon_token = beacon_token

        self.beacon_client = beacon_api.Client(
            url=self.beacon_cache_url,
            jwt_token=self.beacon_token
        )
        
        # Do a check for connecting to the Beacon Cache Layer
        self.beacon_client.check_status()

    # copied for CDSClient so not the best practice but quick solution
    def _split_time_range_by_year(
        self,
        start: datetime,
        end: datetime
    ) -> list[tuple[datetime, datetime]]:
        """
        Split a time range into sub-ranges, each within a single calendar year.
        """
        ranges = []
        current_start = start
        # iterate until within same year as end
        while current_start.year < end.year:
            year_end = datetime(year=current_start.year, month=12, day=31)
            ranges.append((current_start, year_end))
            current_start = datetime(year=current_start.year + 1, month=1, day=1)
        ranges.append((current_start, end))
        return ranges


    def _split_time_rangbe_by_year_and_months(
            self,
            start: datetime,
            end: datetime,
            months: list[str]|list[int]
        ) -> list[tuple[datetime, datetime]]:
        
        result = []

        def last_day_of_month(dt: datetime) -> datetime:
            next_month = dt.replace(day=28) + timedelta(days=4)  # always moves to the next month
            return next_month.replace(day=1) - timedelta(days=1)    # always returns back to the current month
        
        current = datetime(start.year, start.month, start.day)

        while current <= end:
            if current.month in months:
                month_start = current
                month_end = last_day_of_month(current)

                actual_start = max(month_start, start)
                actual_end = min(month_end, end)

                if actual_start <= actual_end:
                    result.append((actual_start, actual_end))
                
            if current.month == 12:
                current = datetime(current.year + 1, 1, 1)
            else:
                current = datetime(current.year, current.month + 1, 1)
        
        return result

    
        
    def _fetch_from_era5_monthly_zarr_gpd(self, bbox: tuple[float,float,float,float], time_range: tuple[datetime,datetime], variable: str) -> gpd.GeoDataFrame:
        """
        Fetch data from ERA5 Monthly Zarr dataset in Beacon Cache.
        Parameters
        ----------
        bbox : tuple (min_lon, min_lat, max_lon, max_lat)
        time_range : tuple (start_datetime, end_datetime)
        variable : str, variable name to fetch. Available: ['t2m', 'total_precipitation']
        Returns
        -------
        GeoDataFrame with data
        """

        era5_table = self.beacon_client.list_tables()['era5_monthly_zarr']
        query = (era5_table.query()
                 .add_select_column('longitude')
                 .add_select_column('latitude')
                 .add_select_column('valid_time')
                 .add_select_column(variable)
                 .add_bbox_filter('longitude','latitude', bbox)
                 .add_range_filter('valid_time', gt_eq=time_range[0], lt_eq=time_range[1]))

        df = query.to_pandas_dataframe()
        if df.empty:
            return gpd.GeoDataFrame(columns=['longitude', 'latitude', 'valid_time'], geometry=gpd.points_from_xy([], []), crs='EPSG:4326')

        return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs='EPSG:4326')

    def _fetch_from_era5_daily_zarr_xr(self, bbox: tuple[float,float,float,float], time_range: tuple[datetime,datetime], variable: str) -> xr.Dataset:
        """
        Fetch data from ERA5 Monthly Zarr dataset in Beacon Cache.
        Parameters
        ----------
        bbox : tuple (min_lon, min_lat, max_lon, max_lat)
        time_range : tuple (start_datetime, end_datetime)
        variable : str, variable name to fetch. Available: ['t2m', 'total_precipitation']
        Returns
        -------
        GeoDataFrame with data
        """

        era5_table = self.beacon_client.list_tables()['daily']
        query = (era5_table.query()
                 .add_select_column('longitude')
                 .add_select_column('latitude')
                 .add_select_column('valid_time')
                 .add_select_column(variable)
                 .add_bbox_filter('longitude','latitude', bbox)
                 .add_range_filter('valid_time', gt_eq=time_range[0], lt_eq=time_range[1]))

        ds = query.to_xarray_dataset(dimension_columns=['longitude','latitude','valid_time'], force=True)
        return ds    

    def _fetch_from_era5_daily_zarr_iris(self, bbox: tuple[float,float,float,float], time_range: tuple[datetime,datetime], variable: str):
        """
        Fetch data from ERA5 Monthly Zarr dataset in Beacon Cache.
        Parameters
        ----------
        bbox : tuple (min_lon, min_lat, max_lon, max_lat)
        time_range : tuple (start_datetime, end_datetime)
        variable : str, variable name to fetch. Available: ['t2m', 'total_precipitation']
        Returns
        -------
        GeoDataFrame with data
        """

        era5_table = self.beacon_client.list_tables()['daily']
        query = (era5_table.query()
                 .add_select_column('longitude')
                 .add_select_column('latitude')
                 .add_select_column('valid_time')
                 .add_select_column(variable)
                 .add_bbox_filter('longitude','latitude', bbox)
                 .add_range_filter('valid_time', gt_eq=time_range[0], lt_eq=time_range[1]))
        import tempfile
        temp_file = tempfile.NamedTemporaryFile()
        query.to_nd_netcdf('some_path.nc', ['longitude','latitude','valid_time'], force=True)

        iris_cube = iris.iris.load_cube('some_path.nc')
        return iris_cube

    def _fetch_from_era5_daily_zarr(self, bbox: tuple[float,float,float,float], time_range: tuple[datetime,datetime], variable: str, months:list[str]|list[int]|None=None, table:str='daily') -> gpd.GeoDataFrame:
        """
        Fetch data from ERA5 Zarr dataset in Beacon Cache.

        Parameters
        ----------
        bbox : tuple (min_lon, min_lat, max_lon, max_lat)
        time_range : tuple (start_datetime, end_datetime)
        variable : str, variable name to fetch. Available: ['t2m', 't2m_max', 't2m_tmin', 'total_precipitation']

        Returns
        -------
        GeoDataFrame with data
        """
        era5_table = self.beacon_client.list_tables()[table]

        # do a lopedy loop loop for the months
        if months is not None and len(months) < 12:
            ranges = self._split_time_rangbe_by_year_and_months(start=time_range[0], end=time_range[1], months=months)
            gdfs = []
            for start_dt, end_dt in ranges:
                #do something
                query = (era5_table.query()
                     .add_select_column('longitude')
                     .add_select_column('latitude')
                     .add_select_column('valid_time')
                     .add_select_column(variable)
                     .add_bbox_filter('longitude','latitude', bbox)
                     .add_range_filter('valid_time', gt_eq=start_dt, lt_eq=end_dt))
                try:
                    df = query.to_pandas_dataframe()
                    if not df.empty:
                        gdfs.append(df)
                except Exception as e:
                    print (e)
                    continue

            df = pd.concat(gdfs, ignore_index=True)

        else:
            # era5_table = self.beacon_client.list_tables()['daily']
            query = (era5_table.query()
                     .add_select_column('longitude')
                     .add_select_column('latitude')
                     .add_select_column('valid_time')
                     .add_select_column(variable)
                     .add_bbox_filter('longitude','latitude', bbox)
                     .add_range_filter('valid_time', gt_eq=time_range[0], lt_eq=time_range[1]))

            df = query.to_pandas_dataframe()
        
        # Wrap longitude values to -180 to 180
        df['longitude'] = (df['longitude'] + 180) % 360 - 180
        
        if df.empty:
            return gpd.GeoDataFrame(columns=['longitude', 'latitude', 'valid_time'], geometry=gpd.points_from_xy([], []), crs='EPSG:4326')

        return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs='EPSG:4326')

    def _fetch_temperature_data(self, bbox: tuple[float, float, float, float], time_range: tuple[datetime, datetime], columns: list[str]) -> gpd.GeoDataFrame:
        
        era5_table = self.beacon_client.list_tables()['era5_daily_mean_2m_temperature']
        query = (era5_table.query()
                 .add_select_column('longitude')
                 .add_select_column('latitude')
                 .add_select_column('valid_time')
                 .add_bbox_filter('longitude','latitude', bbox)
                 .add_range_filter('valid_time', gt_eq=time_range[0], lt_eq=time_range[1]))

        if columns:
            for col in columns:
                query = query.add_select_column(col)

        df = query.to_pandas_dataframe()
        if df.empty:
            return gpd.GeoDataFrame(columns=['longitude', 'latitude', 'valid_time'], geometry=gpd.points_from_xy([], []), crs='EPSG:4326')

        df['longitude'] = (df['longitude'] + 180) % 360 - 180
        return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs='EPSG:4326')
    
    def _fetch_total_precipitation_data(self, bbox: tuple[float, float, float, float], time_range: tuple[datetime, datetime]) -> gpd.GeoDataFrame:
        era5_table = self.beacon_client.list_tables()['era5_daily_total_precipitation']
        query = (era5_table.query()
                 .add_select_column('longitude')
                 .add_select_column('latitude')
                 .add_select_column('valid_time')
                 .add_select_column('tp')
                 .add_bbox_filter('longitude','latitude', bbox)
                 .add_range_filter('valid_time', gt_eq=time_range[0], lt_eq=time_range[1]))

        df = query.to_pandas_dataframe()
        if df.empty:
            return gpd.GeoDataFrame(columns=['longitude', 'latitude', 'valid_time'], geometry=gpd.points_from_xy([], []), crs='EPSG:4326')

        df['longitude'] = (df['longitude'] + 180) % 360 - 180
        return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs='EPSG:4326')

class MarsClient():
    def __init__(self, key: str):
        from ecmwfapi import ECMWFService
        self.key = key
        self.server = ECMWFService("mars", key=key, url='https://api.ecmwf.int/v1')
        self.check_cdo()
    
    def check_cdo(self):
        import shutil, subprocess
        if shutil.which("cdo") is None:
            raise EnvironmentError("❌ CDO not found. Please install it (e.g. `sudo apt install cdo`).")

        try:
            v = subprocess.run(["cdo", "-V"], capture_output=True, text=True, check=True)
            print(f"✅ {v.stdout.splitlines()[0]}")
        except Exception as e:
            raise EnvironmentError(f"⚠️ CDO check failed: {e}")
    
    def get_date_list(self, min_date: datetime, max_date: datetime) -> list[str]:
        # Get current system date
        given_date = datetime.now() - timedelta(days=1)  # Yesterday
        
        # Ensure given_date is within min_date and max_date
        if max_date > given_date:
            max_date = given_date

        # Generate list of days before the current date
        date_list = []
        while min_date <= max_date:
            date_list.append(min_date.strftime("%Y-%m-%d"))
            min_date += timedelta(days=1)

        return date_list    
    
    def get_temp_path(self) -> str:
        import tempfile
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.nc')
        return temp_file.name
    
    def fetch_t2m_mean_operational_data(self, min_date: datetime, max_date: datetime, min_lon: float, max_lon: float, min_lat: float, max_lat: float) -> gpd.GeoDataFrame:
        time = "00:00:00/06:00:00/12:00:00/18:00:00"
        # Fetch the current date -7 days as a list of dates
        print(f"Fetching t2m data for past 7 days from MARS: {self.get_date_list(min_date, max_date)}")
        request = {
            "class": "od",
            "date": self.get_date_list(min_date, max_date),
            "expver": "1",
            "param": self.find_param_code("t2m"), # 2m temperature
            "grid": "0.25/0.25", # 0.25 degree grid
            "time": time,
            "format": format,
            "class": "od",
            "levtype": "sfc",
            "stream": "oper",
            "type": "an",
            "target": "output",
            "format": "netcdf",
        }
        temp_path = self.get_temp_path()
        self.server.execute(request,target=temp_path)
        
        # USE CDO to compute daily mean
        out_daily = temp_path.replace(".nc", "_daily.nc")
        subprocess.run([
            "cdo", "-O", "-r", "-f", "nc4", "-s",
            f"-daymean", f"-shifttime,3hour", temp_path, out_daily
        ], check=True)
        
        ds = xr.open_dataset(out_daily)
        # Convert to dataframe but only keep (lon, lat, time, t2m)
        df = ds[['longitude', 'latitude', 'time', 't2m']].to_dataframe().reset_index()
        out_daily = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")
        
        # Rename time to valid_time for clarity
        out_daily = out_daily.rename(columns={"time": "valid_time"})
        
        # Translate longitude from 0-360 to -180 to 180
        out_daily['longitude'] = (out_daily['longitude'] + 180) % 360 - 180
        
        # Filter by bounding box
        out_daily = out_daily[
            (out_daily['longitude'] >= min_lon) &
            (out_daily['longitude'] <= max_lon) &
            (out_daily['latitude'] >= min_lat) &
            (out_daily['latitude'] <= max_lat)
        ]
        return out_daily
    
    def fetch_t2m_min_operational_data(self, min_date: datetime, max_date: datetime, min_lon: float, max_lon: float, min_lat: float, max_lat: float) -> gpd.GeoDataFrame:
        time = "00:00:00/12:00:00"
        # Fetch the current date -7 days as a list of dates
        print(f"Fetching t2m min data for past 7 days from MARS: {self.get_date_list(min_date, max_date)}")
        request = {
            "class": "od",
            "step": "6/12",
            "date": self.get_date_list(min_date, max_date),
            "expver": "1",
            "param": self.find_param_code("tmin"), # 2m temperature
            "grid": "0.25/0.25", # 0.25 degree grid
            "time": time,
            "format": format,
            "class": "od",
            "levtype": "sfc",
            "stream": "oper",
            "type": "fc",
            "target": "output",
            "format": "netcdf",
        }
        temp_path = self.get_temp_path()
        self.server.execute(request,target=temp_path)
        
        # USE CDO to compute daily min
        out_daily = temp_path.replace(".nc", "_daily.nc")
        subprocess.run([
            "cdo", "-O", "-r", "-f", "nc4", "-s",
            f"-daymin", f"-shifttime,-3hour", temp_path, out_daily
        ], check=True)
        
        ds = xr.open_dataset(out_daily)
        print(ds)
        # Convert to dataframe but only keep (lon, lat, time, tmin)
        df = ds[['longitude', 'latitude', 'time', 'mn2t6']].to_dataframe().reset_index()
        out_daily = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")
        
        # Rename mn2t6 to tmin for clarity
        out_daily = out_daily.rename(columns={"mn2t6": "t2m", "time": "valid_time"})
        
        # Translate longitude from 0-360 to -180 to 180
        out_daily['longitude'] = (out_daily['longitude'] + 180) % 360 - 180
        
        # Filter by bounding box
        out_daily = out_daily[
            (out_daily['longitude'] >= min_lon) &
            (out_daily['longitude'] <= max_lon) &
            (out_daily['latitude'] >= min_lat) &
            (out_daily['latitude'] <= max_lat)
        ]
        
        return out_daily
    
    def fetch_t2m_max_operational_data(self, min_date: datetime, max_date: datetime, min_lon: float, max_lon: float, min_lat: float, max_lat: float) -> gpd.GeoDataFrame:
        time = "00:00:00/12:00:00"
        # Fetch the current date -7 days as a list of dates
        print(f"Fetching t2m max data from MARS: {self.get_date_list(min_date, max_date)}")
        request = {
            "class": "od",
            "step": "6/12",
            "date": self.get_date_list(min_date, max_date),
            "expver": "1",
            "param": self.find_param_code("tmax"), # 2m temperature
            "grid": "0.25/0.25", # 0.25 degree grid
            "time": time,
            "format": format,
            "class": "od",
            "levtype": "sfc",
            "stream": "oper",
            "type": "fc",
            "target": "output",
            "format": "netcdf",
        }
        temp_path = self.get_temp_path()
        self.server.execute(request,target=temp_path)
        
        # USE CDO to compute daily max
        out_daily = temp_path.replace(".nc", "_daily.nc")
        subprocess.run([
            "cdo", "-O", "-r", "-f", "nc4", "-s",
            f"-daymax", f"-shifttime,-3hour", temp_path, out_daily
        ], check=True)
        
        ds = xr.open_dataset(out_daily)
        # Convert to dataframe but only keep (lon, lat, time, tmax)
        df = ds[['longitude', 'latitude', 'time', 'mx2t6']].to_dataframe().reset_index()
        out_daily = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")
        
        # Rename mx2t6 to tmax for consistency
        out_daily = out_daily.rename(columns={"mx2t6": "t2m", "time": "valid_time"})
        
        # Translate longitude from 0-360 to -180 to 180
        out_daily['longitude'] = (out_daily['longitude'] + 180) % 360 - 180
        
        # Filter by bounding box
        out_daily = out_daily[
            (out_daily['longitude'] >= min_lon) &
            (out_daily['longitude'] <= max_lon) &
            (out_daily['latitude'] >= min_lat) &
            (out_daily['latitude'] <= max_lat)
        ]
        
        return out_daily
    
    def fetch_total_precipitation_operational_data(self, min_date: datetime, max_date: datetime, min_lon: float, max_lon: float, min_lat: float, max_lat: float) -> gpd.GeoDataFrame:
        time = "00:00:00/12:00:00"
        # Fetch the current date -7 days as a list of dates
        print(f"Fetching tp data for past 7 days from MARS: {self.get_date_list(min_date, max_date)}")
        request = {
            "class": "od",
            "step": "6/12",
            "date": self.get_date_list(min_date, max_date),
            "expver": "1",
            "param": self.find_param_code("tp"), # total precipitation
            "grid": "0.25/0.25", # 0.25 degree grid
            "time": time,
            "format": format,
            "class": "od",
            "levtype": "sfc",
            "stream": "oper",
            "type": "fc",
            "target": "output",
            "format": "netcdf",
        }
        temp_path = self.get_temp_path()
        self.server.execute(request,target=temp_path)
        
        # USE CDO to compute daily sum
        out_daily = temp_path.replace(".nc", "_daily.nc")
        subprocess.run([
            "cdo", "-O", "-r", "-f", "nc4", "-s",
            f"-daysum", temp_path, out_daily
        ], check=True)
        
        ds = xr.open_dataset(out_daily)
        # Convert to dataframe but only keep (lon, lat, time, tp)
        df = ds[['longitude', 'latitude', 'time', 'tp']].to_dataframe().reset_index()
        out_daily = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")
        
        
        # Translate longitude from 0-360 to -180 to 180
        out_daily['longitude'] = (out_daily['longitude'] + 180) % 360 - 180
        
        # Filter by bounding box
        out_daily = out_daily[
            (out_daily['longitude'] >= min_lon) &
            (out_daily['longitude'] <= max_lon) &
            (out_daily['latitude'] >= min_lat) &
            (out_daily['latitude'] <= max_lat)
        ]
        
        return out_daily
    
    def fetch_t2m_mean_forecast_data(self) -> gpd.GeoDataFrame:
        # Fetch the current date -7 days as a list of dates
        current_date = datetime.utcnow() - timedelta(days=1) # Use yesterday's date to compensate for forecast delay
        date_str = current_date.strftime("%Y-%m-%d")
        print(f"Fetching t2m forecast data from MARS for date: {date_str}")
        request = {
            "class": "od",
            "date": date_str,
            "step": "6/12/18/24/30/36/42/48/54/60/66/72/78/84/90/96/102/108/114/120/126/132/138/144/150/156/162/168/174/180/186",
            "expver": "1",
            "param": self.find_param_code("t2m"), # 2m temperature
            "grid": "0.25/0.25", # 0.25 degree grid
            "time": "00:00:00",
            "format": format,
            "class": "od",
            "levtype": "sfc",
            "stream": "oper",
            "type": "fc",
            "target": "output",
            "format": "netcdf",
        }
        temp_path = self.get_temp_path()
        self.server.execute(request,target=temp_path)
        
        # USE CDO to compute daily mean
        out_daily = temp_path.replace(".nc", "_daily.nc")
        subprocess.run([
            "cdo", "-O", "-r", "-f", "nc4", "-s",
            f"-daymean", f"-shifttime,3hour", temp_path, out_daily
        ], check=True)
        
        ds = xr.open_dataset(out_daily)
        # Convert to dataframe but only keep (lon, lat, time, t2m)
        df = ds[['longitude', 'latitude', 'time', 't2m']].to_dataframe().reset_index()
        out_daily = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")
        return out_daily
    
    def find_param_code(self, name: str) -> str | None:
        """
        Find the ECMWF MARS parameter code for a given variable name.
        Parameters: t2m, z500, mslp, tp, tmin, tmax        
        """
        param_codes = {
            "t2m": "167.128",
            "z500": "129.128",
            "mslp": "151.128",
            "tp": "228.128",
            "tmin": "122.128",
            "tmax": "121.128",
            # Add more mappings as needed
        }
        return param_codes.get(name)

class CordexClient():
    def __init__(self):
        pass
