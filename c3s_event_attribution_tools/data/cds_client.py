import os
from cdsapi import Client
from datetime import datetime, timedelta
from .variable import Variable
from ..utils import Utils
import geopandas as gpd
import pandas as pd
import xarray as xr
import tempfile
import zipfile
import cftime
from hashlib import sha256
import numpy as np

if __import__('sys').platform in ['linux']:
    import iris # type: ignore

class CDSClient():
    CDS_API_URL = "https://cds.climate.copernicus.eu/api"
    
    def __init__(self, cds_key: str):
        """
        Initialize the CDSClient with credentials for the Copernicus Climate Data Store.

        This constructor sets up the underlying CDS API client using the 
        provided API key and the predefined CDS API URL.

        Parameters:
            cds_key (str):
                The API key required to authenticate with the Copernicus 
                Climate Data Store.
        """
        self.cds_client = Client(self.CDS_API_URL, key=cds_key)
        

    def _build_request_monthly_averaged(self, variable: str, time_range: tuple[datetime, datetime], bbox: tuple[float, float, float, float]) -> dict:
        """
        Build a dictionary request for monthly averaged ERA5 reanalysis data.

        This method prepares the parameters required for a CDS API call, extracting
        years and months from the provided time range and reformatting the 
        bounding box to the required coordinate order.

        Parameters:
            variable (str):
                The name of the variable to fetch from the dataset.
            time_range (tuple[datetime, datetime]):
                A tuple containing the (start_date, end_date) for the data request.
            bbox (tuple[float, float, float, float]):
                Bounding box coordinates as (min_longitude, min_latitude, max_longitude, max_latitude).

        Returns:
            dict: A dictionary containing the formatted request parameters for 
            the CDS API.
        """
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
        
    def _fetch_data_monthly_averaged_xr(self, variable: str, bbox: tuple[float, float, float, float], time_range: tuple[datetime, datetime]) -> xr.Dataset:
        """
        Fetch monthly averaged ERA5 data and return it as an xarray Dataset.

        This method downloads the requested data as a NetCDF file and loads it 
        into an xarray Dataset, applying a final temporal slice to match the 
        exact requested time range.

        Parameters:
            variable (str):
                The name of the variable to fetch from the dataset.
            bbox (tuple[float, float, float, float]):
                Bounding box coordinates as (min_longitude, min_latitude, max_longitude, max_latitude).
            time_range (tuple[datetime, datetime]):
                A tuple containing the (start_date, end_date) for the data request.

        Returns:
            xr.Dataset: An xarray Dataset containing the fetched monthly averaged data.
        """
        dataset = "reanalysis-era5-single-levels-monthly-means"
        request = self._build_request_monthly_averaged(variable, time_range, bbox)
        file = self._execute_cds_request(dataset, request)
        ds = xr.open_dataset(file)
        # filter time to exact range
        ds = ds.sel(valid_time=slice(time_range[0], time_range[1]))
        
        # Wrap longitude to -180 to 180
        ds = ds.assign_coords(
            longitude=((ds.longitude + 180) % 360) - 180
        ).sortby("longitude")
        
        return ds
    
    def _fetch_data_monthly_averaged_gpd(self, variable: str, bbox: tuple[float, float, float, float], time_range: tuple[datetime, datetime]) -> gpd.GeoDataFrame:
        """
        Fetch monthly averaged ERA5 data and return it as a GeoDataFrame.

        This method downloads the data, converts the resulting NetCDF into a 
        pandas DataFrame, generates point geometries from the spatial 
        coordinates, and returns a filtered GeoDataFrame.

        Parameters:
            variable (str):
                The name of the variable to fetch from the dataset.
            bbox (tuple[float, float, float, float]):
                Bounding box coordinates as (min_longitude, min_latitude, max_longitude, max_latitude).
            time_range (tuple[datetime, datetime]):
                A tuple containing the (start_date, end_date) for the data request.

        Returns:
            gpd.GeoDataFrame: A GeoDataFrame containing the monthly averaged data 
            with spatial and temporal information.
        """
        ds = self._fetch_data_monthly_averaged_xr(variable, bbox, time_range)
        # convert entire dataset to DataFrame
        df = ds.to_dataframe().reset_index()
        # create geometry
        df['geometry'] = gpd.points_from_xy(df['longitude'], df['latitude'])
        gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
        
        # filter time to exact range
        gdf = gdf[(gdf['valid_time'] >= time_range[0]) & (gdf['valid_time'] <= time_range[1])]

        return gdf
    
    def _build_request_pressure_levels(self, variables: list[str], bbox: tuple[float, float, float, float], time_range: tuple[datetime, datetime], levels: list[int], daily_statistic: str = "daily_mean") -> dict:
        """
        Build a dictionary request for ERA5 pressure level daily statistics.

        This method prepares the parameters for a CDS API call targeting the 
        reanalysis-era5-pressure-levels-daily-statistics dataset. It extracts 
        the year, months, and days from the time range and maps the bounding 
        box to the coordinate order required by the CDS API.

        Parameters:
            variables (list[str]):
                A list of variable names to fetch from the pressure levels dataset.
            bbox (tuple[float, float, float, float]):
                Bounding box coordinates as (min_longitude, min_latitude, max_longitude, max_latitude).
            time_range (tuple[datetime, datetime]):
                A tuple containing the (start_date, end_date) for the data request.
            levels (list[int]):
                A list of pressure levels (in hPa) to retrieve.
            daily_statistic (str):
                The type of daily statistic to compute (e.g., "daily_mean"). 
                Default is "daily_mean".

        Returns:
            dict: A dictionary containing the formatted request parameters for 
            the CDS API.
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

        This method prepares the parameters for a CDS API call targeting the
        reanalysis-era5-single-levels-daily-statistics dataset. It extracts the
        year, months, and days from the time range and maps the bounding box to
        the coordinate order required by the CDS API.

        Parameters:
            variables (list[str]):
                List of variable names for CDS (e.g., ['2m_temperature']).
            bbox (tuple[float, float, float, float]):
                Bounding box coordinates as (min_longitude, min_latitude, max_longitude, max_latitude).
            time_range (tuple[datetime, datetime]):
                A tuple containing the (start_date, end_date) defining a period
                within a single calendar year.
            daily_statistic (str):
                The type of daily statistic to compute (e.g., "daily_mean").
                Default is "daily_mean".

        Returns:
            dict: A dictionary containing the formatted request parameters for
            the CDS API.
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
    
    def _fetch_data_daily_pressure_levels(
        self,
        variable: Variable.ERA5DailyPressureLevels,
        bbox: tuple[float, float, float, float],
        time_ranges: list[tuple[datetime, datetime]],
        levels: list[int],
    ) -> list[str]:
        """
        Fetch data from CDS API for given variables and bbox, splitting by year.

        This method breaks the requested time range into annual sub-ranges,
        retrieves the data for each year from the CDS pressure levels dataset,
        and saves each result as a temporary NetCDF file.

        Parameters:
            variable (Variable):
                An instance of the Variable class containing CDS-specific naming
                and statistic metadata.
            bbox (tuple[float, float, float, float]):
                Bounding box coordinates as (min_longitude, min_latitude, max_longitude, max_latitude).
            time_range (tuple[datetime, datetime]):
                The full temporal range (start, end) for which to fetch data.
            levels (list[int]):
                A list of pressure levels (in hPa) to retrieve.

        Returns:
            list[str]: A list of file paths to the downloaded temporary NetCDF files.
        """
        dataset = "derived-era5-pressure-levels-daily-statistics"
        files = []
        for range in time_ranges:
            inner_range = Utils.split_time_range_by_year(*range)
            for start_dt, end_dt in inner_range:
                req = self._build_request_pressure_levels([variable.cds_name()], bbox, (start_dt, end_dt), levels, daily_statistic=variable.cds_daily_statistic())
                file = self._execute_cds_request(dataset, req)
                files.append(file)

        return files
    
    def fetch_data_daily_pressure_levels_gpd(
        self,
        variable: Variable.ERA5DailyPressureLevels,
        bbox: tuple[float, float, float, float],
        time_ranges: list[tuple[datetime, datetime]],
        levels: list[int],
    ) -> gpd.GeoDataFrame:
        """
        Fetch pressure level data and return it as a combined GeoDataFrame.

        This method retrieves data for specified pressure levels across multiple years 
        if necessary, converts the resulting NetCDF files into a tabular format, 
        generates spatial point geometries, and merges them into a single 
        GeoDataFrame filtered to the requested time range.

        Parameters:
            variable (Variable):
                An instance of the Variable class containing CDS-specific naming 
                and statistic metadata.
            bbox (tuple[float, float, float, float]):
                Bounding box coordinates as (min_longitude, min_latitude, max_longitude, max_latitude).
            time_range (tuple[datetime, datetime]):
                The full temporal range (start, end) for which to fetch data.
            levels (list[int]):
                A list of pressure levels (in hPa) to retrieve.

        Returns:
            gpd.GeoDataFrame: A GeoDataFrame containing the pressure level data 
            with spatial coordinates, pressure levels, and temporal information.
        """
        ds = self.fetch_data_daily_pressure_levels_xr(
            variable,
            bbox,
            time_ranges,
            levels,
        )
        # convert entire dataset to DataFrame
        df = ds.to_dataframe().reset_index()
        # create geometry
        df['geometry'] = gpd.points_from_xy(df['longitude'], df['latitude'])
        gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
        return gdf
    
    def fetch_data_daily_pressure_levels_xr(
        self,
        variable: Variable.ERA5DailyPressureLevels,
        bbox: tuple[float, float, float, float],
        time_ranges: list[tuple[datetime, datetime]],
        levels: list[int],
    ) -> xr.Dataset:
        """
        Fetch pressure level data and return it as a combined xarray Dataset.

        This method retrieves atmospheric data for specified pressure levels, 
        handling multi-year requests by fetching annual files and merging them 
        using xarray's multi-file dataset capabilities. The final dataset is 
        sliced to match the exact temporal boundaries requested.

        Parameters:
            variable (Variable):
                An instance of the Variable class containing CDS-specific naming 
                and statistic metadata.
            bbox (tuple[float, float, float, float]):
                Bounding box coordinates as (min_longitude, min_latitude, max_longitude, max_latitude).
            time_ranges (list[tuple[datetime, datetime]]):
                A list of temporal ranges (start, end) for which to fetch data.
            levels (list[int]):
                A list of pressure levels (in hPa) to retrieve.

        Returns:
            xr.Dataset: An xarray Dataset containing the combined and time-filtered 
            pressure level data.
        """
        files = self._fetch_data_daily_pressure_levels(
            variable,
            bbox,
            time_ranges,
            levels,
        )
        
        dss = []
        for file in files:
            # open dataset
            ds = xr.open_dataset(file)
            # convert entire dataset to DataFrame
            dss.append(ds)
        
        ds = xr.open_mfdataset(
            files,
            combine="by_coords"
        )
        
        # Wrap longitude to -180 to 180
        ds = ds.assign_coords(
            longitude=((ds.longitude + 180) % 360) - 180
        ).sortby("longitude")
        
        time_ranges64 = [
            (np.datetime64(start, "ns"), np.datetime64(end, "ns"))
            for start, end in time_ranges
        ]
        
        t = ds.coords["valid_time"]

        mask = xr.zeros_like(t, dtype=bool)

        for start, end in time_ranges64:
            mask |= (t >= start) & (t <= end)
        
        ds_sel = ds.where(mask, drop=True)
        
        return ds_sel
    
    def _fetch_data_daily_single_levels(
        self,
        variable: Variable.ERA5DailySingleLevel,
        bbox: tuple[float, float, float, float],
        time_ranges: list[tuple[datetime, datetime]],
    ) -> list[str]:
        """
        Fetch data from the CDS API for single-level variables, splitting by year or months.

        This internal method handles the retrieval of ERA5 single-level data by 
        partitioning the requested time range into smaller chunks (annually or 
        filtered by months), performing the CDS API calls, and saving the results 
        into temporary NetCDF files.

        Parameters:
            variable (Variable):
                An instance of the Variable class containing CDS-specific naming 
                and statistic metadata.
            bbox (tuple[float, float, float, float]):
                Bounding box coordinates as (min_longitude, min_latitude, max_longitude, max_latitude).
            time_range (tuple[datetime, datetime]):
                The full temporal range (start, end) for which to fetch data.
            months (list[int] | None):
                An optional list of integers (1-12) to filter the data retrieval 
                by specific months. Default is None.

        Returns:
            list[str]: A list of file paths to the downloaded temporary NetCDF files.
        """
        dataset = "derived-era5-single-levels-daily-statistics"
        files = []
        
        for range in time_ranges:
            ranges = Utils.split_time_range_by_year(*range)
            for start_dt, end_dt in ranges:
                req = self._build_request_single_levels([variable.cds_name()], bbox, (start_dt, end_dt), daily_statistic=variable.cds_daily_statistic())
                file = self._execute_cds_request(dataset, req)
                files.append(file)
                
        return files
    
    def fetch_data_daily_single_levels_gpd(
        self,
        variable: Variable.ERA5DailySingleLevel,
        bbox: tuple[float, float, float, float],
        time_ranges: list[tuple[datetime, datetime]],
    ) -> gpd.GeoDataFrame:
        """
        Fetch single-level data and return it as a combined GeoDataFrame.

        This method retrieves data for specified variables, handles multi-year 
        requests by fetching individual NetCDF files, converts the spatial and 
        temporal data into a tabular format with point geometries, and returns 
        a single GeoDataFrame filtered to the requested time range.

        Parameters:
            variable (Variable):
                An instance of the Variable class containing CDS-specific naming 
                and statistic metadata.
            bbox (tuple[float, float, float, float]):
                Bounding box coordinates as (min_longitude, min_latitude, max_longitude, max_latitude).
            time_range (tuple[datetime, datetime]):
                The full temporal range (start, end) for which to fetch data.

        Returns:
            gpd.GeoDataFrame: A GeoDataFrame containing the single-level data 
            with spatial coordinates and temporal information.
        """
        ds = self.fetch_data_daily_single_levels_xr(
            variable,
            bbox,
            time_ranges,
        )
        # convert entire dataset to DataFrame
        df = ds.to_dataframe().reset_index()
        # create geometry
        df['geometry'] = gpd.points_from_xy(df['longitude'], df['latitude'])
        gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')

        return gdf
    
    def fetch_data_daily_single_levels_xr(
        self,
        variable: Variable.ERA5DailySingleLevel,
        bbox: tuple[float, float, float, float],
        time_ranges: list[tuple[datetime, datetime]],
    ) -> xr.Dataset:
        """
        Fetch single-level data and return it as a combined xarray Dataset.

        This method retrieves atmospheric data for specified single-level variables, 
        handling multi-year requests by fetching annual files and merging them 
        using xarray's multi-file dataset capabilities. The final dataset is 
        sliced to match the exact temporal boundaries requested.

        Parameters:
            variable (Variable):
                An instance of the Variable class containing CDS-specific naming 
                and statistic metadata.
            bbox (tuple[float, float, float, float]):
                Bounding box coordinates as (min_longitude, min_latitude, max_longitude, max_latitude).
            time_range (tuple[datetime, datetime]):
                The full temporal range (start, end) for which to fetch data.

        Returns:
            xr.Dataset: An xarray Dataset containing the combined and time-filtered 
            single-level data.
        """
        files = self._fetch_data_daily_single_levels(
            variable,
            bbox,
            time_ranges,
        )
        
        dss = []
        for file in files:
            # open dataset
            ds = xr.open_dataset(file)
            # convert entire dataset to DataFrame
            dss.append(ds)
        
        ds = xr.open_mfdataset(
            files,
            combine="by_coords"
        )
        
        # Wrap longitude to -180 to 180
        ds = ds.assign_coords(
            longitude=((ds.longitude + 180) % 360) - 180
        ).sortby("longitude")
        
        time_ranges64 = [
            (np.datetime64(start, "ns"), np.datetime64(end, "ns"))
            for start, end in time_ranges
        ]
        
        t = ds.coords["valid_time"]

        mask = xr.zeros_like(t, dtype=bool)

        for start, end in time_ranges64:
            mask |= (t >= start) & (t <= end)
        
        ds_sel = ds.where(mask, drop=True)
        
        return ds_sel

    def _build_request_cmip6(self, variable: str, model:str, time_range: tuple[datetime, datetime], bbox: tuple[float, float, float, float], experiment: str = "ssp5_8_5", temporal_resolution: str = "daily") -> dict:
        start_dt, end_dt = time_range
        # convert to pandas Timestamp for range generation
        start = pd.Timestamp(start_dt)
        end = pd.Timestamp(end_dt)
        dates = pd.date_range(start=start, end=end, freq='D')

        years = sorted({d.strftime('%Y') for d in dates})
        months = sorted({d.strftime('%m') for d in dates})
        days = sorted({d.strftime('%d') for d in dates})
        
        request = {
            "temporal_resolution": temporal_resolution,
            "experiment": experiment,
            "variable": variable,
            "model": model,
            "year": years,
            "month": months,
            "day": days,
            "time_zone": "utc+00:00",
            "frequency": "1_hourly",
            # CDS expects [north, west, south, east]
            "area": [bbox[3], bbox[0], bbox[1], bbox[2]],
        }
        
        return request

    def _fetch_data_cmip6_netcdf(self, variable: Variable.CMIP6, model: str, bbox: tuple[float, float, float, float], time_range: tuple[datetime, datetime], experiment: str = "ssp5_8_5", temporal_resolution: str = "daily") -> str:
        dataset = "projections-cmip6"
        req = self._build_request_cmip6(variable.cds_name(), model, time_range, bbox, experiment, temporal_resolution)
        req_hash = sha256(str(req).encode('utf-8')).hexdigest()
        # Fetch the temporary directory of the OS
        temp_dir = tempfile.gettempdir()
        temp_file_path = os.path.join(temp_dir, f"cds_cmip6_{req_hash}.nc")
        
        if not os.path.exists(temp_file_path):
            print("Downloading CMIP6 data from CDS, this may take a while...")
            zip_path = os.path.join(temp_dir, f"cds_cmip6_{req_hash}.zip")
            # download as zip and extract
            self.cds_client.retrieve(
                dataset,
                req
            ).download(target=zip_path)
            # extract netcdf from zip and copy to temp_file_path
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            # find the netcdf file in the extracted files
            for file_name in zip_ref.namelist():
                if file_name.endswith('.nc'):
                    extracted_path = os.path.join(temp_dir, file_name)
                    os.rename(extracted_path, temp_file_path)
                    break
        else:
            print("Using locally cached CMIP6 data from CDS.")
        return temp_file_path
        
    def fetch_cmip6_xr(self, variable: Variable.CMIP6, model: str, bbox: tuple[float, float, float, float], time_range: tuple[datetime, datetime], experiment: str = "ssp5_8_5", temporal_resolution: str = "daily") -> xr.Dataset:
        file = self._fetch_data_cmip6_netcdf(variable, model, bbox, time_range, experiment, temporal_resolution)
        ds = xr.open_dataset(file)
        # Wrap longitude to -180 to 180
        ds = ds.assign_coords(
            lon=((ds.lon + 180) % 360) - 180
        ).sortby("lon")
        
        # Convert filter time range to approriate calender
        xr_time_start = Utils.datetime_to_xr_time(time_range[0], ds)
        xr_time_end = Utils.datetime_to_xr_time(time_range[1], ds)

        # filter time to exact range
        ds = ds.sel(time=slice(xr_time_start, xr_time_end))
        return ds
    
    def fetch_cmip6_gpd(self, variable: Variable.CMIP6, model: str, bbox: tuple[float, float, float, float], time_range: tuple[datetime, datetime], experiment: str = "ssp5_8_5", temporal_resolution: str = "daily") -> gpd.GeoDataFrame:
        ds = self.fetch_cmip6_xr(variable, model, bbox, time_range, experiment, temporal_resolution)
        # convert entire dataset to DataFrame
        df = ds.to_dataframe().reset_index()
        # create geometry
        df['geometry'] = gpd.points_from_xy(df['lon'], df['lat'])
        gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
        return gdf
    
    def fetch_monthly_single_levels_cmip5_gpd(self, experiment:str, variable: Variable.CMIP5Monthly, model: str, ensemble_member: str, period: str) -> gpd.GeoDataFrame:
        file = self.fetch_monthly_single_levels_cmip5_netcdf(experiment, variable, model, ensemble_member, period)
        ds = xr.open_dataset(file)
        # convert entire dataset to DataFrame
        df = ds.to_dataframe().reset_index()
        # create geometry
        df['geometry'] = gpd.points_from_xy(df['longitude'], df['latitude'])
        gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
        return gdf
    
    def fetch_monthly_single_levels_cmip5_xr(self, experiment:str, variable: Variable.CMIP5Monthly, model: str, ensemble_member: str, period: str) -> xr.Dataset:
        file = self.fetch_monthly_single_levels_cmip5_netcdf(experiment, variable, model, ensemble_member, period)
        ds = xr.open_dataset(file)
        
        # Wrap longitude to -180 to 180
        ds = ds.assign_coords(
            lon=((ds.lon + 180) % 360) - 180
        ).sortby("lon")
                
        return ds
    
    def fetch_monthly_single_levels_cmip5_netcdf(self, experiment:str, variable: Variable.CMIP5Monthly, model: str, ensemble_member: str, period: str) -> str:
        dataset = "projections-cmip5-monthly-single-levels"
        request = {
            "experiment": experiment,
            "variable": [variable.cds_name()],
            "model": model,
            "ensemble_member": ensemble_member,
            "period": [period]
        }
        
        zip_file_path = self._execute_cds_request(dataset, request)
        temp_dir = tempfile.gettempdir()
        req_hash = sha256(str(request).encode('utf-8')).hexdigest()
        temp_file_path = os.path.join(temp_dir, f"cds_cmip5_{req_hash}.nc")
        # If file exists, remove it
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        # extract netcdf from zip and copy to temp_file_path
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
            # find the netcdf file in the extracted files
            for file_name in zip_ref.namelist():
                if file_name.endswith('.nc'):
                    extracted_path = os.path.join(temp_dir, file_name)
                    os.rename(extracted_path, temp_file_path)
                    break
        
        return temp_file_path
    
    def _execute_cds_request(self, dataset: str, request: dict, no_cache: bool = False) -> str:
        """
        Execute a CDS API request and return the path to the downloaded file.
        Args:
            dataset (str): _dataset name in CDS
            request (dict): _request parameters for CDS
            no_cache (bool, optional): If True, force re-download even if cached file exists. Defaults to False.

        Returns:
            str: Path to the downloaded file.
        """
        hashable_request = str(request) + "=>" + dataset
        request_hash = sha256(hashable_request.encode('utf-8')).hexdigest()
        # Fetch the temporary directory of the OS
        temp_dir = tempfile.gettempdir()
        temp_file_path = os.path.join(temp_dir, f"cds_request_{request_hash}.nc")
        
        # Check if file already exists to avoid re-downloading
        if not os.path.exists(temp_file_path) or no_cache:
            print(f"Downloading data from CDS for '{dataset}', this may take a while...")
            self.cds_client.retrieve(
                    dataset,
                    request
                ).download(target=temp_file_path)
        else:
            print(f"Using locally cached data from CDS for '{dataset}'.")
        
        return temp_file_path