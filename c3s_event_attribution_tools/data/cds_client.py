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
    
    def _fetch_data_monthly_netcdf(self, variable: str, bbox: tuple[float, float, float, float], time_range: tuple[datetime, datetime]) -> str:
        """
        Fetch monthly averaged ERA5 data and save it as a temporary NetCDF file.

        This method constructs a request for the ERA5 monthly means dataset, 
        retrieves the data via the CDS client, and stores it in a temporary 
        file on disk.

        Parameters:
            variable (str):
                The name of the variable to fetch from the dataset.
            bbox (tuple[float, float, float, float]):
                Bounding box coordinates as (min_longitude, min_latitude, max_longitude, max_latitude).
            time_range (tuple[datetime, datetime]):
                A tuple containing the (start_date, end_date) for the data request.

        Returns:
            str: The file path to the downloaded temporary NetCDF file.
        """

        dataset = "reanalysis-era5-single-levels-monthly-means"
        
        req = self._build_request_monthly_averaged(variable, time_range, bbox)
        # Create a temporary directory to store the downloaded file
        with tempfile.TemporaryDirectory(delete=False) as temp_dir:
            temp_file_path = os.path.join(temp_dir, "data.nc")
            self.cds_client.retrieve(
                dataset,
                req
            ).download(target=temp_file_path)
            return temp_file_path
        
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
        file = self._fetch_data_monthly_netcdf(variable, bbox, time_range)
        ds = xr.open_dataset(file)
        # filter time to exact range
        ds = ds.sel(valid_time=slice(time_range[0], time_range[1]))
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
        file = self._fetch_data_monthly_netcdf(variable, bbox, time_range)
        # open dataset
        ds = xr.open_dataset(file)
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
    
    def _fetch_data_pressure_levels(
        self,
        dataset: str,
        variable: Variable,
        bbox: tuple[float, float, float, float],
        time_range: tuple[datetime, datetime],
        levels: list[int],
    ) -> list[str]:
        """
        Fetch data from CDS API for given variables and bbox, splitting by year.

        This method breaks the requested time range into annual sub-ranges,
        retrieves the data for each year from the CDS pressure levels dataset,
        and saves each result as a temporary NetCDF file.

        Parameters:
            dataset (str):
                The name of the CDS dataset to query.
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
        ranges = Utils.split_time_range_by_year(*time_range)
        files = []
        for start_dt, end_dt in ranges:
            req = self._build_request_pressure_levels([variable.cds_name()], bbox, (start_dt, end_dt), levels, daily_statistic=variable.cds_daily_statistic())
            with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as tmp:
                self.cds_client.retrieve(
                    dataset,
                    req
                ).download(target=tmp.name)
                files.append(tmp.name)

        return files
    
    def fetch_data_pressure_levels_gpd(
        self,
        dataset: str,
        variable: Variable,
        bbox: tuple[float, float, float, float],
        time_range: tuple[datetime, datetime],
        levels: list[int],
    ) -> gpd.GeoDataFrame:
        """
        Fetch pressure level data and return it as a combined GeoDataFrame.

        This method retrieves data for specified pressure levels across multiple years 
        if necessary, converts the resulting NetCDF files into a tabular format, 
        generates spatial point geometries, and merges them into a single 
        GeoDataFrame filtered to the requested time range.

        Parameters:
            dataset (str):
                The name of the CDS dataset to query.
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
        files = self._fetch_data_pressure_levels(
            dataset,
            variable,
            bbox,
            time_range,
            levels,
        )
        
        gdfs = []
        for file in files:
            # open dataset
            ds = xr.open_dataset(file)
            # convert entire dataset to DataFrame
            df = ds.to_dataframe().reset_index()
            # create geometry
            df['geometry'] = gpd.points_from_xy(df['longitude'], df['latitude'])
            gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
            gdfs.append(gdf)
        
        combined_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs='EPSG:4326')
        
        # filter time to exact range
        min_start, max_end = time_range
        combined_gdf = combined_gdf[(combined_gdf['valid_time'] >= min_start) & (combined_gdf['valid_time'] <= max_end)]

        return combined_gdf
    
    def fetch_data_pressure_levels_xr(
        self,
        dataset: str,
        variable: Variable,
        bbox: tuple[float, float, float, float],
        time_range: tuple[datetime, datetime],
        levels: list[int],
    ) -> xr.Dataset:
        """
        Fetch pressure level data and return it as a combined xarray Dataset.

        This method retrieves atmospheric data for specified pressure levels, 
        handling multi-year requests by fetching annual files and merging them 
        using xarray's multi-file dataset capabilities. The final dataset is 
        sliced to match the exact temporal boundaries requested.

        Parameters:
            dataset (str):
                The name of the CDS dataset to query.
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
            xr.Dataset: An xarray Dataset containing the combined and time-filtered 
            pressure level data.
        """
        files = self._fetch_data_pressure_levels(
            dataset,
            variable,
            bbox,
            time_range,
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
        
        # filter time to exact range
        ds = ds.sel(valid_time=slice(time_range[0], time_range[1]))
        
        return ds
    
    def fetch_data_pressure_levels_iris(
        self,
        dataset: str,
        variable: Variable,
        bbox: tuple[float, float, float, float],
        time_range: tuple[datetime, datetime],
        levels: list[int],
    ) -> xr.Dataset:
        """
        Fetch pressure level data and return it as an Iris cube.

        This method retrieves atmospheric data for specified pressure levels, 
        merges multi-year files into a unified dataset, and converts the 
        final result into an Iris cube. Note that this functionality is 
        restricted to Linux environments due to Iris library dependencies.

        Parameters:
            dataset (str):
                The name of the CDS dataset to query.
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
            iris.cube.Cube: An Iris cube containing the combined and 
            time-filtered pressure level data.

        Raises:
            RuntimeError: If the method is called on a non-Linux platform.
        """
        if __import__('sys').platform not in ['linux']:
            raise RuntimeError("Iris cubes is only supported on Linux platforms.")
        
        files = self._fetch_data_pressure_levels(
            dataset,
            variable,
            bbox,
            time_range,
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
        
        # filter time to exact range
        ds = ds.sel(valid_time=slice(time_range[0], time_range[1]))
        
        # Write the netcdf to local path and open
        with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as tmp:
            ds.to_netcdf(tmp.name)
            iris_cube = iris.iris.load_cube(tmp.name) # type: ignore # we have checked platform above
            return iris_cube

    def _fetch_data_single_levels(
        self,
        dataset: str,
        variable: Variable,
        bbox: tuple[float, float, float, float],
        time_range: tuple[datetime, datetime],
        months: list[int]|None = None
    ) -> list[str]:
        """
        Fetch data from the CDS API for single-level variables, splitting by year or months.

        This internal method handles the retrieval of ERA5 single-level data by 
        partitioning the requested time range into smaller chunks (annually or 
        filtered by months), performing the CDS API calls, and saving the results 
        into temporary NetCDF files.

        Parameters:
            dataset (str):
                The name of the CDS dataset to query (e.g., ERA5 daily statistics).
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
        ranges = None
        if months is not None:
            ranges = Utils.split_time_range_by_year_and_months(time_range[0], time_range[1], months)  
            
        else:
            ranges = Utils.split_time_range_by_year(*time_range)

        files = []
        for start_dt, end_dt in ranges:
            req = self._build_request_single_levels([variable.cds_name()], bbox, (start_dt, end_dt), daily_statistic=variable.cds_daily_statistic())
            with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as tmp:
                self.cds_client.retrieve(
                    dataset,
                    req
                ).download(target=tmp.name)

                files.append(tmp.name)
                
        return files
    
    def fetch_data_single_levels_gpd(
        self,
        dataset: str,
        variable: Variable,
        bbox: tuple[float, float, float, float],
        time_range: tuple[datetime, datetime],
    ) -> gpd.GeoDataFrame:
        """
        Fetch single-level data and return it as a combined GeoDataFrame.

        This method retrieves data for specified variables, handles multi-year 
        requests by fetching individual NetCDF files, converts the spatial and 
        temporal data into a tabular format with point geometries, and returns 
        a single GeoDataFrame filtered to the requested time range.

        Parameters:
            dataset (str):
                The name of the CDS dataset to query.
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
        files = self._fetch_data_single_levels(
            dataset,
            variable,
            bbox,
            time_range,
        )
        min_start, max_end = time_range
        gdfs = []
        for file in files:
            # open dataset
            ds = xr.open_dataset(file)
            # convert entire dataset to DataFrame
            df = ds.to_dataframe().reset_index()
            # create geometry
            df['geometry'] = gpd.points_from_xy(df['longitude'], df['latitude'])
            gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
            gdfs.append(gdf)
        
        combined_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs='EPSG:4326')
        
        # filter on given time range instead of taking the entire month
        combined_gdf = combined_gdf[(combined_gdf['valid_time'] >= min_start) & (combined_gdf['valid_time'] <= max_end)]
        
        return combined_gdf
    
    def fetch_data_single_levels_xr(
        self,
        dataset: str,
        variable: Variable,
        bbox: tuple[float, float, float, float],
        time_range: tuple[datetime, datetime],
    ) -> xr.Dataset:
        """
        Fetch single-level data and return it as a combined xarray Dataset.

        This method retrieves atmospheric data for specified single-level variables, 
        handling multi-year requests by fetching annual files and merging them 
        using xarray's multi-file dataset capabilities. The final dataset is 
        sliced to match the exact temporal boundaries requested.

        Parameters:
            dataset (str):
                The name of the CDS dataset to query.
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
        files = self._fetch_data_single_levels(
            dataset,
            variable,
            bbox,
            time_range,
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
        
        # filter on given time range instead of taking the entire month
        ds = ds.sel(valid_time=slice(time_range[0], time_range[1]))
        
        return ds
    
    def fetch_data_single_levels_iris(
        self,
        dataset: str,
        variable: Variable,
        bbox: tuple[float, float, float, float],
        time_range: tuple[datetime, datetime],
    ) -> xr.Dataset:
        """
        Fetch single-level data and return it as an Iris cube.

        This method retrieves atmospheric data for specified single-level variables, 
        merges multi-year files into a unified dataset, and converts the 
        final result into an Iris cube via a temporary NetCDF file. Note that 
        this functionality is restricted to Linux environments due to Iris 
        library dependencies.

        Parameters:
            dataset (str):
                The name of the CDS dataset to query.
            variable (Variable):
                An instance of the Variable class containing CDS-specific naming 
                and statistic metadata.
            bbox (tuple[float, float, float, float]):
                Bounding box coordinates as (min_longitude, min_latitude, max_longitude, max_latitude).
            time_range (tuple[datetime, datetime]):
                The full temporal range (start, end) for which to fetch data.

        Returns:
            iris.cube.Cube: An Iris cube containing the combined and 
            time-filtered single-level data.

        Raises:
            RuntimeError: If the method is called on a non-Linux platform.
        """
        if __import__('sys').platform not in ['linux']:
            raise RuntimeError("Iris cubes is only supported on Linux platforms.")
        
        files = self._fetch_data_single_levels(
            dataset,
            variable,
            bbox,
            time_range,
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
        
        # filter on given time range instead of taking the entire month
        ds = ds.sel(valid_time=slice(time_range[0], time_range[1]))
        
        # Write the netcdf to local path and open
        with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as tmp:
            ds.to_netcdf(tmp.name)
            iris_cube = iris.iris.load_cube(tmp.name) # type: ignore # we have checked platform above
            return iris_cube      

    def _build_request_cmip6(self, variable: str, model:str, time_range: tuple[datetime, datetime], bbox: tuple[float, float, float, float], experiment: str = "ssp5_8_5", temporal_resolution: str = "daily") -> dict:
        start_dt, end_dt = time_range
        # convert to pandas Timestamp for range generation
        start = pd.Timestamp(start_dt)
        end = pd.Timestamp(end_dt)
        dates = pd.date_range(start=start, end=end, freq='D')

        year = str(start.year)
        months = sorted({d.strftime('%m') for d in dates})
        days = sorted({d.strftime('%d') for d in dates})

        request = {
            "temporal_resolution": temporal_resolution,
            "experiment": experiment,
            "variable": variable,
            "model": model,
            "year": [year],
            "month": months,
            "day": days,
            "time_zone": "utc+00:00",
            "frequency": "1_hourly",
            # CDS expects [north, west, south, east]
            "area": [bbox[3], bbox[0], bbox[1], bbox[2]],
        }
        
        return request

    def _fetch_data_cmip6_netcdf(self, variable: str, model: str, bbox: tuple[float, float, float, float], time_range: tuple[datetime, datetime], experiment: str = "ssp5_8_5", temporal_resolution: str = "daily") -> str:
        dataset = "projections-cmip6"
        req = self._build_request_cmip6(variable, model, time_range, bbox, experiment, temporal_resolution)
        temp_dir = tempfile.TemporaryDirectory(delete=False)
        with temp_dir:
            zip_path = f"{temp_dir.name}/cmip6_data.zip"
            self.cds_client.retrieve(
                dataset,
                req
            ).download(target=zip_path)
            # unzip the file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir.name)
            # find the netcdf file in the extracted files
            for file_name in zip_ref.namelist():
                if file_name.endswith('.nc'):
                    return f"{temp_dir.name}/{file_name}"
            raise FileNotFoundError("No netCDF file found in the downloaded CMIP6 data.")
        
    def fetch_cmip6_xr(self, variable: str, model: str, bbox: tuple[float, float, float, float], time_range: tuple[datetime, datetime], experiment: str = "ssp5_8_5", temporal_resolution: str = "daily") -> xr.Dataset:
        file = self._fetch_data_cmip6_netcdf(variable, model, bbox, time_range, experiment, temporal_resolution)
        ds = xr.open_dataset(file)
        # filter time to exact range
        ds = ds.sel(time=slice(time_range[0], time_range[1]))
        return ds
    
    def fetch_cmip6_gpd(self, variable: str, model: str, bbox: tuple[float, float, float, float], time_range: tuple[datetime, datetime], experiment: str = "ssp5_8_5", temporal_resolution: str = "daily") -> gpd.GeoDataFrame:
        ds = self.fetch_cmip6_xr(variable, model, bbox, time_range, experiment, temporal_resolution)
        # convert entire dataset to DataFrame
        df = ds.to_dataframe().reset_index()
        # create geometry
        df['geometry'] = gpd.points_from_xy(df['longitude'], df['latitude'])
        gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
        return gdf
