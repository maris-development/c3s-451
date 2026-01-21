import beacon_api
import geopandas as gpd
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
import tempfile
from .variable import Variable

# Import iris only on supported platforms
if __import__('sys').platform in ['linux']:
    import iris # type: ignore

class BeaconClient():
    def __init__(self, beacon_cache_url: str, beacon_token: str | None = None) -> None:
        """
        Initialize the BeaconClient with connection details for the Beacon API.

        This constructor sets up the internal beacon client using a specific 
        cache URL and an optional authentication token.

        Parameters:
            beacon_cache_url (str):
                The base URL for the Beacon cache service.
            beacon_token (str | None):
                An optional JSON Web Token (JWT) for authenticated access to 
                the Beacon API. Default is None.
        """
        self.beacon_cache_url = beacon_cache_url
        self.beacon_token = beacon_token

        self.beacon_client = beacon_api.Client(
            url=self.beacon_cache_url,
            jwt_token=self.beacon_token
        )

    def fetch_from_era5_daily_single_levels_query(self, bbox: tuple[float,float,float,float], time_range: tuple[datetime,datetime], variable: Variable.ERA5DailySingleLevel) -> beacon_api.JSONQuery:
        """
        Create a query to fetch data from the ERA5 Daily Zarr dataset in Beacon Cache.

        This method constructs a structured JSON query for the Beacon API, selecting 
        spatial and temporal columns alongside the requested variable. It applies 
        spatial filtering via a bounding box and temporal filtering via the 
        provided time range.

        Parameters:
            bbox (tuple[float, float, float, float]):
                Bounding box coordinates as (min_longitude, min_latitude, max_longitude, max_latitude).
            time_range (tuple[datetime, datetime]):
                A tuple containing the (start_date, end_date) for the data query.
            variable (str):
                The variable name to fetch. Available options include: 
                't2m', 't2m_max', 't2m_tmin', and 'total_precipitation'.

        Returns:
            beacon_api.JSONQuery: A configured query object ready to be executed 
            against the Beacon Cache.
        """

        era5_table = self.beacon_client.list_tables()['daily_single_levels']
        query = (era5_table.query()
                 .add_select_column('longitude')
                 .add_select_column('latitude')
                 .add_select_column('valid_time')
                 .add_select_column(variable.beacon_name(), variable.beacon_alias())
                 .add_bbox_filter('longitude','latitude', bbox)
                 .add_range_filter('valid_time', gt_eq=time_range[0], lt_eq=time_range[1]))

        return query

    def fetch_from_era5_daily_single_levels_xr(self, bbox: tuple[float,float,float,float], time_range: tuple[datetime,datetime], variable: Variable.ERA5DailySingleLevel) -> xr.Dataset:
        """
        Fetch data from the ERA5 Daily Zarr dataset in Beacon Cache as an xarray Dataset.

        This method executes a Beacon API query for the specified variable and 
        geographic area, converting the resulting data into a multi-dimensional 
        xarray Dataset with longitude, latitude, and time as dimensions.

        Parameters:
            bbox (tuple[float, float, float, float]):
                Bounding box coordinates as (min_longitude, min_latitude, max_longitude, max_latitude).
            time_range (tuple[datetime, datetime]):
                A tuple containing the (start_date, end_date) for the data request.
            variable (str):
                The variable name to fetch. Available options include: 
                't2m', 't2m_max', 't2m_tmin', and 'total_precipitation'.

        Returns:
            xr.Dataset: An xarray Dataset containing the fetched data organized 
            by spatial and temporal dimensions.
        """

        query = self.fetch_from_era5_daily_single_levels_query(bbox, time_range, variable)

        ds = query.to_xarray_dataset(dimension_columns=['valid_time','latitude','longitude'], force=True)
        
        # wrap longitude to -180 to 180
        ds = ds.assign_coords(
            longitude=((ds.longitude + 180) % 360) - 180
        ).sortby("longitude")
        
        return ds    

    def fetch_from_era5_daily_single_levels_gpd(self, bbox: tuple[float,float,float,float], time_range: tuple[datetime,datetime], variable: Variable.ERA5DailySingleLevel) -> gpd.GeoDataFrame:
        """
        Fetch data from the ERA5 Daily Zarr dataset in Beacon Cache as a GeoDataFrame.

        This method executes a Beacon API query, converts the resulting tabular
        data into a pandas DataFrame, and then constructs a GeoDataFrame by
        generating point geometries from the longitude and latitude coordinates.

        Parameters:
            bbox (tuple[float, float, float, float]):
                Bounding box coordinates as (min_longitude, min_latitude, max_longitude, max_latitude).
            time_range (tuple[datetime, datetime]):
                A tuple containing the (start_date, end_date) for the data request.
            variable (str):
                The variable name to fetch. Available options include: 
                't2m', 't2m_max', 't2m_tmin', and 'total_precipitation'.

        Returns:
            gpd.GeoDataFrame: A GeoDataFrame containing the fetched data with 
            spatial point geometries and temporal information.
        """
        query = self.fetch_from_era5_daily_single_levels_query(bbox, time_range, variable)

        df = query.to_pandas_dataframe()
        
        if df.empty:
            return gpd.GeoDataFrame(columns=['longitude', 'latitude', 'valid_time', variable], geometry=gpd.points_from_xy([], []), crs='EPSG:4326')
        # Wrap longitude to -180 to 180
        df['longitude'] = ((df['longitude'] + 180) % 360)
        return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs='EPSG:4326')

    def fetch_from_era5_daily_pressure_levels_gpd(self, bbox: tuple[float,float,float,float], time_range: tuple[datetime,datetime], variable: str, levels: list[int]) -> gpd.GeoDataFrame:
        era5_pressure_levels_table = self.beacon_client.list_tables()['daily_pressure_levels']
        
        query = (era5_pressure_levels_table.query()
                .add_select_column('longitude')
                .add_select_column('latitude')
                .add_select_column('pressure_level')
                .add_select_column('valid_time')
                .add_select_column(variable)
                .add_bbox_filter('longitude','latitude', bbox)
                .add_range_filter('valid_time', gt_eq=time_range[0], lt_eq=time_range[1]))
        
        df = query.to_pandas_dataframe()
        if df.empty:
            return gpd.GeoDataFrame(columns=['longitude', 'latitude', 'valid_time', variable, 'pressure_level'], geometry=gpd.points_from_xy([], []), crs='EPSG:4326')
        # Wrap longitude to -180 to 180
        df['longitude'] = ((df['longitude'] + 180) % 360)
        return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs='EPSG:4326')
    
    def fetch_from_era5_daily_pressure_levels_xr(self, bbox: tuple[float,float,float,float], time_range: tuple[datetime,datetime], variable: str, levels: list[int]) -> xr.Dataset:
        era5_pressure_levels_table = self.beacon_client.list_tables()['daily_pressure_levels']
        
        query = (era5_pressure_levels_table.query()
                .add_select_column('longitude')
                .add_select_column('latitude')
                .add_select_column('pressure_level')
                .add_select_column('valid_time')
                .add_select_column(variable)
                .add_bbox_filter('longitude','latitude', bbox)
                .add_range_filter('valid_time', gt_eq=time_range[0], lt_eq=time_range[1]))
        
        ds = query.to_xarray_dataset(dimension_columns=['valid_time','latitude','longitude','pressure_level'], force=True)
        
        # wrap longitude to -180 to 180
        ds = ds.assign_coords(
            longitude=((ds.longitude + 180) % 360) - 180
        ).sortby("longitude")
        
        return ds