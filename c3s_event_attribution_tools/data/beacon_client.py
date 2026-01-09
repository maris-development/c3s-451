import beacon_api
import geopandas as gpd
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
import tempfile

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

    
    
    def fetch_from_era5_daily_query(self, bbox: tuple[float,float,float,float], time_range: tuple[datetime,datetime], variable: str) -> beacon_api.JSONQuery:
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

        era5_table = self.beacon_client.list_tables()['daily']
        query = (era5_table.query()
                 .add_select_column('longitude')
                 .add_select_column('latitude')
                 .add_select_column('valid_time')
                 .add_select_column(variable)
                 .add_bbox_filter('longitude','latitude', bbox)
                 .add_range_filter('valid_time', gt_eq=time_range[0], lt_eq=time_range[1]))

        return query

    def fetch_from_era5_daily_zarr_xr(self, bbox: tuple[float,float,float,float], time_range: tuple[datetime,datetime], variable: str) -> xr.Dataset:
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

        query = self.fetch_from_era5_daily_query(bbox, time_range, variable)

        ds = query.to_xarray_dataset(dimension_columns=['longitude','latitude','valid_time'], force=True)
        return ds    

    def fetch_from_era5_daily_zarr_iris(self, bbox: tuple[float,float,float,float], time_range: tuple[datetime,datetime], variable: str):
        """
        Fetch data from the ERA5 Daily Zarr dataset in Beacon Cache as an Iris cube.

        This method executes a Beacon API query, saves the resulting data to a 
        temporary NetCDF file with defined spatial and temporal dimensions, and 
        loads it as an Iris cube. Note that this functionality is restricted to 
        Linux environments due to Iris library dependencies.

        Parameters:
            bbox (tuple[float, float, float, float]):
                Bounding box coordinates as (min_longitude, min_latitude, max_longitude, max_latitude).
            time_range (tuple[datetime, datetime]):
                A tuple containing the (start_date, end_date) for the data request.
            variable (str):
                The variable name to fetch. Available options include: 
                't2m', 't2m_max', 't2m_tmin', and 'total_precipitation'.

        Returns:
            iris.cube.Cube: An Iris cube containing the fetched daily data with 
            spatial and temporal coordinates.

        Raises:
            RuntimeError: If the method is called on a non-Linux platform.
        """
        
        # Check platform
        if __import__('sys').platform not in ['linux']:
            raise RuntimeError("Iris cubes is only supported on Linux platforms.")

        query = self.fetch_from_era5_daily_query(bbox, time_range, variable)
        
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name
        
        query.to_nd_netcdf(temp_path, ['longitude','latitude','valid_time'], force=True)

        iris_cube = iris.iris.load_cube(temp_path) # type: ignore # we have checked platform above
        return iris_cube

    def fetch_from_era5_daily_zarr_gpd(self, bbox: tuple[float,float,float,float], time_range: tuple[datetime,datetime], variable: str) -> gpd.GeoDataFrame:
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
        query = self.fetch_from_era5_daily_query(bbox, time_range, variable)

        df = query.to_pandas_dataframe()
        
        if df.empty:
            return gpd.GeoDataFrame(columns=['longitude', 'latitude', 'valid_time', variable], geometry=gpd.points_from_xy([], []), crs='EPSG:4326')

        return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs='EPSG:4326')
