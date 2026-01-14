from .beacon_client import *
from .cds_client import *
from .cordex_client import *
from .conversions import *
from .variable import *

try:
    # Available in Python 3.12+ as per PEP 702
    from warnings import deprecated
except ImportError:
    # Backport available in typing_extensions for older versions
    from typing_extensions import deprecated

if __import__('sys').platform in ['linux']:
    from .mars_client import *
    import iris # type: ignore

class DataClient():
    """
        DataClient is the main class that provides methods to fetch and process climate data
        from various sources including the Climate Data Store (CDS), Beacon Cache, and MARS.
        
        The other classes e.g. CDSClient, MarsClient etc. are used in this client class.
        
    """
    def __init__(self, cds_key: str, beacon_cache_url: str | None = None, beacon_token: str | None = None, mars_key: str | None = None, cordex_arco_token : str | None = None) -> None:
        """
        Instantiate the DataClient by calling the constructor.
        
        Parameters:
            cds_key (str): The API key for the Climate Data Store (CDS).
            beacon_cache_url (str | None): Optional URL for the Beacon Cache service.
            beacon_token (str | None): Optional token for accessing the Beacon Cache.
            mars_key (str | None): Optional API key for accessing MARS data.
        """
        self.cds_client = CDSClient(cds_key)
        self.beacon_cache = None
        self.mars_client = None
        self.cordex_client = None
        if beacon_cache_url:
            self.beacon_cache = BeaconClient(beacon_cache_url=beacon_cache_url, beacon_token=beacon_token)
        if mars_key:
            self.mars_client = MarsClient(key=mars_key)  # type: ignore for mars on non-linux platforms
        if cordex_arco_token:
            self.cordex_client = CordexClient(cordex_token=cordex_arco_token)
    
    def get_beacon_cache_daily_gpd(
        self, 
        bbox: tuple[float,float,float,float], 
        time_ranges: list[tuple[datetime,datetime]],
        variable: Variable) -> gpd.GeoDataFrame | None:
        if self.beacon_cache is None:
            print("Beacon Cache client not initialized")
            return None
        
        adjusted_bboxes = Conversions.convert_bbox_to_0_360(bbox)
        
        gdfs = []
        for range in time_ranges:
            for adj_bbox in adjusted_bboxes:
                gdf = self.beacon_cache.fetch_from_era5_daily_zarr_gpd(adj_bbox, range, variable.beacon_name())
                gdfs.append(gdf)
        
        return gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))
    
    def get_beacon_cache_daily_xr(
        self,
        bbox: tuple[float,float,float,float],
        time_ranges: list[tuple[datetime,datetime]],
        variable: Variable) -> xr.Dataset | None:
        if self.beacon_cache is None:
            print("Beacon Cache client not initialized")
            return None
        
        dss = []
        adjusted_bboxes = Conversions.convert_bbox_to_0_360(bbox)
        for range in time_ranges:
            for adj_bbox in adjusted_bboxes:
                ds = self.beacon_cache.fetch_from_era5_daily_zarr_xr(adj_bbox, range, variable.beacon_name())
                dss.append(ds)

        return xr.concat(dss, dim='time') # type: ignore
    
    def get_cds_daily_data_gpd(
        self,
        bbox:tuple[float, float, float, float],
        time_ranges: list[tuple[datetime, datetime]],
        variable: Variable,
        ) -> gpd.GeoDataFrame:
        
        gdfs = []
        for time_range in time_ranges:
            print(f"Fetching data from CDS for range: {time_range[0]} - {time_range[1]}")
            gdf = self.cds_client.fetch_data_single_levels_gpd("derived-era5-single-levels-daily-statistics", bbox=bbox, time_range=time_range, variable=variable)
            gdfs.append(gdf)
        
        return gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))
    
    def get_cds_daily_data_xr(
        self,
        bbox:tuple[float, float, float, float],
        time_ranges: list[tuple[datetime, datetime]],
        variable: Variable,
        ) -> xr.Dataset:

        dss = []
        for time_range in time_ranges:
            print(f"Fetching data from CDS for range: {time_range[0]} - {time_range[1]}")
            ds = self.cds_client.fetch_data_single_levels_xr("derived-era5-single-levels-daily-statistics", bbox=bbox, time_range=time_range, variable=variable)
            dss.append(ds)
        return xr.concat(dss, dim='time') # type: ignore

    def get_daily_data_gpd(
        self,
        bbox:tuple[float, float, float, float],
        time_ranges: list[tuple[datetime, datetime]],
        variable: Variable,
        ) -> gpd.GeoDataFrame | None:
        
        all_gdfs = []
        for time_range in time_ranges:
            print(f"Fetching data for range: {time_range[0]} - {time_range[1]}")
            
            gdfs = []
            min_retrieved_time = None
            max_retrieved_time = None
            if self.beacon_cache is not None:
                gdf = self.get_beacon_cache_daily_gpd(bbox, [time_range], variable)
                if gdf is not None and not gdf.empty:
                    gdfs.append(gdf)
                    min_retrieved_time = gdf['valid_time'].min()
                    max_retrieved_time = gdf['valid_time'].max()
            
            if min_retrieved_time is None or max_retrieved_time is None or min_retrieved_time > time_range[0] or max_retrieved_time < time_range[1]:
                # No valid data found in beacon cache or we are missing data for the requested time range
                print("Missing data in beacon cache, fetching missing data from CDS...")

                if min_retrieved_time is None or min_retrieved_time > time_range[0]:
                    # Request missing data from CDS
                    fetch_start = time_range[0]
                    fetch_end = min_retrieved_time if min_retrieved_time is not None else time_range[1]
                    print(f"Fetching missing data from CDS for range: {fetch_start} - {fetch_end}")
                    gdf_cds = self.cds_client.fetch_data_single_levels_gpd("derived-era5-single-levels-daily-statistics", bbox=bbox, time_range=(fetch_start, fetch_end), variable=variable)
                    gdf_cds = gdf_cds.rename(columns=variable.cds_variable_renames())
                    max_retrieved_time = gdf_cds['valid_time'].max()
                    gdfs.append(gdf_cds)
                
                if max_retrieved_time is None or max_retrieved_time < time_range[1]:
                    # Request missing data from CDS
                    fetch_start = max_retrieved_time if max_retrieved_time is not None else time_range[0]
                    fetch_end = time_range[1]
                    print(f"Fetching missing data from CDS for range: {fetch_start} - {fetch_end}")
                    gdf_cds = self.cds_client.fetch_data_single_levels_gpd("derived-era5-single-levels-daily-statistics", bbox=bbox, time_range=(fetch_start, fetch_end), variable=variable)
                    gdf_cds = gdf_cds.rename(columns=variable.cds_variable_renames())
                    gdfs.append(gdf_cds)
        
            all_gdfs.append(gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True)))
            
        # ToDo re-implement MARS data fetching if needed
            
        return gpd.GeoDataFrame(pd.concat(all_gdfs, ignore_index=True))
        
    def get_daily_data_xr(
        self,
        bbox:tuple[float, float, float, float],
        time_ranges: list[tuple[datetime, datetime]],
        variable: Variable,
        ) -> xr.Dataset | None:
        
        all_dss = []
        for time_range in time_ranges:
            
            print(f"Fetching data for range: {time_range[0]} - {time_range[1]}")
            
            dss = []
            min_retrieved_time = None
            max_retrieved_time = None
            if self.beacon_cache is not None:
                ds = self.get_beacon_cache_daily_xr(bbox, [time_range], variable)
                if ds is not None and 'valid_time' in ds:
                    dss.append(ds)
                    min_retrieved_time = ds['valid_time'].values.min()
                    max_retrieved_time = ds['valid_time'].values.max()
            
            if min_retrieved_time is None or max_retrieved_time is None or min_retrieved_time > np.datetime64(time_range[0]) or max_retrieved_time < np.datetime64(time_range[1]):
                # No valid data found in beacon cache or we are missing data for the requested time range
                print("Missing data in beacon cache, fetching missing data from CDS...")

                if min_retrieved_time is None or min_retrieved_time > np.datetime64(time_range[0]):
                    # Request missing data from CDS
                    fetch_start = time_range[0]
                    fetch_end = pd.to_datetime(min_retrieved_time).to_pydatetime() if min_retrieved_time is not None else time_range[1]
                    print(f"Fetching missing data from CDS for range: {fetch_start} - {fetch_end}")
                    ds_cds = self.cds_client.fetch_data_single_levels_xr("derived-era5-single-levels-daily-statistics", bbox=bbox, time_range=(fetch_start, fetch_end), variable=variable)
                    dss.append(ds_cds)
                
                if max_retrieved_time is None or max_retrieved_time < np.datetime64(time_range[1]):
                    # Request missing data from CDS
                    fetch_start = pd.to_datetime(max_retrieved_time).to_pydatetime() if max_retrieved_time is not None else time_range[0]
                    fetch_end = time_range[1]
                    print(f"Fetching missing data from CDS for range: {fetch_start} - {fetch_end}")
                    ds_cds = self.cds_client.fetch_data_single_levels_xr("derived-era5-single-levels-daily-statistics", bbox=bbox, time_range=(fetch_start, fetch_end), variable=variable)
                    dss.append(ds_cds)
        
            all_dss.append(xr.concat(dss, dim='valid_time')) # type: ignore
        
        return xr.concat(all_dss, dim='valid_time') # type: ignore
    
    def temperature_2m_mean_gpd(self, bbox: tuple[float,float,float,float], time_ranges: list[tuple[datetime, datetime]], from_unit: str = "k", to_unit:str = "c") -> gpd.GeoDataFrame | None:
        """
        Fetch mean 2m temperature data for a specified bounding box and time ranges.

        This method retrieves daily mean 2m temperature data from the ERA5
        single levels daily statistics dataset via the CDS client, and converts
        the temperature units as specified.

        Parameters:
            bbox (tuple[float, float, float, float]):
                Bounding box coordinates as (min_longitude, min_latitude, max_longitude, max_latitude).
            time_ranges (list[tuple[datetime, datetime]]):
                List of time ranges as tuples of (start_date, end_date) defining the periods for which to fetch data.
            from_unit (str):
                The unit of the fetched temperature data. Default is "k" (Kelvin).
            to_unit (str):
                The desired unit for the temperature data. Default is "c" (Celsius).    
        Returns:
            gpd.GeoDataFrame: A GeoDataFrame containing mean 2m temperature data with
            spatial and temporal information.
        """
        gdf = self.get_daily_data_gpd(bbox, time_ranges, Variable.t2mean)
        if gdf is not None and not gdf.empty:
            gdf['t2m'] = Conversions.convert_temperature(gdf['t2m'], from_unit, to_unit)
        return gdf
    
    def temperature_2m_mean_xr(self, bbox: tuple[float,float,float,float], time_ranges: list[tuple[datetime, datetime]], from_unit: str = "k", to_unit:str = "c") -> xr.Dataset | None:
        """
        Fetch mean 2m temperature data for a specified bounding box and time ranges.

        This method retrieves daily mean 2m temperature data from the ERA5
        single levels daily statistics dataset via the CDS client, and converts
        the temperature units as specified.

        Parameters:
            bbox (tuple[float, float, float, float]):
                Bounding box coordinates as (min_longitude, min_latitude, max_longitude, max_latitude).
            time_ranges (list[tuple[datetime, datetime]]):
                List of time ranges as tuples of (start_date, end_date) defining the periods for which to fetch data.
            from_unit (str):
                The unit of the fetched temperature data. Default is "k" (Kelvin).
            to_unit (str):
                The desired unit for the temperature data. Default is "c" (Celsius).    
        Returns:
            xr.Dataset: An xarray Dataset containing mean 2m temperature data with
            spatial and temporal information.
        """
        ds = self.get_daily_data_xr(bbox, time_ranges, Variable.t2mean)
        if ds is not None and 't2m' in ds:
            ds['t2m'].values = Conversions.convert_temperature(ds['t2m'].values, from_unit, to_unit)
        return ds
        
    def temperature_2m_min_gpd(self, bbox: tuple[float,float,float,float], time_ranges: list[tuple[datetime, datetime]], from_unit: str = "k", to_unit:str = "c") -> gpd.GeoDataFrame | None:
        """
        Fetch minimum 2m temperature data for a specified bounding box and time ranges.

        This method retrieves daily minimum 2m temperature data from the ERA5
        single levels daily statistics dataset via the CDS client, and converts
        the temperature units as specified.

        Parameters:
            bbox (tuple[float, float, float, float]):
                Bounding box coordinates as (min_longitude, min_latitude, max_longitude, max_latitude).
            time_ranges (list[tuple[datetime, datetime]]):
                List of time ranges as tuples of (start_date, end_date) defining the periods for which to fetch data.
            from_unit (str):
                The unit of the fetched temperature data. Default is "k" (Kelvin).
            to_unit (str):
                The desired unit for the temperature data. Default is "c" (Celsius).    
        Returns:
            gpd.GeoDataFrame: A GeoDataFrame containing minimum 2m temperature data with
            spatial and temporal information.
        """
        gdf = self.get_daily_data_gpd(bbox, time_ranges, Variable.t2min)
        if gdf is not None and not gdf.empty:
            gdf['t2m_min'] = Conversions.convert_temperature(gdf['t2m_min'], from_unit, to_unit)
        return gdf
    
    def temperature_2m_min_xr(self, bbox: tuple[float,float,float,float], time_ranges: list[tuple[datetime, datetime]], from_unit: str = "k", to_unit:str = "c") -> xr.Dataset | None:
        """
        Fetch minimum 2m temperature data for a specified bounding box and time ranges.

        This method retrieves daily minimum 2m temperature data from the ERA5
        single levels daily statistics dataset via the CDS client, and converts
        the temperature units as specified.

        Parameters:
            bbox (tuple[float, float, float, float]):
                Bounding box coordinates as (min_longitude, min_latitude, max_longitude, max_latitude).
            time_ranges (list[tuple[datetime, datetime]]):
                List of time ranges as tuples of (start_date, end_date) defining the periods for which to fetch data.
            from_unit (str):
                The unit of the fetched temperature data. Default is "k" (Kelvin).
            to_unit (str):
                The desired unit for the temperature data. Default is "c" (Celsius).    
        Returns:
            xr.Dataset: An xarray Dataset containing minimum 2m temperature data with
            spatial and temporal information.
        """
        ds = self.get_daily_data_xr(bbox, time_ranges, Variable.t2min)
        if ds is not None and 't2m_min' in ds:
            ds['t2m_min'].values = Conversions.convert_temperature(ds['t2m_min'].values, from_unit, to_unit)
        return ds
    
    def temperature_2m_max_gpd(self, bbox: tuple[float,float,float,float], time_ranges: list[tuple[datetime, datetime]], from_unit: str = "k", to_unit:str = "c") -> gpd.GeoDataFrame | None:
        """
        Fetch maximum 2m temperature data for a specified bounding box and time ranges.

        This method retrieves daily maximum 2m temperature data from the ERA5
        single levels daily statistics dataset via the CDS client, and converts
        the temperature units as specified.

        Parameters:
            bbox (tuple[float, float, float, float]):
                Bounding box coordinates as (min_longitude, min_latitude, max_longitude, max_latitude).
            time_ranges (list[tuple[datetime, datetime]]):
                List of time ranges as tuples of (start_date, end_date) defining the periods for which to fetch data.
            from_unit (str):
                The unit of the fetched temperature data. Default is "k" (Kelvin).
            to_unit (str):
                The desired unit for the temperature data. Default is "c" (Celsius).    
        Returns:
            gpd.GeoDataFrame: A GeoDataFrame containing maximum 2m temperature data with
            spatial and temporal information.
        """
        gdf = self.get_daily_data_gpd(bbox, time_ranges, Variable.t2max)
        if gdf is not None and not gdf.empty:
            gdf['t2m_max'] = Conversions.convert_temperature(gdf['t2m_max'], from_unit, to_unit)
        return gdf
    
    def temperature_2m_max_xr(self, bbox: tuple[float,float,float,float], time_ranges: list[tuple[datetime, datetime]], from_unit: str = "k", to_unit:str = "c") -> xr.Dataset | None:
        """
        Fetch maximum 2m temperature data for a specified bounding box and time ranges.

        This method retrieves daily maximum 2m temperature data from the ERA5
        single levels daily statistics dataset via the CDS client, and converts
        the temperature units as specified.

        Parameters:
            bbox (tuple[float, float, float, float]):
                Bounding box coordinates as (min_longitude, min_latitude, max_longitude, max_latitude).
            time_ranges (list[tuple[datetime, datetime]]):
                List of time ranges as tuples of (start_date, end_date) defining the periods for which to fetch data.
            from_unit (str):
                The unit of the fetched temperature data. Default is "k" (Kelvin).
            to_unit (str):
                The desired unit for the temperature data. Default is "c" (Celsius).    
        Returns:
            xr.Dataset: An xarray Dataset containing maximum 2m temperature data with
            spatial and temporal information.
        """ 
        ds = self.get_daily_data_xr(bbox, time_ranges, Variable.t2max)
        if ds is not None and 't2m_max' in ds:
            ds['t2m_max'].values = Conversions.convert_temperature(ds['t2m_max'].values, from_unit, to_unit)
        return ds
    
    def total_precipitation_gpd(self, bbox: tuple[float,float,float,float], time_ranges: list[tuple[datetime,datetime]]) -> gpd.GeoDataFrame | None:
        """
        Fetch total precipitation data for a specified bounding box and time ranges.

        This method retrieves daily total precipitation data from the ERA5
        single levels daily statistics dataset via the CDS client.

        Parameters:
            bbox (tuple[float, float, float, float]):
                Bounding box coordinates as (min_longitude, min_latitude, max_longitude, max_latitude).
            time_ranges (list[tuple[datetime, datetime]]):
                List of time ranges as tuples of (start_date, end_date) defining the periods for which to fetch data.
        Returns:
            gpd.GeoDataFrame: A GeoDataFrame containing total precipitation data with
            spatial and temporal information.
        """
        gdf = self.get_daily_data_gpd(bbox, time_ranges, Variable.tp)
        return gdf
    
    def total_precipitation_xr(self, bbox: tuple[float,float,float,float], time_ranges: list[tuple[datetime,datetime]]) -> xr.Dataset | None:
        """
        Fetch total precipitation data for a specified bounding box and time ranges.

        This method retrieves daily total precipitation data from the ERA5
        single levels daily statistics dataset via the CDS client.

        Parameters:
            bbox (tuple[float, float, float, float]):
                Bounding box coordinates as (min_longitude, min_latitude, max_longitude, max_latitude).
            time_ranges (list[tuple[datetime, datetime]]):
                List of time ranges as tuples of (start_date, end_date) defining the periods for which to fetch data.
        Returns:
            xr.Dataset: An xarray Dataset containing total precipitation data with
            spatial and temporal information.
        """
        ds = self.get_daily_data_xr(bbox, time_ranges, Variable.tp)
        return ds

    def mean_sea_level_pressure_gpd(self, bbox: tuple[float,float,float,float], time_range: tuple[datetime,datetime]) -> gpd.GeoDataFrame:
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
        return self.cds_client.fetch_data_single_levels_gpd("derived-era5-single-levels-daily-statistics", Variable.mslp, bbox, time_range)
    
    def mean_sea_level_pressure_xr(self, bbox: tuple[float,float,float,float], time_range: tuple[datetime,datetime]) -> xr.Dataset:
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
            xr.Dataset: An xarray Dataset containing mean sea level pressure data with
            spatial and temporal information.
        """
        return self.cds_client.fetch_data_single_levels_xr("derived-era5-single-levels-daily-statistics", Variable.mslp, bbox, time_range)

    def z500_geopotential_mean_gpd(self, bbox: tuple[float,float,float,float], time_range: tuple[datetime,datetime]) -> gpd.GeoDataFrame:
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
        return self.cds_client.fetch_data_pressure_levels_gpd("derived-era5-pressure-levels-daily-statistics", Variable.geopotential, bbox, time_range, levels=[500])
    
    def z500_geopotential_mean_xr(self, bbox: tuple[float,float,float,float], time_range: tuple[datetime,datetime]) -> xr.Dataset:
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
            xr.Dataset: 
                An xarray Dataset containing the daily mean geopotential data
                at 500 hPa pressure level for the specified spatial and temporal extent.
        """
        return self.cds_client.fetch_data_pressure_levels_xr("derived-era5-pressure-levels-daily-statistics", Variable.geopotential, bbox, time_range, levels=[500])
    
    def fetch_data(self, parameter:str, bbox: tuple[float,float,float,float], time_range: tuple[datetime,datetime], from_unit:str|None = None, to_unit:str|None = None) -> gpd.GeoDataFrame | None:
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
        variable_map = {
            "tmean": Variable.t2mean,
            "tmin": Variable.t2min,
            "tmax": Variable.t2max,
            "precipitation": Variable.tp,
            "z500": Variable.geopotential,
            "slp": Variable.mslp
        }
        parameter = parameter.lower()
        if parameter not in variable_map:
            raise ValueError(f"Invalid parameter: {parameter}. Supported parameters are: {list(variable_map.keys())}")

        variable = variable_map[parameter]

        match variable:
            case Variable.t2mean | Variable.t2min | Variable.t2max | Variable.tp:
                # ToDo implement month filtering
                gdf = self.get_daily_data_gpd(bbox, [time_range], variable)
            case Variable.geopotential:
                gdf = self.z500_geopotential_mean_gpd(bbox, time_range)
            case Variable.mslp:
                gdf = self.mean_sea_level_pressure_gpd(bbox, time_range)
            case _:
                raise ValueError(f"Unsupported variable: {variable}")
        
        return gdf
    
    def fetch_data_xr(self, parameter:str, bbox: tuple[float,float,float,float], time_range: tuple[datetime,datetime], months:list[str]|list[int]|None=None, from_unit:str|None = None, to_unit:str|None = None) -> xr.Dataset | None:
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
            xr.Dataset:
                xarray Dataset containing the requested climate data.
        Raises:
            ValueError:
                Raised if an invalid `parameter` is provided, or if month values are invalid
                (integers not in 1–12, or unrecognized month names).
        """
        variable_map = {
            "tmean": Variable.t2mean,
            "tmin": Variable.t2min,
            "tmax": Variable.t2max,
            "precipitation": Variable.tp,
            "z500": Variable.geopotential,
            "slp": Variable.mslp
        }
        parameter = parameter.lower()
        if parameter not in variable_map:
            raise ValueError(f"Invalid parameter: {parameter}. Supported parameters are: {list(variable_map.keys())}")

        variable = variable_map[parameter]

        match variable:
            case Variable.t2mean | Variable.t2min | Variable.t2max | Variable.tp:
                # ToDo implement month filtering
                ds = self.get_daily_data_xr(bbox, [time_range], variable)
            case Variable.geopotential:
                # ToDo implement geopotential xr fetching
                ds = self.z500_geopotential_mean_xr(bbox, time_range)
            case Variable.mslp:
                # ToDo implement mslp xr fetching
                ds = self.mean_sea_level_pressure_xr(bbox, time_range)
            case _:
                raise ValueError(f"Unsupported variable for xarray output: {variable}")
        
        return ds
    
    def fetch_data_gpd(self, *args, **kwargs) -> gpd.GeoDataFrame | None:
        """Alias for :meth:`fetch_data` returning a GeoDataFrame.

        This exists mostly for backwards compatibility and symmetry with other
        `*_gpd` helpers.

        Returns:
            gpd.GeoDataFrame | None: Output of :meth:`fetch_data`.
        """
        return self.fetch_data(*args, **kwargs)

    def fetch_cmip6_gpd(self, *args, **kwargs) -> gpd.GeoDataFrame:
        """Alias for :meth:`fetch_cmip6` returning a GeoDataFrame.

        This wrapper exists for consistency with the `*_gpd` naming used in the
        library.

        Returns:
            gpd.GeoDataFrame: CMIP6 data for the requested variable/extent/time.
        """
        return self.fetch_cmip6(*args, **kwargs)

    def fetch_cmip6(
        self,
        variable: str,
        model: str,
        bbox: tuple[float, float, float, float],
        time_range: tuple[datetime, datetime],
        experiment: str = "ssp5_8_5",
        temporal_resolution: str = "daily",
    ) -> gpd.GeoDataFrame:
        """Fetch CMIP6 data from CDS as a GeoDataFrame.

        Parameters:
            variable (str): CMIP6 variable name (e.g. "tasmax", "tas", "pr").
            model (str): CMIP6 model name (e.g. "access_cm2").
            bbox (tuple[float, float, float, float]):
                Bounding box as (min_lon, min_lat, max_lon, max_lat).
            time_range (tuple[datetime, datetime]): (start, end) datetime range.
            experiment (str, optional): Scenario/experiment identifier.
                Defaults to "ssp5_8_5".
            temporal_resolution (str, optional): Temporal resolution (e.g. "daily").
                Defaults to "daily".

        Returns:
            gpd.GeoDataFrame: CMIP6 data for the requested selection.
        """
        return self.cds_client.fetch_cmip6_gpd(
            variable, model, bbox, time_range, experiment, temporal_resolution
        )

    def fetch_cmip6_xr(
        self,
        variable: str,
        model: str,
        bbox: tuple[float, float, float, float],
        time_range: tuple[datetime, datetime],
        experiment: str = "ssp5_8_5",
        temporal_resolution: str = "daily",
    ) -> xr.Dataset:
        """Fetch CMIP6 data from CDS as an xarray Dataset.

        Parameters:
            variable (str): CMIP6 variable name (e.g. "tasmax", "tas", "pr").
            model (str): CMIP6 model name (e.g. "access_cm2").
            bbox (tuple[float, float, float, float]):
                Bounding box as (min_lon, min_lat, max_lon, max_lat).
            time_range (tuple[datetime, datetime]): (start, end) datetime range.
            experiment (str, optional): Scenario/experiment identifier.
                Defaults to "ssp5_8_5".
            temporal_resolution (str, optional): Temporal resolution (e.g. "daily").
                Defaults to "daily".

        Returns:
            xr.Dataset: CMIP6 data for the requested selection.
        """
        return self.cds_client.fetch_cmip6_xr(
            variable, model, bbox, time_range, experiment, temporal_resolution
        )
        
    def fetch_cordex_xr(self, variable: str, model_url: str, bbox: tuple[float, float, float, float], time_range: tuple[datetime, datetime]) -> xr.Dataset:
        """Fetch CORDEX data as an xarray Dataset.

        Parameters:
            variable (str): CORDEX variable name (e.g. "tasmax", "tas", "pr").
            model_url (str): CORDEX model URL (e.g. "eur11-hist-day-cccma_canesm2-clmcom_clm_cclm4_8_17-r1i1p1").
            bbox (tuple[float, float, float, float]):
                Bounding box as (min_lon, min_lat, max_lon, max_lat).
            time_range (tuple[datetime, datetime]): (start, end) datetime range.

        Returns:
            xr.Dataset: CORDEX data for the requested selection.
        """
        if self.cordex_client is None:
            raise ValueError("CORDEX client not initialized.")
        
        return self.cordex_client.fetch_cordex_xr(
            variable=variable,
            model_url=model_url,
            bbox=bbox,
            time_range=time_range
        )
        
    def fetch_cordex(self, variable: str, model_url: str, bbox: tuple[float, float, float, float], time_range: tuple[datetime, datetime]) -> gpd.GeoDataFrame:
        """Fetch CORDEX data as a GeoDataFrame.
        
        Parameters:
            variable (str): CORDEX variable name (e.g. "tasmax", "tas", "pr").
            model_url (str): CORDEX model URL (e.g. "eur11-hist-day-cccma_canesm2-clmcom_clm_cclm4_8_17-r1i1p1").
            bbox (tuple[float, float, float, float]):
                Bounding box as (min_lon, min_lat, max_lon, max_lat).
            time_range (tuple[datetime, datetime]): (start, end) datetime range.

        Returns:
            gpd.GeoDataFrame: CORDEX data for the requested selection.
        """
        if self.cordex_client is None:
            raise ValueError("CORDEX client not initialized.")
        
        return self.cordex_client.fetch_cordex_gpd(
            variable=variable,
            model_url=model_url,
            bbox=bbox,
            time_range=time_range
        )
        
    def fetch_cordex_gpd(self, *args, **kwargs) -> gpd.GeoDataFrame:
        """Alias for :meth:`fetch_cordex` returning a GeoDataFrame.

        This wrapper exists for consistency with the `*_gpd` naming used in the
        library.

        Returns:
            gpd.GeoDataFrame: CORDEX data for the requested variable/extent/time.
        """
        return self.fetch_cordex(*args, **kwargs)
    
    def list_cordex_model_urls(self) -> list[str]:
        """List available CORDEX model URLs.

        Returns:
            list[str]: List of available CORDEX model URLs.
        """
        if self.cordex_client is None:
            raise ValueError("CORDEX client not initialized.")
        
        return self.cordex_client.list_available_models()

    @deprecated("Use `self.fetch_data()` instead.")
    def GET(self, *args, **kwargs):
        return self.fetch_data(*args, **kwargs)
    

    def gmst(self, time_range: tuple[datetime,datetime], from_unit:str = "k", to_unit:str = "c", bbox: tuple[float,float,float,float] | None = None) -> gpd.GeoDataFrame:
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
        # 
        if bbox is None:
            bbox = (-180.0, -90.0, 180.0, 90.0)
        gdf = self.cds_client._fetch_data_monthly_averaged_gpd('2m_temperature', bbox, time_range)
        # Convert temperature units
        gdf['t2m'] = Conversions.convert_temperature(gdf['t2m'], from_unit, to_unit)
        return gdf
    
    def gmst_xr(self, time_range: tuple[datetime,datetime], bbox: tuple[float,float,float,float] | None = None, from_unit:str = "k", to_unit:str = "c") -> xr.Dataset:
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
            xr.Dataset: xarray Dataset containing global mean surface temperature data
            in the specified unit.
        """
        if bbox is None:
            bbox = (-180.0, -90.0, 180.0, 90.0)
        
        ds = self.cds_client._fetch_data_monthly_averaged_xr('2m_temperature', bbox, time_range)
        # Concatenate all Datasets
        ds['t2m'].values = Conversions.convert_temperature(ds['t2m'].values, from_unit, to_unit)
        return ds # type: ignore