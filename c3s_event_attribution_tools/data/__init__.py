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
    
    def _get_beacon_cache_daily_single_levels_gpd(
        self, 
        bbox: tuple[float,float,float,float], 
        time_ranges: list[tuple[datetime,datetime]],
        variable: Variable.ERA5DailySingleLevel) -> gpd.GeoDataFrame | None:
        if self.beacon_cache is None:
            print("Beacon Cache client not initialized")
            return None
        
        adjusted_bboxes = Conversions.convert_bbox_to_0_360(bbox)
        
        gdfs = []
        for range in time_ranges:
            for adj_bbox in adjusted_bboxes:
                gdf = self.beacon_cache.fetch_from_era5_daily_single_levels_gpd(adj_bbox, range, variable)
                gdfs.append(gdf)
        
        return gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))
    
    def _get_beacon_cache_daily_single_levels_xr(
        self,
        bbox: tuple[float,float,float,float],
        time_ranges: list[tuple[datetime,datetime]],
        variable: Variable.ERA5DailySingleLevel) -> xr.Dataset | None:
        if self.beacon_cache is None:
            print("Beacon Cache client not initialized")
            return None
        
        dss = []
        adjusted_bboxes = Conversions.convert_bbox_to_0_360(bbox)
        for range in time_ranges:
            adjusted_dss = []
            for adj_bbox in adjusted_bboxes:
                ds = self.beacon_cache.fetch_from_era5_daily_single_levels_xr(adj_bbox, range, variable)
                adjusted_dss.append(ds)
                
            # Merge adjusted_dss along longitude dimension
            dss.append(xr.merge(adjusted_dss, join="outer"))

        return xr.concat(dss, dim='valid_time') # type: ignore
    
    def _get_cds_daily_data_single_levels_gpd(
        self,
        bbox:tuple[float, float, float, float],
        time_ranges: list[tuple[datetime, datetime]],
        variable: Variable.ERA5DailySingleLevel,
        ) -> gpd.GeoDataFrame:
        return self.cds_client.fetch_data_daily_single_levels_gpd(bbox=bbox, time_ranges=time_ranges, variable=variable)
    
    def _get_cds_daily_data_single_levels_xr(
        self,
        bbox:tuple[float, float, float, float],
        time_ranges: list[tuple[datetime, datetime]],
        variable: Variable.ERA5DailySingleLevel,
        ) -> xr.Dataset:
        return self.cds_client.fetch_data_daily_single_levels_xr(bbox=bbox, time_ranges=time_ranges, variable=variable)

    def _get_daily_data_single_levels_gpd(
        self,
        bbox:tuple[float, float, float, float],
        time_ranges: list[tuple[datetime, datetime]],
        variable: Variable.ERA5DailySingleLevel,
        ) -> gpd.GeoDataFrame | None:
        
        all_gdfs = []
        for time_range in time_ranges:
            print(f"Fetching data for range: {time_range[0]} - {time_range[1]}")
            
            gdfs = []
            min_retrieved_time = None
            max_retrieved_time = None
            if self.beacon_cache is not None:
                gdf = self._get_beacon_cache_daily_single_levels_gpd(bbox, [time_range], variable)
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
                    gdf_cds = self.cds_client.fetch_data_daily_single_levels_gpd(bbox=bbox, time_ranges=[(fetch_start, fetch_end)], variable=variable)
                    gdf_cds = gdf_cds.rename(columns=variable.cds_variable_renames())
                    max_retrieved_time = gdf_cds['valid_time'].max()
                    gdfs.append(gdf_cds)
                
                if max_retrieved_time is None or max_retrieved_time < time_range[1]:
                    # Request missing data from CDS
                    fetch_start = max_retrieved_time if max_retrieved_time is not None else time_range[0]
                    fetch_end = time_range[1]
                    print(f"Fetching missing data from CDS for range: {fetch_start} - {fetch_end}")
                    gdf_cds = self.cds_client.fetch_data_daily_single_levels_gpd(bbox=bbox, time_ranges=[(fetch_start, fetch_end)], variable=variable)
                    gdf_cds = gdf_cds.rename(columns=variable.cds_variable_renames())
                    gdfs.append(gdf_cds)
        
            all_gdfs.append(gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True)))
            
        # ToDo re-implement MARS data fetching if needed
            
        return gpd.GeoDataFrame(pd.concat(all_gdfs, ignore_index=True))
        
    def _get_daily_data_single_levels_xr(
        self,
        bbox:tuple[float, float, float, float],
        time_ranges: list[tuple[datetime, datetime]],
        variable: Variable.ERA5DailySingleLevel,
        ) -> xr.Dataset | None:
        
        all_dss = []
        for time_range in time_ranges:
            
            print(f"Fetching data for range: {time_range[0]} - {time_range[1]}")
            
            dss = []
            min_retrieved_time = None
            max_retrieved_time = None
            if self.beacon_cache is not None:
                ds = self._get_beacon_cache_daily_single_levels_xr(bbox, [time_range], variable)
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
                    ds_cds = self.cds_client.fetch_data_daily_single_levels_xr(bbox=bbox, time_ranges=[(fetch_start, fetch_end)], variable=variable)
                    max_retrieved_time = ds_cds['valid_time'].values.max()
                    dss.append(ds_cds)
                
                if max_retrieved_time is None or max_retrieved_time < np.datetime64(time_range[1]):
                    # Request missing data from CDS
                    fetch_start = pd.to_datetime(max_retrieved_time).to_pydatetime() if max_retrieved_time is not None else time_range[0]
                    fetch_end = time_range[1]
                    print(f"Fetching missing data from CDS for range: {fetch_start} - {fetch_end}")
                    ds_cds = self.cds_client.fetch_data_daily_single_levels_xr(bbox=bbox, time_ranges=[(fetch_start, fetch_end)], variable=variable)
                    dss.append(ds_cds)
        
            all_dss.append(xr.concat(dss, dim='valid_time')) # type: ignore
        
        return xr.concat(all_dss, dim='valid_time') # type: ignore
    
    def _get_beacon_cache_daily_pressure_levels_gpd(
        self,
        bbox:tuple[float, float, float, float],
        time_ranges: list[tuple[datetime, datetime]],
        variable: Variable.ERA5DailyPressureLevels,
        levels: list[int],
        ) -> gpd.GeoDataFrame | None:
        if self.beacon_cache is None:
            print("Beacon Cache client not initialized")
            return None
        
        adjusted_bboxes = Conversions.convert_bbox_to_0_360(bbox)
        
        gdfs = []
        for range in time_ranges:
            for adj_bbox in adjusted_bboxes:
                gdf = self.beacon_cache.fetch_from_era5_daily_pressure_levels_gpd(adj_bbox, range, variable.beacon_name(), levels)
                gdfs.append(gdf)
        
        return gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))
    
    def _get_beacon_cache_daily_pressure_levels_xr(
        self,
        bbox:tuple[float, float, float, float],
        time_ranges: list[tuple[datetime, datetime]],
        variable: Variable.ERA5DailyPressureLevels,
        levels: list[int],
        ) -> xr.Dataset | None:
        if self.beacon_cache is None:
            print("Beacon Cache client not initialized")
            return None
        
        adjusted_bboxes = Conversions.convert_bbox_to_0_360(bbox)
        
        dss = []
        for range in time_ranges:
            adjusted_dss = []
            for adj_bbox in adjusted_bboxes:
                ds = self.beacon_cache.fetch_from_era5_daily_pressure_levels_xr(adj_bbox, range, variable.beacon_name(), levels)
                adjusted_dss.append(ds)
            # Merge adjusted_dss along longitude dimension
            dss.append(xr.merge(adjusted_dss, join="outer"))
        
        return xr.concat(dss, dim='valid_time') # type: ignore
    
    def _get_cds_daily_data_pressure_levels_gpd(
        self,
        bbox:tuple[float, float, float, float],
        time_ranges: list[tuple[datetime, datetime]],
        variable: Variable.ERA5DailyPressureLevels,
        levels: list[int],
        ) -> gpd.GeoDataFrame:
        return self.cds_client.fetch_data_daily_pressure_levels_gpd(bbox=bbox, time_ranges=time_ranges, variable=variable, levels=levels)
    
    def _get_cds_daily_data_pressure_levels_xr(
        self,
        bbox:tuple[float, float, float, float],
        time_ranges: list[tuple[datetime, datetime]],
        variable: Variable.ERA5DailyPressureLevels,
        levels: list[int],
        ) -> xr.Dataset:
        return self.cds_client.fetch_data_daily_pressure_levels_xr(bbox=bbox, time_ranges=time_ranges, variable=variable, levels=levels)
    
    def _get_daily_data_pressure_levels_gpd(
        self,
        bbox:tuple[float, float, float, float],
        time_ranges: list[tuple[datetime, datetime]],
        variable: Variable.ERA5DailyPressureLevels,
        levels: list[int],
        ) -> gpd.GeoDataFrame | None:
        all_gdfs = []
        for time_range in time_ranges:
            print(f"Fetching data for range: {time_range[0]} - {time_range[1]}")
            
            gdfs = []
            min_retrieved_time = None
            max_retrieved_time = None
            if self.beacon_cache is not None:
                gdf = self._get_beacon_cache_daily_pressure_levels_gpd(bbox, [time_range], variable, levels)
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
                    gdf_cds = self.cds_client.fetch_data_daily_pressure_levels_gpd(bbox=bbox, time_ranges=[(fetch_start, fetch_end)], variable=variable, levels=levels)
                    max_retrieved_time = gdf_cds['valid_time'].max()
                    gdfs.append(gdf_cds)
                
                if max_retrieved_time is None or max_retrieved_time < time_range[1]:
                    # Request missing data from CDS
                    fetch_start = max_retrieved_time if max_retrieved_time is not None else time_range[0]
                    fetch_end = time_range[1]
                    print(f"Fetching missing data from CDS for range: {fetch_start} - {fetch_end}")
                    gdf_cds = self.cds_client.fetch_data_daily_pressure_levels_gpd(bbox=bbox, time_ranges=[(fetch_start, fetch_end)], variable=variable, levels=levels)
                    gdfs.append(gdf_cds)
        
            all_gdfs.append(gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True)))
            
        # ToDo re-implement MARS data fetching if needed
        return gpd.GeoDataFrame(pd.concat(all_gdfs, ignore_index=True))
    
    def _get_daily_data_pressure_levels_xr(
        self,
        bbox:tuple[float, float, float, float],
        time_ranges: list[tuple[datetime, datetime]],
        variable: Variable.ERA5DailyPressureLevels,
        levels: list[int],
        ) -> xr.Dataset | None:
                
        all_dss = []
        for time_range in time_ranges:
            
            print(f"Fetching data for range: {time_range[0]} - {time_range[1]}")
            
            dss = []
            min_retrieved_time = None
            max_retrieved_time = None
            if self.beacon_cache is not None:
                ds = self._get_beacon_cache_daily_pressure_levels_xr(bbox, [time_range], variable, levels)
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
                    ds_cds = self.cds_client.fetch_data_daily_pressure_levels_xr(bbox=bbox, time_ranges=[(fetch_start, fetch_end)], variable=variable, levels=levels)
                    max_retrieved_time = ds_cds['valid_time'].values.max()
                    dss.append(ds_cds)
                
                if max_retrieved_time is None or max_retrieved_time < np.datetime64(time_range[1]):
                    # Request missing data from CDS
                    fetch_start = pd.to_datetime(max_retrieved_time).to_pydatetime() if max_retrieved_time is not None else time_range[0]
                    fetch_end = time_range[1]
                    print(f"Fetching missing data from CDS for range: {fetch_start} - {fetch_end}")
                    ds_cds = self.cds_client.fetch_data_daily_pressure_levels_xr(bbox=bbox, time_ranges=[(fetch_start, fetch_end)], variable=variable, levels=levels)
                    dss.append(ds_cds)
        
            all_dss.append(xr.concat(dss, dim='valid_time')) # type: ignore
        
        return xr.concat(all_dss, dim='valid_time') # type: ignore
    
    @deprecated("Use fetch_data_daily_single_levels instead")
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
        gdf = self._get_daily_data_single_levels_gpd(bbox, time_ranges, Variable.ERA5DailySingleLevel.temperature_2m_mean)
        if gdf is not None and not gdf.empty:
            gdf['t2m'] = Conversions.convert_temperature(gdf['t2m'], from_unit, to_unit)
        return gdf
    
    @deprecated("Use fetch_data_daily_single_levels_xr instead")
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
        ds = self._get_daily_data_single_levels_xr(bbox, time_ranges, Variable.ERA5DailySingleLevel.temperature_2m_mean)
        if ds is not None and 't2m' in ds:
            ds['t2m'].values = Conversions.convert_temperature(ds['t2m'].values, from_unit, to_unit)
        return ds
    
    @deprecated("Use fetch_data_daily_single_levels instead")    
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
        gdf = self._get_daily_data_single_levels_gpd(bbox, time_ranges, Variable.ERA5DailySingleLevel.temperature_2m_min)
        if gdf is not None and not gdf.empty:
            gdf['t2m_min'] = Conversions.convert_temperature(gdf['t2m_min'], from_unit, to_unit)
        return gdf
    
    @deprecated("Use fetch_data_daily_single_levels_xr instead")
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
        ds = self._get_daily_data_single_levels_xr(bbox, time_ranges, Variable.ERA5DailySingleLevel.temperature_2m_min)
        if ds is not None and 't2m_min' in ds:
            ds['t2m_min'].values = Conversions.convert_temperature(ds['t2m_min'].values, from_unit, to_unit)
        return ds
    
    @deprecated("Use fetch_data_daily_single_levels instead")
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
        gdf = self._get_daily_data_single_levels_gpd(bbox, time_ranges, Variable.ERA5DailySingleLevel.temperature_2m_max)
        if gdf is not None and not gdf.empty:
            gdf['t2m_max'] = Conversions.convert_temperature(gdf['t2m_max'], from_unit, to_unit)
        return gdf
    
    @deprecated("Use fetch_data_daily_single_levels_xr instead")
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
        ds = self._get_daily_data_single_levels_xr(bbox, time_ranges, Variable.ERA5DailySingleLevel.temperature_2m_max)
        if ds is not None and 't2m_max' in ds:
            ds['t2m_max'].values = Conversions.convert_temperature(ds['t2m_max'].values, from_unit, to_unit)
        return ds
    
    @deprecated("Use fetch_data_daily_single_levels instead")
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
        gdf = self._get_daily_data_single_levels_gpd(bbox, time_ranges, Variable.ERA5DailySingleLevel.total_precipitation)
        return gdf
    
    @deprecated("Use fetch_data_daily_single_levels_xr instead")
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
        ds = self._get_daily_data_single_levels_xr(bbox, time_ranges, Variable.ERA5DailySingleLevel.total_precipitation)
        return ds

    @deprecated("Use fetch_data_daily_single_levels instead")
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
        return self.cds_client.fetch_data_daily_single_levels_gpd(bbox=bbox, time_ranges=[time_range], variable=Variable.ERA5DailySingleLevel.mean_sea_level_pressure)
    
    @deprecated("Use fetch_data_daily_single_levels_xr instead")
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
        return self.cds_client.fetch_data_daily_single_levels_xr(bbox=bbox, time_ranges=[time_range], variable=Variable.ERA5DailySingleLevel.mean_sea_level_pressure)

    @deprecated("Use fetch_data_daily_single_levels instead")
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
        return self.cds_client.fetch_data_daily_pressure_levels_gpd(bbox=bbox, time_ranges=[time_range], variable=Variable.ERA5DailyPressureLevels.geopotential, levels=[500])
    
    @deprecated("Use fetch_data_daily_single_levels_xr instead")
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
        return self.cds_client.fetch_data_daily_pressure_levels_xr(Variable.ERA5DailyPressureLevels.geopotential, bbox, time_ranges=[time_range], levels=[500])
    
    def fetch_era5_daily_single_levels(self, variable: Variable.ERA5DailySingleLevel, bbox: tuple[float,float,float,float], time_ranges: list[tuple[datetime,datetime]], from_unit:str|None = None, to_unit:str|None = None) -> gpd.GeoDataFrame | None:
        gdf = self._get_daily_data_single_levels_gpd(bbox, time_ranges, variable)
        
        if from_unit and to_unit and gdf is not None and not gdf.empty:
            # Convert units based on variable type. 
            # ToDo: Implement a more general conversion mechanism for other variable types
            gdf[variable.column_name()] = Conversions.convert_unit(gdf[variable.column_name()], from_unit, to_unit)
            
        return gdf
    
    def fetch_era5_daily_single_levels_xr(self, variable: Variable.ERA5DailySingleLevel, bbox: tuple[float,float,float,float], time_ranges: list[tuple[datetime,datetime]], from_unit:str|None = None, to_unit:str|None = None) -> xr.Dataset | None:
        ds = self._get_daily_data_single_levels_xr(bbox, time_ranges, variable)
        
        if from_unit and to_unit and ds is not None and variable.column_name() in ds:
            # Convert units based on variable type.
            # ToDo: Implement a more general conversion mechanism for other variable types
            ds[variable.column_name()].values = Conversions.convert_unit(ds[variable.column_name()].values, from_unit, to_unit)
        return ds
    
    def fetch_era5_daily_pressure_levels(self, variable: Variable.ERA5DailyPressureLevels, bbox: tuple[float,float,float,float], time_ranges: list[tuple[datetime,datetime]], levels: list[int], from_unit:str|None = None, to_unit:str|None = None) -> gpd.GeoDataFrame | None:
        gdf = self._get_daily_data_pressure_levels_gpd(variable=variable, bbox=bbox, time_ranges=time_ranges, levels=levels)
        if from_unit and to_unit and gdf is not None and not gdf.empty:
            # Convert units based on variable type.
            # ToDo: Implement a more general conversion mechanism for other variable types
            gdf[variable.column_name()] = Conversions.convert_temperature(gdf[variable.column_name()], from_unit, to_unit)
        return gdf
    
    def fetch_era5_daily_pressure_levels_xr(self, variable: Variable.ERA5DailyPressureLevels, bbox: tuple[float,float,float,float], time_ranges: list[tuple[datetime,datetime]], levels: list[int], from_unit:str|None = None, to_unit:str|None = None) -> xr.Dataset | None:
        ds = self._get_daily_data_pressure_levels_xr(variable=variable, bbox=bbox, time_ranges=time_ranges, levels=levels)
        if from_unit and to_unit and ds is not None and variable.column_name() in ds:
            # Convert units based on variable type.
            # ToDo: Implement a more general conversion mechanism for other variable types
            ds[variable.column_name()].values = Conversions.convert_temperature(ds[variable.column_name()].values, from_unit, to_unit)
        return ds
    
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
            "tmean": Variable.ERA5DailySingleLevel.temperature_2m_mean,
            "tmin": Variable.ERA5DailySingleLevel.temperature_2m_min,
            "tmax": Variable.ERA5DailySingleLevel.temperature_2m_max,
            "precipitation": Variable.ERA5DailySingleLevel.total_precipitation,
            "z500": Variable.ERA5DailyPressureLevels.geopotential,
            "slp": Variable.ERA5DailySingleLevel.mean_sea_level_pressure
        }
        parameter = parameter.lower()
        if parameter not in variable_map:
            raise ValueError(f"Invalid parameter: {parameter}. Supported parameters are: {list(variable_map.keys())}")

        variable = variable_map[parameter]

        match variable:
            case Variable.ERA5DailySingleLevel.temperature_2m_mean | Variable.ERA5DailySingleLevel.temperature_2m_min | Variable.ERA5DailySingleLevel.temperature_2m_max | Variable.ERA5DailySingleLevel.total_precipitation:
                # ToDo implement month filtering
                gdf = self.fetch_era5_daily_single_levels(variable, bbox, [time_range], from_unit, to_unit)
            case Variable.ERA5DailyPressureLevels.geopotential:
                gdf = self.fetch_era5_daily_pressure_levels(variable, bbox, [time_range], levels=[500], from_unit=from_unit, to_unit=to_unit)
            case Variable.ERA5DailySingleLevel.mean_sea_level_pressure:
                gdf = self.fetch_era5_daily_single_levels(variable, bbox, [time_range], from_unit, to_unit)
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
            "tmean": Variable.ERA5DailySingleLevel.temperature_2m_mean,
            "tmin": Variable.ERA5DailySingleLevel.temperature_2m_min,
            "tmax": Variable.ERA5DailySingleLevel.temperature_2m_max,
            "precipitation": Variable.ERA5DailySingleLevel.total_precipitation,
            "z500": Variable.ERA5DailyPressureLevels.geopotential,
            "slp": Variable.ERA5DailySingleLevel.mean_sea_level_pressure
        }
        parameter = parameter.lower()
        if parameter not in variable_map:
            raise ValueError(f"Invalid parameter: {parameter}. Supported parameters are: {list(variable_map.keys())}")

        variable = variable_map[parameter]

        match variable:
            case Variable.ERA5DailySingleLevel.temperature_2m_mean | Variable.ERA5DailySingleLevel.temperature_2m_min | Variable.ERA5DailySingleLevel.temperature_2m_max | Variable.ERA5DailySingleLevel.total_precipitation:
                ds = self.fetch_era5_daily_single_levels_xr(variable, bbox, [time_range], from_unit=from_unit, to_unit=to_unit)
            case Variable.ERA5DailyPressureLevels.geopotential:
                ds = self.fetch_era5_daily_pressure_levels_xr(variable, bbox, [time_range], levels=[500], from_unit=from_unit, to_unit=to_unit)
            case Variable.ERA5DailySingleLevel.mean_sea_level_pressure:
                ds = self.fetch_era5_daily_single_levels_xr(variable, bbox, [time_range], from_unit=from_unit, to_unit=to_unit)
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
        variable: Variable.CMIP6,
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
        variable: Variable.CMIP6,
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
    
    def fetch_cmip5_monthly_single_levels_xr(self, experiment:str, variable: Variable.CMIP5Monthly, model: str, ensemble_member: str, period: str) -> xr.Dataset:
        """
        Fetch CMIP5 monthly data from CDS as an xarray Dataset.

        Parameters:
            experiment (str): CMIP5 experiment name (e.g. "historical", "rcp85").
            variable (str): CMIP5 variable name (e.g. "tasmax", "tas", "pr").
            model (str): CMIP5 model name (e.g. "access1_0").
            ensemble_member (str): Ensemble member identifier (e.g. "r1i1p1").
            period (str): Time period in the format "YYYY-MM" or "YYYY-MM to YYYY-MM".

        Returns:
            xr.Dataset: CMIP5 data for the requested selection.
        """
        return self.cds_client.fetch_monthly_single_levels_cmip5_xr(
            experiment, variable, model, ensemble_member, period
        )
    
    def fetch_cmip5_monthly_single_levels_gpd(self, experiment:str, variable: Variable.CMIP5Monthly, model: str, ensemble_member: str, period: str) -> gpd.GeoDataFrame:
        """
        Fetch CMIP5 monthly data from CDS as a GeoDataFrame.

        Parameters:
            experiment (str): CMIP5 experiment name (e.g. "historical", "rcp85").
            variable (str): CMIP5 variable name (e.g. "tasmax", "tas", "pr").
            model (str): CMIP5 model name (e.g. "access1_0").
            ensemble_member (str): Ensemble member identifier (e.g. "r1i1p1").
            period (str): Time period in the format "YYYY-MM" or "YYYY-MM to YYYY-MM".

        Returns:
            gpd.GeoDataFrame: CMIP5 data for the requested selection.
        """
        return self.cds_client.fetch_monthly_single_levels_cmip5_gpd(
            experiment, variable, model, ensemble_member, period
        )

    def fetch_climate_scenarios(self,
        analysis_type,
        models, 
        variable_name, 
        bbox, 
        hist_range=(datetime(1950, 1, 1), datetime(2005, 12, 31)),
        fut_range=(datetime(2006, 1, 1), datetime(2100, 12, 31)),
        temp_res="daily",
        max_models=None
    ):
        """
        Unified fetcher for CMIP6 and CORDEX data.
        
        Parameters:
        - analysis_type: 'cmip6' or 'cordex'
        - models: 
            If 'cmip6': List of model names (strings).
            If 'cordex': List of dictionaries containing {'hist_url', 'rcp85_url', 'driving_model'}.
        """
        
        results_local = {}
        results_gmst = {}
        processed_count = 0

        # Configuration based on Analysis Type
        if analysis_type == 'cmip6':
            exp_hist = "historical"
            exp_fut  = "ssp5_8_5"
            gmst_fetch_fn = self.fetch_cmip6_xr

            
        elif analysis_type == 'cordex':
            exp_hist = "historical"
            exp_fut  = "rcp85"
            gmst_fetch_fn = self.fetch_cmip5_monthly_single_levels_xr  # GMST comes from CMIP5 (Driver)
            
        else:
            raise ValueError("analysis_type must be 'cmip6' or 'cordex'")

        print(f"\n--- Starting {analysis_type.upper()} Processing (Exp: {exp_fut}) ---")

        gcm_map = Utils.get_gcm_cordex_to_cmip5()
        # 2. Main Loop
        for entry in models:
            if max_models and processed_count >= max_models:
                break

            model_id = None
            driving_model = None       
            if analysis_type == 'cmip6':
                model_id = entry 
                driving_model = entry
            else:
                # CORDEX Logic
                gcm = entry.get('gcm', 'UnknownGCM')
                rcm = entry.get('rcm', 'UnknownRCM')
                ensemble = entry.get('member', 'UnknownMember')
                model_id = f"{gcm}_{rcm}"
                
                # 2. Map to CMIP5 name
                if gcm in gcm_map:
                    gcm_conf = gcm_map[gcm]
                    driving_model = gcm_conf['driving_model']

                    if ensemble in gcm_conf['ensembles']:
                        ens_cfg = gcm_conf['ensembles'][ensemble]
                        hist_periods = ens_cfg['historical']  
                        rcp_periods = ens_cfg['rcp_8_5']

                    else:
                        driving_model = None

                # 3. Explicit check if the mapping failed
                if not driving_model:
                    print(f"⚠️ Skipping {model_id}: Driving GCM '{gcm}' not found in CMIP5 mapping.")
                    continue

            try:
                print(f"Processing: {model_id}...")

                # Fetch Local Variable (The Study Data)
                if analysis_type == 'cmip6':
                    # CMIP6 Local Fetch
                    variable_name_ = getattr(Variable.CMIP6, variable_name)   
                    ds_hist = self.fetch_cmip6_xr(
                        variable=variable_name_, model=model_id, bbox=bbox,
                        time_range=hist_range, experiment=exp_hist, temporal_resolution=temp_res
                    )
                    ds_fut = self.fetch_cmip6_xr(
                        variable=variable_name_, model=model_id, bbox=bbox,
                        time_range=fut_range, experiment=exp_fut, temporal_resolution=temp_res
                    )
                else:
                    # CORDEX Local Fetch (via URL)
                    ds_hist = self.fetch_cordex_xr(
                        variable=variable_name, model_url=entry["hist_url"], bbox=bbox, time_range=hist_range
                    )
                    ds_fut = self.fetch_cordex_xr(
                        variable=variable_name, model_url=entry["rcp85_url"], bbox=bbox, time_range=fut_range
                    )

                # Merge Local
                local_merged = xr.concat([ds_hist, ds_fut], dim="time", combine_attrs="override")

                # Fetch Global Mean Surface Temp (GMST)
                if driving_model is not None:
                    if analysis_type == 'cmip6':
                        print(f"   -> Fetching GMST for {driving_model}...")
                        gmst_hist = gmst_fetch_fn(
                            variable=Variable.CMIP6.near_surface_air_temperature, model=driving_model, bbox=(-180, -90, 180, 90),
                            time_range=hist_range, experiment="historical", temporal_resolution="monthly"
                        )
                        gmst_fut = gmst_fetch_fn(
                            variable=Variable.CMIP6.near_surface_air_temperature, model=driving_model, bbox=(-180, -90, 180, 90),
                            time_range=fut_range, experiment=exp_fut, temporal_resolution="monthly"
                        )
                        gmst_merged = xr.concat([gmst_hist, gmst_fut], dim="time", combine_attrs="override")
                    
                    if analysis_type == 'cordex':
                        print(f"   -> Fetching GMST for {driving_model} (CMIP5)...")
                        gmst_hist = []
                        for period in hist_periods:
                            print(f"      - Historical Period: {period}")
                            chunk = gmst_fetch_fn(
                                experiment="historical", variable=Variable.CMIP5Monthly.temperature_2m,
                                model=driving_model, ensemble_member=ensemble, period=period
                            )
                            gmst_hist.append(chunk)
                        gmst_hist = xr.concat(gmst_hist, dim="time")
                        gmst_hist = gmst_hist.convert_calendar("standard", use_cftime=False)
                        gmst_hist = gmst_hist.interpolate_na(dim="time", method="linear")
                        # subset to hist_range
                        gmst_hist = gmst_hist.sel(time=slice(hist_range[0], hist_range[1]))
                        gmst_fut = []
                        for period in rcp_periods:

                            print(f"      - RCP8.5 Period: {period}")
                            chunk = gmst_fetch_fn(
                                experiment="rcp_8_5", variable=Variable.CMIP5Monthly.temperature_2m,
                                model=driving_model, ensemble_member=ensemble, period=period
                            )
                            gmst_fut.append(chunk)
                        gmst_fut = xr.concat(gmst_fut, dim="time")
                        gmst_fut = gmst_fut.convert_calendar("standard", use_cftime=False)
                        gmst_fut = gmst_fut.interpolate_na(dim="time", method="linear")
                        # subset to fut_range
                        gmst_fut = gmst_fut.sel(time=slice(fut_range[0], fut_range[1]))
                        gmst_merged = xr.concat([gmst_hist, gmst_fut], dim="time", combine_attrs="override")
                else:
                    print(f"   ⚠️ Skipping GMST: No driving model provided.")
                    gmst_merged = None

                # Store Results 
                results_local[model_id] = local_merged
                if gmst_merged is not None:
                    results_gmst[model_id] = gmst_merged
                
                processed_count += 1
                print(f"   ✅ Success: {model_id}")

            except Exception as e:
                print(f"   ❌ Failed {model_id}: {str(e)}")
                continue

        return results_local, results_gmst