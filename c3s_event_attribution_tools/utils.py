from datetime import timedelta
import requests
import numpy as np
from shapely.geometry import Polygon
import webbrowser
from urllib.parse import urlencode
from typing import Dict, Any
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from shapely.geometry import Polygon, box
import base64
from io import BytesIO
from .plot import *
import os
import warnings

from typing import Dict, Union
import xarray as xr
import ipywidgets as widgets
from IPython.display import display, clear_output

class Utils:
    """Utility class for various geospatial & temporal data operations, including region selection and data manipulation, splitting time ranges, etc."""

    @staticmethod
    def get_save_directory(dir:str="data", relative:bool=True, makedir:bool=True) -> str :
        '''
        Get (and) create a directory path for saving files.

        Parameters:
            subfolder (str):
                directory name or path ('data' by default relative to the current working directory)
            relative (bool):
                whether the subfolder is relative to the current working directory (True by default)
            makedir (bool):
                whether to create the directory if it does not exist (True by default)

        Returns:
            str: The absolute path to the save directory.
        '''

        CURRENT_DIRECTORY = os.getcwd()
        your_save_directory = os.path.abspath(os.path.join(CURRENT_DIRECTORY, dir)) if relative else dir

        if makedir:
            os.makedirs(your_save_directory, exist_ok=True)

        return your_save_directory

    @staticmethod
    def split_time_range_by_year_and_months(
        start: datetime,
        end: datetime,
        months: list[str]|list[int]
    ) -> list[tuple[datetime, datetime]]:
        '''
        Split a time range into sub-ranges filtered by specific months.

        This helper method iterates through the time period between start and end,
        extracting intervals that fall within the requested months. Each resulting
        tuple represents a continuous range within a single calendar month.

        Parameters:
            start (datetime):
                The beginning of the overall time range.
            end (datetime):
                The end of the overall time range.
            months (list[str] | list[int]):
                A list of months to include, provided as integers (1-12) or 
                strings.
        
        Returns:
            list[tuple[datetime, datetime]]: A list of time ranges as tuples of 
            (start_date, end_date) defining the periods within the specified months.
        '''
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
    
    
    @staticmethod
    def split_time_range_by_year(
        start: datetime,
        end: datetime
    ) -> list[tuple[datetime, datetime]]:
        '''
        Split a time range into sub-ranges, each within a single calendar year.

        This helper method takes a start and end datetime and breaks the interval
        into a list of tuples, ensuring that no single range spans across
        multiple years. This is useful for API requests that only allow
        single-year queries.

        Parameters:
            start (datetime):
                The beginning of the time range to be split.
            end (datetime):
                The end of the time range to be split.

        Returns:
            (list[tuple[datetime, datetime]]):
                A list of (start, end) tuples, where each tuple represents a period contained within one calendar year.
        '''
        ranges = []
        current_start = start
        # iterate until within same year as end
        while current_start.year < end.year:
            year_end = datetime(year=current_start.year, month=12, day=31)
            ranges.append((current_start, year_end))
            current_start = datetime(year=current_start.year + 1, month=1, day=1)
        ranges.append((current_start, end))
        return ranges


    @staticmethod
    def select_region(regionType:str, bbox:tuple[float, float, float, float]|None=None,
                    overlays:dict[str, str]|None=None, params:Dict[str, Any]=None):
        '''
        Initiates a web-based geographical region selection tool and retrieves the selected polygon.

        This method starts the Copernicus Event Attribution Region Picker service for a specified
        type of geographical unit ('wraf' or 'hydrobasin'). It opens a URL in the user's browser,
        waits for the user to make a selection, and then polls a server endpoint until the selection
        is complete, returning the resulting GeoJSON polygon data.

        Parameters:
            regionType (str):
                The type of region layer to use for selection. Must be 'wraf' or 'hydrobasin'.
            bbox (tuple[float, float, float, float] | None, optional):
                A bounding box (min_lon, min_lat, max_lon, max_lat) to initially focus the map in the region picker. Defaults to None.
            overlays (dict[str, str] | None, optional):
                A dictionary of base64-encoded PNG images to overlay on the map in the region picker, keyed by image name. Defaults to None.
            params (Dict[str, Any], optional):
                Additional parameters to pass to the region picker service. Defaults to None.

        Returns:
            Dict[str, Any]:
                The GeoJSON polygon data of the selected region upon successful completion.

        Raises:
            ValueError:
                If an invalid `regionType` is provided.
        '''
        
        params = params if params else {}

        # add a bbox
        if bbox is not None:
            params["bbox"] = ",".join(map(str, bbox))

        if overlays is not None and len(overlays) > 0:
            params["images"] = {}
            for key, value in overlays.items():
                params["images"][key] = f'data:image/png;base64,{value}'

        allowed_region_types = ['wraf', 'hydrobasin']
        
        if regionType not in allowed_region_types:
            raise ValueError(f"Invalid regionType '{regionType}'. Allowed values are: {allowed_region_types}")
        
        print('The region picker will shortly open in your web browser. Please select a region, close the browser tab and return to the notebook when done.')
        
        url = f"https://event-attribution.copernicus-climate.eu/region-picker/start-m2m/{regionType}"

        #if params != None:
        #    url += urlencode(params)
        
        response = requests.post(url=url, json=params)
        
        poll_url = False
        
        if response.status_code == 200:
            data = response.json()
            print(f"Region Picker started successfully for {regionType}:")
            print(f"Open the following page in your browser to select a region: ")
            print(f"\t\t{data['url']}")
            webbrowser.open(data['url'])
            poll_url = data['poll_url']
        else:
            print(f"Failed to start Region Picker for {regionType}. Status code: {response.status_code}")
            print(f"Response: {response.text}")
        
        result = None
        
        if poll_url:
            print(f"Polling for region selection...")
            
            done = False
            
            while not done:
                response = requests.get(poll_url)
                if response.status_code == 200:
                    data = response.json()
                    if data['done']:
                        done = True
                        result = data['result']
                else:
                    print(f"Failed to poll Region Picker for {regionType}. Status code: {response.status_code}")
                    print(f"Response: {response.text}")
                    break
            
            print("Region selection process done.")
        
        print("Received polygon data:")
        print(result)

        return result


    @staticmethod
    def wrap_lon(ds):
        '''
        Wraps longitude coordinates from the 0° to 360° range to the standard -180° to 180° range.

        This function detects longitude coordinates (named either 'longitude' or 'lon') and transforms
        them if any value exceeds 180°. It then re-indexes the dataset to ensure the coordinates
        (both longitude and latitude) are sorted in ascending order.

        Parameters:
            ds (xarray.Dataset or xarray.DataArray): The dataset or data array containing
                longitude and latitude coordinates.

        Returns:
            xarray.Dataset|xarray.DataArray: The dataset with longitudes wrapped to -180° to 180°
                and sorted coordinates.
        '''
        if "longitude" in ds.coords:
            lon = "longitude"
            lat = "latitude"
        elif "lon" in ds.coords:
            lon = "lon"
            lat = "lat"
        else: 
            # can only wrap longitude
            return ds
        
        if ds[lon].max() > 180:
            ds[lon] = (ds[lon].dims, (((ds[lon].values + 180) % 360) - 180), ds[lon].attrs)
            
        if lon in ds.dims:
            ds = ds.reindex({ lon : np.sort(ds[lon]) })
            ds = ds.reindex({ lat : np.sort(ds[lat]) })
        return ds

    @staticmethod
    def data_2_poly(data):
        """Converts GeoJSON-like dictionary data (containing coordinates) into Shapely Polygon objects and extracts all coordinates.

        This is typically used to parse user-selected regions from a web service into usable Shapely geometry objects
        and their defining points for bounding box calculations.

        Parameters:
            data (dict):
                A GeoJSON-like dictionary object, expected to have a structure with
                `"features"`, where each feature contains a `"geometry"` with `"coordinates"`
                defining a polygon exterior ring.

        Returns:
            tuple[list[Polygon],list[list[float]]]:
                A tuple containing:
                - polygons: A list of shapely.geometry.Polygon objects.
                - all_coords: A flattened list of all [longitude, latitude] coordinate pairs
                used to define the polygons.
        """
        all_coords = []  
        polygons = []    

        for feature in data["features"]:
            coords = feature['geometry']['coordinates'][0]
            all_coords.extend(coords)  
            polygons.append(Polygon(coords))
        
        return polygons, all_coords


    @staticmethod
    def get_base_fig(date, gdf, value_col:str, datetime_col:str='valid_time', dpi:int=100, cmap=None, projection=ccrs.PlateCarree(), show_fig:bool=False, marker:str='s'):
        '''
        Generates a base map figure for a single day's data and returns it as a base64-encoded PNG image string.

        This function is intended to create a visual overlay for use in a web context (like a region picker tool).
        It subsets the GeoDataFrame for a specific date, applies a determined colormap/normalization,
        plots the data, and returns the figure output as a string instead of saving it to a file.

        Parameters:
            date (datetime.date or str):
                The specific date for which to subset and plot the data.
            gdf (gpd.GeoDataFrame):
                The GeoDataFrame containing the time series data.
            value_col (str):
                The column name containing the values to color the plot.
            datetime_col (str):
                The column name containing datetime objects for filtering. Defaults to 'valid_time'.
            dpi (int):
                Dots per inch for the figure resolution. Defaults to 100.
            cmap (str):
                The colormap identifier (e.g., 't2m', 'tp', 'anomaly') or a standard 
                Matplotlib colormap name. Defaults to None (inferred from `value_col`).
            projection (cartopy.crs):
                The Cartopy projection for the map. Defaults to ccrs.PlateCarree().
            show_fig (bool):
                If True, the Matplotlib figure is kept open and displayed (useful for
                debugging). If False, the figure is closed after encoding. Defaults to False.
            marker (str):
                The marker style to use for plotting point data. Points are converted
                to small squares/polygons for raster-like appearance. Defaults to 's' (square).

        Returns:
            str:
                A base64-encoded PNG image string of the generated plot.
        '''

        selected_gdf_anomoly = gdf[(gdf[datetime_col] >= date) & (gdf[datetime_col] <= date)]

        vmin = gdf[value_col].min()
        vmax = gdf[value_col].max()

        cmap, norm = Plot.get_colormap(cmap if cmap else value_col, vmin, vmax)


        fig, ax = plt.subplots(
            ncols = 1, nrows = 1, figsize = (5,5), dpi = dpi, 
            subplot_kw = {"projection" : projection}
        )

        # ax.plot(selected_gdf_anomoly['longitude'], selected_gdf_anomoly['latitude'], "o", markersize=1)  # markersize = diameter in points

        temp_kwargs = {"cmap" : cmap, "norm": norm}

        cell_size = 0.25  # degrees
        selected_gdf_anomoly["geometry"] = selected_gdf_anomoly.geometry.apply(
                lambda p: box(p.x - cell_size/2, p.y - cell_size/2,
                            p.x + cell_size/2, p.y + cell_size/2)
            )

        selected_gdf_anomoly.plot(ax = ax, **temp_kwargs,
            column = value_col,
            vmin = vmin,
            vmax = vmax,
            marker = marker
        )

        ax.set_axis_off()
        plt.tight_layout()

        # Save to memory buffer instead of file
        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=100, transparent=True, bbox_inches="tight", pad_inches=0)
        if not show_fig:
            plt.close(fig)  # Close the figure to avoid displaying it in non-interactive environments
        buf.seek(0)

        # Encode to base64
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        buf.close()

        return img_base64


    @staticmethod
    def add_doy_column(gdf, datetime_col:str, doy_col:str='doy') -> gpd.GeoDataFrame:   
        '''
        Adds a column to the GeoDataFrame representing the day of the year.

        The function ensures the specified datetime column is in datetime format and extracts
        the day number into a new column.

        Parameters:
            gdf (gpd.GeoDataFrame):
                The input GeoDataFrame.
            datetime_col (str):
                The column name containing datetime objects.
            doy_col (str):
                The name of the new column to hold the day of the year
                (labeled as 'doy' in the code, but extracting day number). Defaults to 'doy'.

        Returns:
            gpd.GeoDataFrame:
                A copy of the input GeoDataFrame with the new day-of-year column added.
        '''
        gdf = gdf.copy()

        gdf[datetime_col] = pd.to_datetime(gdf[datetime_col])
        gdf[doy_col] = gdf[datetime_col].dt.day

        return gdf

    @staticmethod
    def add_month_column(gdf, datetime_col:str, month_col:str='month') -> gpd.GeoDataFrame:  
        '''
        Adds a column to the GeoDataFrame representing the month number (1-12) extracted from a datetime column.

        Parameters:
            gdf (gpd.GeoDataFrame):
                The input GeoDataFrame.
            datetime_col (str):
                The column name containing datetime objects.
            month_col (str, optional):
                The name of the new column to hold the month number. Defaults to 'month'.

        Returns:
            gpd.GeoDataFrame:
                A copy of the input GeoDataFrame with the new month number column added.
        ''' 
        gdf = gdf.copy()

        gdf[datetime_col] = pd.to_datetime(gdf[datetime_col])
        gdf[month_col] = gdf[datetime_col].dt.month

        return gdf

    @staticmethod
    def add_year_column(gdf, datetime_col:str, year_col:str='year', drop_datetime_col:bool=False) -> gpd.GeoDataFrame:  
        '''
        Adds a column to the GeoDataFrame representing the year extracted from a datetime column.

        Parameters:
            gdf (gpd.GeoDataFrame):
                The input GeoDataFrame.
            datetime_col (str):
                The column name containing datetime objects.
            year_col (str):
                The name of the new column to hold the year. Defaults to 'year'.
            drop_datetime_col (bool):
                If True, the original datetime column is dropped from the returned DataFrame. Defaults to False.

        Returns:
            gpd.GeoDataFrame:
                A copy of the input GeoDataFrame with the new year column added (and optionally the datetime column dropped).
        '''
        gdf = gdf.copy()

        gdf[datetime_col] = pd.to_datetime(gdf[datetime_col])
        gdf[year_col] = gdf[datetime_col].dt.year

        if drop_datetime_col:
            gdf = gdf.drop(columns=[datetime_col])

        return gdf

    @staticmethod
    def select_study_region_gdf(gdf:gpd.GeoDataFrame, study_region:gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        '''
        Selects the subset of a GeoDataFrame that falls within the boundaries of a specified study region geometry.
        
        Parameters:
            gdf (gpd.GeoDataFrame):
                The GeoDataFrame containing the data to be clipped (e.g., climate observations).
            study_region (gpd.GeoDataFrame):
                The GeoDataFrame defining the boundary of the region of interest
                (expected to contain one or more polygon geometries).

        Returns:
            gpd.GeoDataFrame:
                A new GeoDataFrame containing only the features from `gdf` that overlap
                with the geometry in `study_region`.
        '''
        return gpd.overlay(gdf, study_region, how='intersection')

    @staticmethod
    def select_date_range_gdf(gdf:gpd.GeoDataFrame, datetime_col:str, time_range:tuple[datetime, datetime]) -> gpd.GeoDataFrame:
        '''
        Filters a GeoDataFrame to retain only the rows where the datetime column falls within a specified inclusive range.

        Parameters:
            gdf (gpd.GeoDataFrame):
                The input GeoDataFrame containing time-series data.
            datetime_col (str):
                The column name in `gdf` containing datetime objects used for filtering.
            time_range (tuple[datetime, datetime]):
                A tuple specifying the start and end of the desired time period, i.e., (start_datetime, end_datetime).

        Returns:
            gpd.GeoDataFrame:
                A new GeoDataFrame containing only the data within the specified time range.
        '''
        return gdf[(gdf[datetime_col] >= time_range[0]) & (gdf[datetime_col] <= time_range[1])]

    @staticmethod
    def select_year_gdf(gdf: gpd.GeoDataFrame, datetime_col: str, year_range: tuple[int, int]) -> gpd.GeoDataFrame:
        '''
            Filters a GeoDataFrame to retain only the rows whose datetime column falls within a specified range of years.

        Parameters:
            gdf (gpd.GeoDataFrame):
                The input GeoDataFrame containing time-series data.
            datetime_col (str):
                The column name in `gdf` containing datetime objects used for filtering.
            year_range (tuple[int, int]):
                A tuple specifying the start and end of the desired year period, i.e., (start_year, end_year). The range is inclusive.

        Returns:
            gpd.GeoDataFrame:
                A new GeoDataFrame containing only the data within the specified year range.
        '''
        gdf[datetime_col] = pd.to_datetime(gdf[datetime_col])
        years = gdf[datetime_col].dt.year
        return gdf[(years >= year_range[0]) & (years <= year_range[1])]

    @staticmethod
    def select_month_gdf(gdf:gpd.GeoDataFrame, datetime_col:str, month_range:tuple[int, int]) -> gpd.GeoDataFrame:
        '''
        Filters a GeoDataFrame to retain only the rows whose datetime column falls within a specified range of months.

        This function correctly handles ranges that cross the year boundary (e.g., December to February). For cross-year
        ranges, months in the second part of the range are temporarily shifted back one year to enable
        correct chronological filtering, though the original date values remain chronologically correct.

        Parameters:
            gdf (gpd.GeoDataFrame):
                The input GeoDataFrame containing time-series data.
            datetime_col (str):
                The column name in `gdf` containing datetime objects used for filtering.
            month_range (tuple[int, int]):
                A tuple specifying the start month and end month as integers (1-12), i.e., (start_month, end_month).

        Returns:
            gpd.GeoDataFrame:
                A new GeoDataFrame containing only the data within the specified month range.
        '''
        gdf[datetime_col] = pd.to_datetime(gdf[datetime_col])

        months = gdf[datetime_col].dt.month
        start_month, end_month = month_range

        # if month range does not cross year boundary
        if end_month >= start_month:
            return gdf[(months >= start_month) & (months <= end_month)]
        else:
            # Cross-year range: select months across year boundary
            gdf = gdf[(months >= start_month) | (months <= end_month)]
            # Shift early months (before start_month) back one year
            shift_mask = gdf[datetime_col].dt.month < start_month
            gdf.loc[shift_mask, datetime_col] = gdf.loc[shift_mask, datetime_col] - pd.DateOffset(years=1)
            return gdf

    @staticmethod
    def select_doy_gdf(gdf:gpd.GeoDataFrame, datetime_col:str, doy_range:tuple[int, int]) -> gpd.GeoDataFrame:
        '''
        Filters a GeoDataFrame to retain only the rows whose datetime column falls within a specified range of days of the year (DOY).

        This function correctly handles ranges that cross the year boundary (e.g., DOY 350 to DOY 10). For cross-year
        ranges, dates in the second part of the range are temporarily shifted back one year to enable
        correct chronological filtering, although the original date values remain chronologically correct.

        Parameters:
            gdf (gpd.GeoDataFrame):
                The input GeoDataFrame containing time-series data.
            datetime_col (str):
                The column name in `gdf` containing datetime objects used for filtering.
            doy_range (tuple[int, int]):
                A tuple specifying the start and end of the desired day-of-year period, i.e., (start_doy, end_doy).

        Returns:
            gpd.GeoDataFrame:
                A new GeoDataFrame containing only the data within the specified DOY range.
        '''
        
        gdf[datetime_col] = pd.to_datetime(gdf[datetime_col])

        doys = gdf[datetime_col].dt.dayofyear
        start_doy, end_doy = doy_range

        if end_doy >= start_doy:
            return gdf[(doys >= start_doy) & (doys <= end_doy)]
        else:
            gdf = gdf[(doys >= start_doy) | (doys <= end_doy)]
            shift_mask = gdf[datetime_col].dt.dayofyear < start_doy
            gdf.loc[shift_mask, datetime_col] = gdf.loc[shift_mask, datetime_col] - pd.DateOffset(years=1)
            return gdf

    @staticmethod
    def subset_gdf(gdf:gpd.GeoDataFrame, datetime_col:str|None=None,
                date_range:tuple[datetime, datetime]|None=None,
                year_range:tuple[int, int]|None=None,
                month_range:tuple[int, int]|None=None,
                doy_range:tuple[int, int]|None=None,
                study_region:gpd.GeoDataFrame|None=None
                ) -> gpd.GeoDataFrame:
        '''
        Creates a subset of a GeoDataFrame by applying various spatio-temporal filters.

        This function sequentially filters the input GeoDataFrame based on date range, year range,
        month range, day-of-year range, and spatial intersection with a study region geometry.

        Parameters:
            gdf (gpd.GeoDataFrame, required):
                The input GeoDataFrame to be filtered.
            datetime_col (str | None, optional):
                The name of the datetime column used for all temporal filtering options.
                Must be provided if any temporal filter is used. Defaults to None.
            date_range (tuple[datetime, datetime] | None, optional):
                Filters data by an exact start and end datetime. Defaults to None.
            year_range (tuple[int, int] | None, optional):
                Filters data to a range of years (inclusive). Defaults to None.
            month_range (tuple[int, int] | None, optional):
                Filters data to a range of months, correctly handling cross-year spans (e.g., Nov-Mar).
                Defaults to None.
            doy_range (tuple[int, int] | None, optional):
                Filters data to a range of days-of-year (DOY), correctly handling cross-year spans (e.g., DOY 350 to DOY 10). Defaults to None.
            study_region (gpd.GeoDataFrame | None, optional):
                Filters data by spatial intersection with the geometry in this GeoDataFrame. Defaults to None.

        Returns:
            gpd.GeoDataFrame:
                The filtered GeoDataFrame containing the subset of data.
        '''
        gdf = gdf.copy()

        if datetime_col is not None:
            if date_range is not None:
                gdf = Utils.select_date_range_gdf(gdf, datetime_col=datetime_col, time_range=date_range)
                if gdf.empty: warnings.warn("The resulting GeoDataFrame is empty after applying the `date_range` filter.", stacklevel=2)
            if year_range is not None:
                gdf = Utils.select_year_gdf(gdf, datetime_col=datetime_col, year_range=year_range)
                if gdf.empty: warnings.warn("The resulting GeoDataFrame is empty after applying the `year_range` filter.", stacklevel=2)
            if month_range is not None:
                gdf = Utils.select_month_gdf(gdf, datetime_col=datetime_col, month_range=month_range)
                if gdf.empty: warnings.warn("The resulting GeoDataFrame is empty after applying the `month_range` filter.", stacklevel=2)
            if doy_range is not None:
                gdf = Utils.select_doy_gdf(gdf, datetime_col=datetime_col, doy_range=doy_range)
                if gdf.empty: warnings.warn("The resulting GeoDataFrame is empty after applying the `doy_range` filter.", stacklevel=2)

        if study_region is not None:
            gdf = Utils.select_study_region_gdf(gdf, study_region)
            if gdf.empty: 
                warnings.warn(message="The resulting GeoDataFrame is empty after applying the `studyregion` filter.", stacklevel=2)

        return gdf

    @staticmethod
    def shift_datetime_by_months(gdf:gpd.GeoDataFrame, shift_by:int, datetime_col:str='valid_time', direction:str='forward') -> gpd.GeoDataFrame:
        '''
        Shifts the datetime values in a specified column forward or backward by a given number of months.

        Parameters:
            gdf (gpd.GeoDataFrame, required):
                The input GeoDataFrame.
            shift_by (int, required):
                The number of months by which to shift the dates.
            datetime_col (str, optional):
                The column name containing datetime objects to be shifted.
            direction (str, optional):
                The direction of the shift. Must be 'forward' (increase date) or 'backward' (decrease date). Defaults to 'forward'.

        Returns:
            gpd.GeoDataFrame: A copy of the input GeoDataFrame with the datetime column shifted.
        '''
        direction = 1 if direction == 'forward' else -1 if direction == 'backward' else 0

        gdf = gdf.copy()

        gdf[datetime_col] = pd.to_datetime(gdf[datetime_col])
        gdf[datetime_col] = gdf[datetime_col] + pd.DateOffset(months=shift_by) * direction

        return gdf

    @staticmethod
    def get_value_col(parameter:str) -> str:
        '''
        Maps a full parameter name to its corresponding simplified column name used within a dataset.

        This function supports mapping common meteorological parameters (like temperature types and
        precipitation) to their abbreviated column identifiers (e.g., 't2m' or 'tp').

        Parameters:
            parameter (str, required):
                The descriptive name of the meteorological parameter (e.g., 'Tmean', 'Precipitation').

        Returns:
            str:
                The abbreviated column name (e.g., 't2m', 'tp').

        Raises:
            ValueError:
                If the provided parameter name is not supported.
        '''
        match(parameter):
            case 'Tmean' | 'Tmin' | 'Tmax':
                return 't2m'
            case 'Precipitation':
                return 'tp'
            case _:
                raise ValueError(f"Unsupported parameter: {parameter}")


    @staticmethod
    def get_seasonal_cycle_plot_values(data:gpd.GeoDataFrame, datetime_col:str='valid_time',
                                       month_range:tuple[int, int]=(1,12)
                                       ) -> tuple[gpd.GeoDataFrame, pd.Index, pd.DatetimeIndex]:
        '''
        Prepares a GeoDataFrame for seasonal cycle plotting by adjusting datetime values for correct chronological ordering across month boundaries.

        This is crucial for visualizing data that spans across the year boundary (e.g., a winter season from October to March).
        It also generates the appropriate x-axis tick labels and locations for monthly plotting.

        Parameters:
            data (gpd.GeoDataFrame, required):
                The input GeoDataFrame containing time-series data.
            datetime_col (str, optional):
                The column name in `data` containing the datetime objects. Defaults to 'valid_time'.
            month_range (tuple[int, int], optional):
                The start and end month numbers (1-12) defining the seasonal cycle.
                This is used to determine if a year-end wrap-around adjustment is needed. Defaults to (1, 12).

        Returns:
            tuple[gpd.GeoDataFrame, list[str], pd.DatetimeIndex]:
                A tuple containing:
                    - plot_df:
                        A copy of the input DataFrame with a new 'plot_time' column containing adjusted
                        datetime values for correct chronological plotting (especially for cross-year ranges).
                    - labels:
                        A list of short month names (e.g., 'Jan', 'Feb') for x-axis tick labels.
                    - labelticks:
                        A DatetimeIndex of the first day of each month for x-axis tick locations.
        '''
        plot_df = data.copy()
        plot_df[datetime_col] = pd.to_datetime(plot_df[datetime_col])
        plot_df["plot_time"] = plot_df[datetime_col]

        start_month, end_month = month_range
        start_year = data[datetime_col].dt.year.min()

        # Determine if the period crosses the year boundary
        crosses_year = (end_month < start_month)

        # Adjust months so they plot in correct chronological order
        if crosses_year:
            # For ranges like (7,6) or (9,3): shift early months (those before start_month) forward by one year
            early_mask = plot_df["plot_time"].dt.month < start_month
            plot_df.loc[early_mask, "plot_time"] += pd.DateOffset(years=1)

        # Sort chronologically after shifting
        plot_df = plot_df.sort_values("plot_time").reset_index(drop=True)

        # ----- Create label ticks -----
        # Define the logical start month (the first month of the period)
        if crosses_year:
            # e.g. (7,6) → start in July 2024 and wrap to June 2025
            label_start = pd.Timestamp(f"{start_year}-{start_month:02d}-01")
        else:
            # e.g. (1,6) or (3,9): simple one-year span
            label_start = pd.Timestamp(f"{start_year}-{start_month:02d}-01")

        # Always 12 months long
        labelticks = pd.date_range(label_start, periods=12, freq="MS")
        labels = labelticks.strftime("%b")

        return plot_df, labels, labelticks

    @staticmethod
    def convert_bbox(south: float, west: float, north: float, east: float) -> tuple:
        '''
        Converts a bounding box defined in (South, West, North, East) order to the standard geospatial format (min_lon, min_lat, max_lon, max_lat).

        Parameters:
            south (float, required):
                Southern boundary (minimum latitude).
            west (float, required):
                Western boundary (minimum longitude).
            north (float, required):
                Northern boundary (maximum latitude).
            east (float, required):
                Eastern boundary (maximum longitude).

        Returns:
            tuple[float, float, float, float]:
                The bounding box in the order (min_lon, min_lat, max_lon, max_lat).
        '''
        return (west, south, east, north)
    
    @staticmethod
    def datetime_to_xr_time(dt: datetime, ds: xr.Dataset) -> Any:
        '''
        Convert a Python datetime.datetime to a value compatible with ds.time.

        Parameters:
            dt (datetime.datetime, required):
                Python datetime (naive or timezone-removed).
            ds (xarray.Dataset | DataArray, required):
                Dataset with a time coordinate.

        Returns:
            datetime.datetime | cftime.datetime
        '''

        import cftime
        sample = ds.time.values[0]

        # Dataset uses cftime (non-standard calendar)
        if isinstance(sample, cftime.datetime):
            calendar = ds.time.encoding.get("calendar", "standard")

            cls = {
                "noleap": cftime.DatetimeNoLeap,
                "365_day": cftime.DatetimeNoLeap,
                "360_day": cftime.Datetime360Day,
                "julian": cftime.DatetimeJulian,
                "gregorian": cftime.DatetimeGregorian,
                "standard": cftime.DatetimeGregorian,
                "proleptic_gregorian": cftime.DatetimeProlepticGregorian,
            }.get(calendar)

            if cls is None:
                raise ValueError(f"Unsupported calendar: {calendar}")

            return cls(
                dt.year, dt.month, dt.day,
                dt.hour, dt.minute, dt.second
            )

        # Dataset already uses pandas / numpy datetime
        return pd.Timestamp(dt).to_pydatetime()
    
    @staticmethod
    def find_covering_domain(gdf, study_region, bbox_coords):
        """Identifies which domain fully contains the study area."""
        bbox_poly = box(*bbox_coords)
        study_geom = study_region.geometry.union_all()
        
        covering = []
        for index, row in gdf.iterrows():
            if row['geometry'].contains(bbox_poly) and row['geometry'].contains(study_geom):
                covering.append(index)
                
        if covering:
            # Return the one with the smallest area (tightest fit)
            return min(covering, key=lambda d: gdf.loc[d, 'geometry'].area)
        return None
    
    @staticmethod
    def create_cordex_gdf(domains_dict, base_crs):
        """Converts the dictionary into a GeoDataFrame."""
        return gpd.GeoDataFrame(
            index=domains_dict.keys(),
            geometry=[Polygon(shell=v["vertices"].values()) for v in domains_dict.values()],
            data={
                "projection": [v["projection"] for v in domains_dict.values()],
                "colour": [v["colour"] for v in domains_dict.values()],
                "long_name": [v["long_name"] for v in domains_dict.values()]
            },
            crs=base_crs
        )
    
    @staticmethod
    def convert_annual_series_to_dfs(
        series_dict: Dict[str, Union[xr.DataArray, xr.Dataset]], 
        value_name: str = "value"
    ) -> Dict[str, pd.DataFrame]:
        """
        Converts yearly or daily xarray objects into cleaned pandas DataFrames.
        """
        df_dict = {}

        for name, obj in series_dict.items():
            try:
                # Ensure we have a DataArray
                if isinstance(obj, xr.Dataset):
                    var_name = list(obj.data_vars)[0]
                    da = obj[var_name]
                else:
                    da = obj

                # Convert to DataFrame
                df = da.to_dataframe(name=value_name).reset_index()

                # Handle the "Year" column
                if "year" in df.columns:
                    pass
                elif "time" in df.columns:
                    # It's a daily series, extract year from the datetime objects
                    df["year"] = pd.to_datetime(df["time"]).dt.year
                else:
                    raise KeyError(f"Could not find 'year' or 'time' in coordinates for {name}")

                # Final selection and cleaning

                df = df[["year", value_name]].copy()
                
                # Remove NaNs
                df = df.dropna(subset=[value_name])
                
                # Optional: Sort by year to ensure clean plots/stats
                df = df.sort_values("year")
                
                df_dict[name] = df

            except Exception as e:
                print(f"Error converting model '{name}': {e}")
                continue

        return df_dict
    
    @staticmethod
    def get_validation_details(mod_est, mod_low, mod_high, obs, param_name):
        """
        Returns (Status, Summary_String)
        """
        # Best estimate inside observed CI
        if obs['lower'] <= mod_est <= obs['upper']:
            summary = f"{param_name}: Good (Est {mod_est:.2f} in Obs CI [{obs['lower']:.2f}, {obs['upper']:.2f}])"
            return "Good", summary
        
        # Calculate Overlap
        overlap_min = max(mod_low, obs['lower'])
        overlap_max = min(mod_high, obs['upper'])
        
        if overlap_max > overlap_min:
            overlap_width = overlap_max - overlap_min
            ref_width = min((mod_high - mod_low), (obs['upper'] - obs['lower']))
            overlap_pct = (overlap_width / ref_width) * 100
            
            if overlap_pct >= 5:
                summary = f"{param_name}: Reasonable ({overlap_pct:.1f}% overlap)"
                return "Reasonable", summary
            else:
                summary = f"{param_name}: Bad (Insufficient overlap: {overlap_pct:.1f}%)"
                return "Bad", summary
        
        summary = f"{param_name}: Bad (No overlap between [{mod_low:.2f}, {mod_high:.2f}] and [{obs['lower']:.2f}, {obs['upper']:.2f}])"
        return "Bad", summary   

    @staticmethod
    def extract_results(parameter, df, df_res, obs_sigma, obs_shape, obs_return_period, obs_event_magnitude):
        """ 
        Compare model validation results with observations and update the DataFrame.
        df: DataFrame to update (model hub)
        df_res: DataFrame with validation results
        obs_sigma: dict with observed sigma values
        obs_shape: dict with observed shape values
        """
        # Helper to format results with uncertainty
        def fmt(row, prefix, suffix):
            try:
                # For temperature, we typically use dI-abs
                val = row[f'{prefix}_{suffix}_est']
                low = row[f'{prefix}_{suffix}_lower']
                upp = row[f'{prefix}_{suffix}_upper']
                return f"{val:.2f} ({low:.2f}, {upp:.2f})"
            except:
                return "N/A"

        for model_name in df_res['model'].unique():
            m_rows = df_res[df_res['model'] == model_name]
            mask = df['model'].str.lower() == model_name.lower()
            if not mask.any(): continue

            # A. Validation Status (using 'eval_' columns)
            v_row = m_rows[m_rows['scenario'] == 'Validation']
            if not v_row.empty:
                r = v_row.iloc[0]
                s_status, s_sum = Utils.get_validation_details(r['eval_sigma0_est'], r['eval_sigma0_lower'], r['eval_sigma0_upper'], obs_sigma, "Σ")
                x_status, x_sum = Utils.get_validation_details(r['eval_shape_est'], r['eval_shape_lower'], r['eval_shape_upper'], obs_shape, "ξ")
                
                df.loc[mask, 'sigma_validation'] = s_status
                df.loc[mask, 'shape_validation'] = x_status
                df.loc[mask, 'validation_summary'] = f"{s_sum}; {x_sum}"

                # Model Threshold is the 'rp_value' calculated during validation
                model_threshold = r['rp_value']
                
                df.loc[mask, 'Magnitude (Obs / Validation)'] = f"{obs_event_magnitude:.2f} / {model_threshold:.2f}"
                df.loc[mask, 'RP (Obs / Validation)'] = f"{obs_return_period:.1f} / {obs_return_period:.1f}"
                
                # Auto-Recommendation Logic
                ranks = {"Good": 3, "Reasonable": 2, "Bad": 1}
                score = min(ranks[s_status], ranks[x_status])
                df.loc[mask, 'Stat Fit'] = [k for k, v in ranks.items() if v == score][0]

            # B. Past Analysis (using 'attr_' columns)
            p_row = m_rows[m_rows['scenario'] == 'Past-Full']
            if not p_row.empty:
                r = p_row.iloc[0]
                df.loc[mask, 'Past_PR'] = fmt(r, 'attr', 'PR')
                if parameter == 'Precipitation':
                    df.loc[mask, 'Past_dI'] = fmt(r, 'attr', 'dI-rel')
                else:
                    df.loc[mask, 'Past_dI'] = fmt(r, 'attr', 'dI-abs')
            
            if p_row.empty:
                p_row = m_rows[m_rows['scenario'] == 'Validation']
                if not p_row.empty:
                    r = p_row.iloc[0]
                    df.loc[mask, 'Past_PR'] = fmt(r, 'attr', 'PR')
                    if parameter == 'Precipitation':
                        df.loc[mask, 'Past_dI'] = fmt(r, 'attr', 'dI-rel')
                    else:
                        df.loc[mask, 'Past_dI'] = fmt(r, 'attr', 'dI-abs')

            # C. Future Projections (using 'proj_' columns)
            f20_row = m_rows[m_rows['scenario'] == 'Future-2.0']
            if not f20_row.empty:
                r = f20_row.iloc[0]
                df.loc[mask, 'Fut_2.0_PR'] = fmt(r, 'proj', 'PR')
                if parameter == 'Precipitation':
                    df.loc[mask, 'Fut_2.0_dI'] = fmt(r, 'proj', 'dI-rel')
                else:
                    df.loc[mask, 'Fut_2.0_dI'] = fmt(r, 'proj', 'dI-abs')

            f26_row = m_rows[m_rows['scenario'] == 'Future-2.6']
            if not f26_row.empty:
                r = f26_row.iloc[0]
                df.loc[mask, 'Fut_2.6_PR'] = fmt(r, 'proj', 'PR')
                if parameter == 'Precipitation':
                    df.loc[mask, 'Fut_2.6_dI'] = fmt(r, 'proj', 'dI-rel')
                else:
                    df.loc[mask, 'Fut_2.6_dI'] = fmt(r, 'proj', 'dI-abs') 

    @staticmethod
    def create_decision_hub(df_validation, step='full', project_filter='all', save_path=None):
        """
        Creates an interactive, scrollable table for Model Validation.
        """
        df_validation['Include T/F'] = df_validation['Include T/F'].astype(object)
        df_filtered = df_validation.copy()
        if project_filter.lower() != 'all':
            df_filtered = df_filtered[df_filtered['project'].str.lower() == project_filter.lower()]
        
        rows = []
        decision_widgets = {}

        # 1. Define Standard Column Widths
        w_model, w_ens, w_val, w_res, w_drop, w_obs = '280px', '80px', '100px', '130px', '110px', '250px'
        
        # Calculate Total Width based on step to prevent "squishing"
        if step == 'full':
            total_width = '1650px'
        elif step == 'statistics':
            total_width = '1300px'
        else:
            total_width = '820px'

        header_list = [
            widgets.HTML(f"<b>Model</b>", layout={'width': w_model}), 
            widgets.HTML(f"<b>Project</b>", layout={'width': w_ens})
        ]

        if step == 'full':
            header_list += [
                widgets.HTML(f'<b>Seasonal</b>', layout={'width': w_val}), 
                widgets.HTML(f'<b>Spatial</b>', layout={'width': w_val}),
                widgets.HTML(f'<b>Stat Fit</b>', layout={'width': w_val}), 
                widgets.HTML(f'<b>Mag (Obs/Val)</b>', layout={'width': w_res}),
                widgets.HTML(f'<b>Past PR/dI</b>', layout={'width': w_res}), 
                widgets.HTML(f'<b>Fut 2.0 PR/dI</b>', layout={'width': w_res}), 
                widgets.HTML(f'<b>Fut 2.6 PR/dI</b>', layout={'width': w_res}), 
                widgets.HTML(f'<b>Include?</b>', layout={'width': w_drop}), 
                widgets.HTML(f'<b>Comments</b>', layout={'width': w_obs})
            ]
        elif step == 'visual':
            header_list += [ 
                widgets.HTML(f'<b>Include?</b>', layout={'width': w_drop}), 
                widgets.HTML(f'<b>Comments</b>', layout={'width': w_obs})
            ]
        elif step == 'statistics':
            header_list += [
                widgets.HTML(f'<b>Σ</b>', layout={'width': '80px'}), 
                widgets.HTML(f'<b>ξ</b>', layout={'width': '80px'}),
                widgets.HTML(f'<b>Summary</b>', layout={'width': '300px'}), 
                widgets.HTML(f'<b>Mag (Obs/Val)</b>', layout={'width': w_res}),
                widgets.HTML(f'<b>Include?</b>', layout={'width': w_drop}), 
                widgets.HTML(f'<b>Comments</b>', layout={'width': w_obs})
            ]
        else:
            header_list.extend([
                widgets.HTML(f"<b>{step.capitalize()} Score</b>", layout={'width': '120px'}), 
                widgets.HTML(f'<b>Include?</b>', layout={'width': w_drop}), 
                widgets.HTML(f'<b>Comments</b>', layout={'width': w_obs})
            ])

        header = widgets.HBox(header_list, layout={
            'background_color': '#f0f0f0', 
            'border_bottom': '2px solid #444444', 
            'padding': '5px 0px'
        })
        

        for idx, row in df_filtered.iterrows():
            # Using HTML for Model/Ens instead of Label to ensure same height/alignment as status tags
            line_items = [
                widgets.HTML(f"<div>{row['model']}</div>", layout={'width': w_model, 'height': '32px'}), 
                widgets.HTML(f"<div>{row['project']}</div>", layout={'width': w_ens, 'height': '32px'})
            ]
            decision_widgets[row['model']] = {}

            # Default Include logic
            current_inc = row.get('Include T/F')
            current_obs = row.get('Comments', '')

            default_inc = True 

            inc_drop = widgets.Dropdown(
                options=[True, False], 
                value=current_inc if pd.notna(current_inc) and current_inc in [True, False] else default_inc, 
                layout={'width': w_drop}
            )
            obs_text = widgets.Text(value=str(current_obs) if pd.notna(current_obs) else "", layout={'width': w_obs})

            if step == 'statistics':
                for p in ['sigma_validation', 'shape_validation']:
                    val = str(row.get(p, 'Pending'))
                    color = 'green' if val == 'Good' else 'orange' if val == 'Reasonable' else 'red'
                    line_items.append(widgets.HTML(f"<div><b style='color:{color}'>{val}</b></div>", layout={'width': '80px'}))
                
                line_items.append(widgets.HTML(f"<div style='padding-left:8px; line-height:14px; display:flex; align_items:center; height:32px;'><small>{row.get('validation_summary', '')}</small></div>", layout={'width': '300px'}))
                line_items.append(widgets.HTML(f"<div><small>{row.get('Magnitude (Obs / Validation)', 'N/A')}</small></div>", layout={'width': w_res}))
                line_items.extend([inc_drop, obs_text])

            elif step == 'visual':
                line_items.extend([inc_drop, obs_text])

            elif step == 'full':
                for col in ['Seasonal cycle', 'Spatial maps', 'Stat Fit']:
                    val = str(row.get(col, 'Pending'))
                    color = 'green' if val == 'Good' else 'orange' if val == 'Reasonable' else 'red'
                    line_items.append(widgets.HTML(f"<div><b style='color:{color}'>{val}</b></div>", layout={'width': w_val}))
                
                line_items.append(widgets.HTML(f"<div><small>{row.get('Magnitude (Obs / Validation)', 'N/A')}</small></div>", layout={'width': w_res}))
                
                for prefix in ['Past', 'Fut_2.0', 'Fut_2.6']:
                    val_str = f"PR: {row.get(f'{prefix}_PR', 'N/A')}<br>dI: {row.get(f'{prefix}_dI', 'N/A')}"
                    line_items.append(widgets.HTML(f"<div style='padding-left:8px; line-height:14px; display:flex; align_items:center; height:32px;'><small>{val_str}</small></div>", layout={'width': w_res}))
                
                line_items.extend([inc_drop, obs_text])

            elif step in ['seasonal', 'spatial']:
                col_name = 'Seasonal cycle' if step == 'seasonal' else 'Spatial maps'
                current_val = row.get(col_name)
                s_val = current_val if current_val in ['Good', 'Reasonable', 'Bad'] else 'Reasonable'
                drop = widgets.Dropdown(options=['Good', 'Reasonable', 'Bad'], value=s_val, layout={'width': '120px'})
                line_items.append(drop)
                decision_widgets[row['model']][step] = drop
                line_items.extend([inc_drop, obs_text])

            decision_widgets[row['model']]['include'] = inc_drop
            decision_widgets[row['model']]['obs'] = obs_text
            
            # Assemble Row with Vertical Margins and Zebra Striping
            row_box = widgets.HBox(
                line_items, 
                layout={
                    'border_bottom': '1px solid #cccccc', 
                    'padding': '6px 0px',
                    'background_color': '#ffffff' if idx % 2 == 0 else '#fafafa'
                }
            )
            rows.append(row_box)
        
        # 4. Create the Scrollable Table Container
        table_body = widgets.VBox(rows, layout={'width': total_width})
        table_full = widgets.VBox([header, table_body], layout={'width': total_width})
        
        # This wrapper allows the table to be wider than the screen
        scrollable_wrapper = widgets.VBox(
            [table_full], 
            layout={
                'overflow_x': 'auto', 
                'width': '100%', 
                'border': '1px solid #cccccc',
                'margin': '10px 0px'
            }
        )

        # 5. Buttons and Output
        save_button = widgets.Button(description=f"💾 Save {step.capitalize()}", button_style='success', layout={'margin': '10px 0px'})
        output = widgets.Output()
        
        def on_save_clicked(b):
            with output:
                clear_output()
                for model, w in decision_widgets.items():
                    mask = df_validation['model'] == model
                    if 'seasonal' in w: df_validation.loc[mask, 'Seasonal cycle'] = w['seasonal'].value
                    if 'spatial' in w: df_validation.loc[mask, 'Spatial maps'] = w['spatial'].value
                    if 'include' in w: df_validation.loc[mask, 'Include T/F'] = w['include'].value
                    if 'obs' in w: df_validation.loc[mask, 'Comments'] = w['obs'].value
                
                if save_path:
                    df_validation.to_csv(save_path, index=False)
                    print(f"✅ Changes saved to memory AND disk: {save_path}")
                else:
                    print(f"✅ Changes saved to memory")

        save_button.on_click(on_save_clicked)
        
        
        # Final Display
        ui = widgets.VBox([
            widgets.HTML(f"<h3>Step 4: {step.capitalize()} Validation Hub</h3>"), 
            scrollable_wrapper, 
            save_button, 
            output
        ])
        return ui

    @staticmethod
    def get_parameter_config(parameter):
        PARAMETER_CONFIG = {
                "Tmax": {
                    "variable": "Maximum Temperature",
                    "value_col": "t2m",
                    "y_label": "c",
                    "unit": "°C",
                    "calculation": "absolute",
                    "method": "mean",
                    "datetime_col": "valid_time",
                    "from_unit": "k",
                    "to_unit": "c",
                },
                "Tmean": {
                    "variable": "Mean Temperature",
                    "value_col": "t2m",
                    "y_label": "c",
                    "unit": "°C",
                    "calculation": "absolute",
                    "method": "mean",
                    "datetime_col": "valid_time",
                    "from_unit": "k",
                    "to_unit": "c",
                },
                "Tmin": {
                    "variable": "Minimum Temperature",
                    "value_col": "t2m",
                    "y_label": "c",
                    "unit": "°C",
                    "calculation": "absolute",
                    "method": "mean",
                    "datetime_col": "valid_time",
                    "from_unit": "k",
                    "to_unit": "c",
                },
                "Precipitation": {
                    "variable": "Total Precipitation",
                    "value_col": "tp",
                    "y_label": "mm",
                    "unit": "mm",
                    "calculation": "relative",
                    "method": "sum",
                    "datetime_col": "valid_time",
                    "from_unit": "m",
                    "to_unit": "mm",
                },
            }
        try:
            return PARAMETER_CONFIG.get(parameter)
        except Exception as e:
            print(f"Error retrieving parameter config for {parameter}: {e}")
            return None

    @staticmethod
    def var_map(parameter, model):
        VAR_MAP = {
            "Tmean":         {"cordex": "tas",    "cmip6": "near_surface_air_temperature", "era5": "temperature_2m_mean"},
            "Tmin":          {"cordex": "tasmin", "cmip6": "daily_minimum_near_surface_air_temperature", "era5": "temperature_2m_min"},
            "Tmax":          {"cordex": "tasmax", "cmip6": "daily_maximum_near_surface_air_temperature", "era5": "temperature_2m_max"},
            "Precipitation": {"cordex": "pr",     "cmip6": "precipitation", "era5": "total_precipitation"},
        }

        entry = VAR_MAP.get(parameter, {model: parameter})
        
        if model not in entry:
            raise ValueError(f"Model '{model}' not found for parameter '{parameter}'. "
                            f"Available models: {list(entry.keys())}")
                            
        return entry[model]
    
    @staticmethod
    def get_cordex_domain_configs():
        """Returns the raw configuration dictionary for CORDEX domains."""
        return {"SAM" : {"vertices" : {"TLC":(273.26, 18.50), "TRC":(327.52, 17.23), "BRC" :(343.02, -54.6), "BLC" :(254.28, -52.66)},
                        "projection" : ccrs.RotatedPole(pole_longitude = -56.06, pole_latitude = 70.6, central_rotated_longitude=180),
                        "colour" : "red", 
                        "long_name" : "South America"},
            "CAM" : {"vertices" : {"TLC":(235.74, 28.79), "TRC":(337.78, 31.40), "BRC" :(329.46, -17.23), "BLC" :(246.10, -19.46)},
                        "projection" : ccrs.RotatedPole(pole_longitude = 113.98, pole_latitude = 75.74),
                        "colour" : "darkorange", 
                        "long_name" : "Central America"},
            "NAM" : {"vertices" : {"TLC" :(189.26, 59.28), "TRC" :(336.74, 59.28), "BRC": (293.16, 12.55), "BLC" :(232.84, 12.56)},
                        "projection" : ccrs.RotatedPole(pole_longitude = 83.0, pole_latitude = 42.5),
                        "colour" : "forestgreen", 
                        "long_name" : "North America"},
            "EUR" : {"vertices" : {"TLC" :(315.86-360, 60.21), "TRC" :(64.4, 66.65), "BRC" :(36.30, 25.36), "BLC" :(350.01-360, 22.20)},
                        "projection" : ccrs.RotatedPole(pole_longitude = -162.0, pole_latitude = 39.25),
                        "colour" : "royalblue", 
                        "long_name" : "Europe"},
            "AFR" : {"vertices" : {"TLC" :(335.36, 42.24), "TRC" :(60.28, 42.24), "BRC" :(60.28, -45.76), "BLC" :(335.36, -45.76)},
                        "projection" : ccrs.RotatedPole(pole_longitude = 180.0, pole_latitude = 90.0),
                        "colour" : "teal", 
                        "long_name" : "Africa"},
            "WAS" : {"vertices" : {"TLC" :(19.88, 43.5), "TRC" :(115.55, 41.0), "BRC" :(106.43, -15.23), "BLC" :(26.19, -12.97)},
                        "projection" : ccrs.RotatedPole(pole_longitude = 236.66, pole_latitude = 79.95),
                        "colour" : "chocolate",
                        "long_name" : "South Asia"},
            "EAS" : {"vertices" : {"TLC" :(51.59, 50.50), "TRC" :(181.50, 50.31), "BRC" :(156.08, -0.24), "BLC" :(76.91, -0.10)},
                        "projection" : ccrs.RotatedPole(pole_longitude = 296.3, pole_latitude = 61.0),
                        "colour" : "crimson",
                        "long_name" : "East Asia"},
            "CAS" : {"vertices" : {"TLC" :(11.05, 54.76), "TRC" :(139.13, 56.48), "BRC" :(108.44, 19.39), "BLC" :(42.41, 18.34)},
                        "projection" : ccrs.RotatedPole(pole_longitude = 256.61, pole_latitude = 43.48),
                        "colour" : "darkgoldenrod",
                        "long_name" : "Central Asia"},
            "AUS" : {"vertices" : {"TLC" :(110.19, 8.76), "TRC" :(182.02, 12.21), "BRC" :(206.57, -39.25), "BLC" :(89.25, -44.28)},
                        "projection" : ccrs.RotatedPole(pole_longitude = 321.38, pole_latitude = -60.31, central_rotated_longitude=180),
                        "colour" : "deeppink",
                        "long_name" : "Australasia"},
            "ANT" : {"vertices" : {"TLC" :(140.58, -56.0), "TRC" :(245.58, -56.0), "BRC" :(326.14, -56.26), "BLC" :(60.02, -56.26)},
                        "projection" : ccrs.RotatedPole(pole_longitude = 13.09, pole_latitude = -6.08, central_rotated_longitude=180),
                        "colour" : "slategray",
                        "long_name" : "Antarctica"},
            "ARC" : {"vertices" : {"TLC" :(214.68, 55.43), "TRC" :(140.59, 52.53), "BRC" :(40.35, 46.06), "BLC" :(324.82, 52.0)},
                        "projection" : ccrs.RotatedPole(pole_longitude = 0.0, pole_latitude = 6.55),
                        "colour" : "Crimson",
                        "long_name" : "Arctic"},
            "MED" : {"vertices" : {"TLC" :(339.79, 50.65), "TRC" :(50.85, 52.34), "BRC" :(38.33, 26.73), "BLC" :(353.96, 25.63)},
                        "projection" : ccrs.RotatedPole(pole_longitude = 198.0, pole_latitude = 39.25),
                        "colour" : "MediumOrchid",
                        "long_name" : "Mediterranean"},
            "MNA" : {"vertices" : {"TLC" :(333.0, 45.0), "TRC" :(76.0, 45), "BRC" :(76.0, -7), "BLC" :(333.0, -7)},
                        "projection" : ccrs.RotatedPole(pole_longitude = 180.0, pole_latitude = 90.0),
                        "colour" : "MediumSeaGreen",
                        "long_name" : "Middle East / North Africa"},
    }

    @staticmethod
    def get_gcm_cordex_to_cmip5():
        """
        Returns a nested mapping: CORDEX_GCM_Name -> Driving_Model & Ensembles.
        Each ensemble contains the specific periods for Historical and RCP8.5.
        """
        return {
            'cccma_canesm2': {
                'driving_model': "canesm2",
                'ensembles': {
                    'r1i1p1': {
                        'historical': ["185001-200512"],
                        'rcp_8_5': ["200601-210012"]
                    }
                }
            },
            'cnrm_cerfacs_cm5': {
                'driving_model': "cnrm_cm5",
                'ensembles': {
                    'r1i1p1': {
                        'historical': ["185001-189912", "190001-194912", "195001-200512"],
                        'rcp_8_5': ["200601-205512", "205601-210012"]
                    }
                }
            },
            'ichec_ec_earth': {
                'driving_model': "ec_earth",
                'ensembles': {
                    'r12i1p1': {
                        'historical': ["185001-189912", "190001-194912", "195001-201212"],
                        'rcp_8_5': ["200601-210012"]
                    }
                }
            },
            'ipsl_cm5a_mr': {
                'driving_model': "ipsl_cm5a_mr",
                'ensembles': {
                    'r1i1p1': {
                        'historical': ["185001-200512"],
                        'rcp_8_5': ["200601-210012"]
                    }
                }
            },
            'mohc_hadgem2_es': {
                'driving_model': "hadgem2_es",
                'ensembles': {
                    'r1i1p1': {
                        'historical': ["185912-188411", "188412-190911", "190912-193411", "193412-195911", "195912-198411", "198412-200511"],
                        'rcp_8_5': ["200512-203011", "203012-205511", "205512-208011", "208012-210011"]
                    }
                }
            },
            'mpi_m_mpi_esm_lr': {
                'driving_model': "mpi_esm_lr",
                'ensembles': {
                    'r1i1p1': {
                        'historical': ["185001-200512"],
                        'rcp_8_5': ["200601-210012"]
                    },
                    'r3i1p1': {
                        'historical': ["185001-200512"],
                        'rcp_8_5': ["200601-210012"]
                    }
                }
            },
            'ncc_noresm1_m': {
                'driving_model': "noresm1_m",
                'ensembles': {
                    'r1i1p1': {
                        'historical': ["185001-200512"],
                        'rcp_8_5': ["200601-210012"]
                    }
                }
            }
        }