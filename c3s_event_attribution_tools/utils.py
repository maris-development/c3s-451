from datetime import timedelta
import requests
import numpy as np
from shapely.geometry import Polygon
import webbrowser
from urllib.parse import urlencode
from typing import Dict, Any
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import base64
from io import BytesIO
from .plot import *
import os
import warnings


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