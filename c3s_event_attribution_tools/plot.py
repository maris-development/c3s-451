import matplotlib.pyplot as plt
import plotly.express as px
import geopandas as gpd
import pandas as pd
import contextily as ctx
import math
import cartopy 
from shapely.geometry import shape, Polygon, mapping, MultiPolygon, GeometryCollection
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import rasterio
from rasterio import features
import numpy as np
from shapely import contains_xy
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from datetime import datetime
import xarray as xr
import matplotlib.font_manager as fm
import os
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, BoundaryNorm, TwoSlopeNorm
import matplotlib.colors as mcolors
import cmocean
import matplotlib.image as mpimg
from matplotlib.colors import LogNorm
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from IPython.display import display, clear_output
import re
from shapely.geometry import box

# set font directory
BASE_DIR = os.path.dirname(__file__)
FONT_PATH = os.path.join(BASE_DIR, "RobotoCondensed-Regular.ttf")
FONT_ROBOTO_CONDENSED_REGULAR = fm.FontProperties(fname=FONT_PATH)
LOGO_HORIZON_PATH = os.path.join(BASE_DIR, "LogoLine_horizon_C3S.png")

fm.fontManager.addfont(FONT_PATH)

# set font family and settings globally
plt.rcParams["font.family"] = FONT_ROBOTO_CONDENSED_REGULAR.get_name()
plt.rcParams["axes.titlecolor"] = "#364563"
plt.rcParams["figure.titlesize"] = 27
plt.rcParams["axes.titlesize"] = 27
plt.rcParams["font.size"] = 13
plt.rcParams["axes.labelcolor"] = "#6a6a6b"     # 🔹 x/y axis labels
plt.rcParams["xtick.color"] = "#6a6a6b"         # 🔹 x-axis tick labels
plt.rcParams["xtick.labelcolor"] = "#6a6a6b"
plt.rcParams["xtick.labelsize"] = 13
plt.rcParams["ytick.color"] = "#6a6a6b"         # 🔹 y-axis tick labels
plt.rcParams["ytick.labelcolor"] = "#6a6a6b"
plt.rcParams["ytick.labelsize"] = 13

# set colormap color values
TEMPERATURE_COLORS = ["#204182", "#24569c", "#559bd4", "#95d0f0", "#cee9f5", "#f6fcfe","#fff1ba", "#ffc656", "#f6862f", "#e8432a", "#b92027"]
PRECIPITATION_COLORS = ["#f3f7fb",   "#deebf7",  "#c6dbef",   "#9ecae1",  "#6baed6",  "#4292c6", "#1d609b",  "#08396b"]
ANOMALY_COLORS = ["#693f18", "#B1967E", "#D2C7BE", "#ffffff", "#A6B4CB", "#7888A4" , "#204282"]

# create colormap from colors
temperature_cmap = ListedColormap(TEMPERATURE_COLORS, name="temperature_cmap")
temperature_positive_cmap = ListedColormap(TEMPERATURE_COLORS[5:], name="temperature_positive_cmap")
temperature_negative_cmap = ListedColormap(TEMPERATURE_COLORS[:6], name="temperature_negative_cmap")
precipitation_cmap = LinearSegmentedColormap.from_list("precipitation_cmap", PRECIPITATION_COLORS, N=len(PRECIPITATION_COLORS))
anomaly_cmap = ListedColormap(ANOMALY_COLORS, name="anomaly_cmap")
anomaly_cmap = LinearSegmentedColormap.from_list("anomaly_cmap", ANOMALY_COLORS, N=256)
anomaly_positive_cmap = ListedColormap(ANOMALY_COLORS[1:], name="anomaly_positive_cmap")
anomaly_negative_cmap = ListedColormap(ANOMALY_COLORS[:3], name="anomaly_negative_cmap")

class Plot:
    '''
    Class containing static methods for plotting geospatial data with appropriate colormaps and styles.
    '''
    
    
    @staticmethod
    def cmap_norm_boundary(vmin:float, vmax:float, steps:int) -> BoundaryNorm:
        '''
        Generates a discrete colormap normalization for plotting.

        Parameters:
            vmin (float, required):
                The minimum data value.
            vmax (float, required):
                The maximum data value.
            steps (int, required):
                The approximate number of discrete steps/colors desired.

        Returns:
            matplotlib.colors.BoundaryNorm:
                A BoundaryNorm object suitable for discrete colormapping.
        '''
        vmin_i, vmax_i = int(np.floor(vmin)), int(np.ceil(vmax))
        # Create integer step boundaries
        step_size = max(1, math.ceil((vmax_i - vmin_i) / steps))
        boundaries = np.arange(vmin_i, vmax_i + step_size, step_size)
        # Ensure we include the upper limit
        if boundaries[-1] < vmax_i:
            boundaries = np.append(boundaries, vmax_i)
        return BoundaryNorm(boundaries, len(boundaries) - 1)

    @staticmethod
    def cmap_norm_twoslope(vmin:float, vmax:float, center:float) -> TwoSlopeNorm:
        '''
        Generates a TwoSlopeNorm colormap normalization.

        This normalization is useful for plotting data where the color mapping needs
        to be non-linear around a specified center value (e.g., zero or a reference point),
        allowing for different color gradients on either side of the center.

        Parameters:
            vmin (float, required):
                The minimum data value.
            vmax (float, required):
                The maximum data value.
            center (float, required):
                The data value that should be mapped to the center of the colormap.

        Returns:
            matplotlib.colors.TwoSlopeNorm:
                A TwoSlopeNorm object.
        '''
        return TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)

    # set global style paramaters
    @staticmethod
    def set_style(param:str, value:str|int|float):
        '''
        Sets a global Matplotlib style parameter.

        Parameters:
            param (str, required):
                The Mathplotlib rcParams parameter to set (e.g., 'font.size')
            value (str | int | float, required):
                The value to assign to the specified parameter
        '''
        plt.rcParams[param] = value

    @staticmethod
    def precip_bins(vmax: float):
        '''
        Calculates a set of adaptive precipitation bins based on the maximum data value.

        The bins are designed to provide an appropriate color-mapping resolution for
        different ranges of precipitation intensity (e.g., mm/day or total mm).

        Parameters:
            vmax (float, required):
                The maximum precipitation value in the dataset.

        Returns:
            numpy.ndarray:
                An array of precipitation bin boundaries.
        '''
        if vmax <= 15:
            return np.array([0, 1, 2, 3, 4, 6, 7, 8, 10])
        elif vmax <= 40:
            return np.array([0, 1, 2, 4, 6, 8, 10, 15, 20])
        elif vmax <= 75:
            return np.array([0, 5, 10, 15, 20, 25, 30, 40, 50])
        elif vmax <= 150:
            return np.array([0, 5, 10, 20, 30, 40, 60, 80, 100])
        elif vmax <= 300:
            return np.array([0, 25, 50, 75, 100, 125, 150, 175, 200])
        else:
            return np.array([0, 50, 100, 150, 200, 250, 300, 350, 400])


    # get colormap
    @staticmethod
    def get_colormap(map:str, vmin:float, vmax:float, value_col:str=None) -> tuple[str|ListedColormap, BoundaryNorm|TwoSlopeNorm]: 
        '''
        Retrieves the appropriate colormap and normalization for plotting based on the data type.

        Parameters:
            map (str, required):
                A key identifying the data type (e.g., 't2m', 'tp', 'anomaly', or 'sst').
            vmin (float, required):
                The minimum data value.
            vmax (float, required):
                The maximum data value.
            value_col (str, optional):
                A secondary key, primarily for 'anomaly', to specify
                the underlying variable type (e.g., "t2m", "sst"). Defaults to None.

        Returns:
            tuple[str | matplotlib.colors.Colormap, matplotlib.colors.BoundaryNorm | matplotlib.colors.TwoSlopeNorm]:
                The colormap object and the normalization object for the data.
        '''
        original_map = map

        match map:
            case 't2m':
                cmap = temperature_positive_cmap if vmin >= 0 else temperature_negative_cmap if vmax <= 0 else temperature_cmap
                norm = Plot.cmap_norm_boundary(vmin, vmax, 6) if vmin >= 0 else Plot.cmap_norm_boundary(vmin, vmax, 6) if vmax <= 0 else Plot.cmap_norm_boundary(vmin, vmax, 11)
                return cmap, norm
            case 'tp':
                boundaries = Plot.precip_bins(vmax)
                cmap = precipitation_cmap
                norm = BoundaryNorm(boundaries, len(boundaries) - 1)
                return cmap, norm
            case 'anomaly' | 'sst':
                if value_col in ["t2m", "sst"]:
                    max_abs = max(abs(vmin), abs(vmax))
                    cmap = temperature_cmap
                    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)
                    return cmap, norm
                else:
                    cmap = anomaly_cmap
                    vmax_adj = 0.7 * vmax
                    if vmax_adj <= 0: # Prevent error when anomaly is all negative
                        vmax_adj = 1
                    norm = Plot.cmap_norm_twoslope(vmin=-1, vmax=vmax_adj, center=0)
                    return cmap, norm
            case _:
                return original_map, Plot.cmap_norm_boundary(vmin, vmax, 11)

    @staticmethod
    def visualize_geo(
        df,
        value_col: str,
        lon_col: str = "longitude",
        lat_col: str = "latitude",
        backend: str = "plotly",      
        # matplotlib kwargs
        figsize=(10, 10),
        cmap="viridis",
        edgecolor="black",
        alpha=0.7,        
        # plotly kwargs
        mapbox_style="carto-positron",
        zoom=3,
        width=800,
        height=600,
        size=None,
        hover_name=None,
    ) -> plt.Figure:
        '''
        Visualize a tabular or GeoDataFrame spatially.

        This function supports plotting geographical data using either Matplotlib/GeoPandas
        or Plotly, handling the conversion of standard DataFrames (with lon/lat columns)
        to GeoDataFrames if necessary.

        Parameters:
            df (pd.DataFrame | gpd.GeoDataFrame, required):
                Your table containing lon/lat or a geometry column.
            value_col (str, required):
                Column name to drive the color (or size) of points/polygons.
            lon_col (str, optional):
                Column name for longitude (if df is not yet a GeoDataFrame).
            lat_col (str, optional):
                Column name for latitude (if df is not yet a GeoDataFrame).
            backend (str, optional):
                Which renderer to use. Must be 'matplotlib' or 'plotly'. Defaults to "plotly".
            figsize (tuple, optional):
                Matplotlib figure size (width, height). Defaults to (10, 10).
            cmap (str, optional):
                Matplotlib colormap name. Defaults to "viridis".
            edgecolor (str, optional):
                Matplotlib edge color for geometries. Defaults to "black".
            alpha (float, optional):
                Matplotlib transparency level. Defaults to 0.7.
            mapbox_style (str, optional):
                Plotly map style (e.g., "carto-positron"). Defaults to "carto-positron".
            zoom (int, optional):
                Plotly initial zoom level. Defaults to 3.
            width (int, optional):
                Plotly figure width. Defaults to 800.
            height (int, optional):
                Plotly figure height. Defaults to 600.
            size (str, optional):
                Plotly column name to scale point size. Defaults to None.
            hover_name (str, optional):
                Plotly column name for hover labels. Defaults to None.

        Returns:
            matplotlib.figure.Figure | plotly.graph_objs._figure.Figure:
                The generated figure object.

        Raises:
            ValueError:
                If an unknown backend is provided.
        '''
        # 1) ensure GeoDataFrame
        if not isinstance(df, gpd.GeoDataFrame):
            df = gpd.GeoDataFrame(
                df.copy(),
                geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
                crs="EPSG:4326"
            )
        else:
            # if it's already a GeoDataFrame but has no geometry, force from lon/lat
            if df.geometry.isna().all():
                df = df.set_geometry(
                    gpd.points_from_xy(df[lon_col], df[lat_col])
                )
        
        if backend.lower() == "matplotlib":
            fig, ax = plt.subplots(figsize=figsize)
            df.plot(
                column=value_col,
                cmap=cmap,
                edgecolor=edgecolor,
                alpha=alpha,
                legend=True,
                ax=ax
            )
            ax.set_axis_off()
            plt.tight_layout()
            return fig

        elif backend.lower() == "plotly":
            fig = px.scatter_mapbox(
                df,
                lat=lat_col,
                lon=lon_col,
                color=value_col,
                size=size,
                hover_name=hover_name,
                zoom=zoom,
                width=width,
                height=height,
                mapbox_style=mapbox_style
            )
            return fig

        else:
            raise ValueError(f"Unknown backend '{backend}'. Choose 'matplotlib' or 'plotly'.")

    @staticmethod
    def add_image_below(fig,
                        image_path=LOGO_HORIZON_PATH,
                        min_height_frac=0.06,
                        max_height_frac=0.40,
                        pad_frac=0.02,
                        save_path=None
    ) -> tuple[plt.Figure, plt.Axes]:
        '''
        Adds an image (e.g., a logo or watermark) below a Matplotlib figure, preserving its aspect ratio.

        This function modifies the figure in-place by adjusting the position of all existing
        axes upward to make room for the new image at the bottom. The resulting figure is
        displayed and can be saved to a file.

        Parameters:
            fig (matplotlib.figure.Figure, required):
                The Matplotlib figure object to modify.
            image_path (str, optional):
                The file path to the image to be inserted.
            min_height_frac (float, optional):
                The minimum fractional height of the figure to reserve for the image. Defaults to 0.06.
            max_height_frac (float, optional):
                The maximum fractional height of the figure to reserve for the image. Defaults to 0.40.
            pad_frac (float, optional):
                The fractional padding space between the original plot area and the added image. Defaults to 0.02.
            save_path (str, optional):
                The path where the final figure should be saved. If None, the figure is not saved. Defaults to None.

        Returns:
            tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
                A tuple containing:
                - fig: The modified Matplotlib figure object.
                - ax_img: The new Axes object containing the added image.
        '''
        img = mpimg.imread(image_path)
        img_h, img_w = img.shape[0:2]

        fig_w, fig_h = fig.get_size_inches()
        img_aspect = img_h / img_w
        fig_aspect = fig_w / fig_h
        true_height_frac = img_aspect * fig_aspect

        # Clamp height
        height = max(min_height_frac, min(max_height_frac, true_height_frac))

        # Reposition existing axes upward
        available_space = 1.0 - height - pad_frac
        for ax in fig.get_axes():
            left, bottom, width, h = ax.get_position().bounds
            new_bottom = height + pad_frac + bottom * available_space
            new_height = h * available_space
            ax.set_position([left, new_bottom, width, new_height])

        # Add logo axes
        ax_img = fig.add_axes([0.1, 0.0, .8, height*.8])
        ax_img.imshow(img, aspect="auto")
        ax_img.axis("off")

        # Save to file
        if save_path is not None:
            fig.savefig(save_path, bbox_inches="tight", dpi=fig.dpi)

        clear_output(wait=True)
        display(fig)

        return fig, ax_img




    # plots a single plot of a GeoDataFrame
    @staticmethod
    def plot_gdf(gdf:gpd.GeoDataFrame,
                value_col:str,
                borders:bool=True,
                coastlines:bool=True,
                gridlines:bool=True,
                title:str|None=None,
                legend:bool=True,
                legend_title:str|None=None,
                cmap:str=None,
                fig_size:tuple[int, int]=(7,5),
                polygons:list[Polygon]|None=None,
                projection:cartopy.crs=ccrs.PlateCarree(),
                extends:tuple[float, float, float, float]|None=None,
                dpi:int=100,
                marker:str='s',
                add_logos:bool=True,
                polygon_color='cyan',
                ax=None):
        '''
        Plots a single map of a GeoDataFrame, applying appropriate colormap and cartographic context.

        The function handles setting up the Matplotlib figure, Cartopy projection, colormapping,
        and adding map features like coastlines, borders, and a custom colorbar.

        Parameters:
            gdf (gpd.GeoDataFrame, required):
                The GeoDataFrame to plot. Assumes point geometries that will be
                converted to small box polygons for visualization.
            value_col (str, required):
                The column name in the GeoDataFrame whose values determine the color.
            borders (bool, optional):
                Whether to draw country borders. Defaults to True.
            coastlines (bool, optional):
                Whether to draw coastlines. Defaults to True.
            gridlines (bool, optional):
                Whether to draw latitude/longitude gridlines and labels. Defaults to True.
            title (str | None, optional):
                The title of the plot. Defaults to None.
            legend (bool, optional):
                Whether to display the colorbar. Defaults to True.
            legend_title (str | None, optional):
                The title for the colorbar. Defaults to the value_col name.
            cmap (str, optional):
                The desired colormap type (e.g., "t2m", "tp", "anomaly") or a standard
                Matplotlib colormap name. Defaults to None (inferred from `value_col`).
            fig_size (tuple[int, int], optional):
                Matplotlib figure size (width, height) in inches. Defaults to (7, 5).
            polygons (list[Polygon] | None, optional)
                 A list of shapely Polygon objects to overlay on the map
                 (e.g., study area boundaries). Defaults to None.
            projection (cartopy.crs, optional):
                The Cartopy coordinate reference system for the map. Defaults to ccrs.PlateCarree().
            extends (tuple[float, float, float, float] | None, optional):
                The map extent as [lon_min, lon_max, lat_min, lat_max]. Defaults to None (auto-extent).
            dpi (int, optional):
                Dots per inch for the figure resolution. Defaults to 100.
            marker (str, optional):
                The marker style to use for plotting point data. Defaults to 's' (square).
            add_logos (bool, optional):
                Whether to add a custom image/logo below the plot. Defaults to True.
            polygon_color (str, optional):
                Color for the overlaid polygons. Defaults to 'cyan'.
            ax (matplotlib.axes.Axes, optional):
                An existing Matplotlib Axes object to plot onto. Defaults to None.

        Returns:
            tuple[matplotlib.figure.Figure, matplotlib.axes.Axes] | tuple[matplotlib.figure.Figure, matplotlib.axes.Axes, matplotlib.axes.Axes]:
                A tuple containing:
                - fig: The generated Matplotlib Figure.
                - ax: The Matplotlib Axes object with the map.
                - img_ax: The Axes object containing the added logo (only returned if `add_logos` is True).
        '''
        # get colormap
        gdfs_local = gdf.copy()

        if ax is None:
            fig, ax = plt.subplots(
                ncols = 1, nrows = 1, figsize = fig_size, dpi = dpi, 
                subplot_kw = {"projection" : projection}
                )
        else:
            fig = ax.figure

        if cmap == "anomaly" and value_col in ["tp"]:
            gdfs_local[value_col] = gdfs_local[value_col].clip(lower=-0.5, upper=None)

        # set color map   
        vmin = gdfs_local[value_col].min()
        vmax = gdfs_local[value_col].max()
        cmap, norm = Plot.get_colormap(cmap if cmap else value_col, vmin, vmax, value_col=value_col)

        # Plot the GeoDataFrame
        cell_size = 0.25  # degrees
        gdfs_local["geometry"] = gdfs_local.geometry.apply(
                    lambda p: box(p.x - cell_size/2, p.y - cell_size/2,
                                p.x + cell_size/2, p.y + cell_size/2)
            )
        gdfs_local.plot(
            ax=ax, column=value_col, cmap=cmap, norm=norm,
            legend=False, marker=marker
        )

        # Add contextily basemap
        if gridlines:
            ax.gridlines(crs=projection, linewidth=0.5, color='black', draw_labels=["bottom", "left"], alpha=0.2)
        # Add coastlines
        if coastlines:
            ax.coastlines()
        # Add contextily basemap
        if borders:
            ax.add_feature(cartopy.feature.BORDERS, lw = 1, alpha = 0.7, ls = "--", zorder = 99)

        # Draw polygons if provided
        if polygons is not None:
            for poly in polygons:
                x, y = poly.exterior.xy
                ax.plot(x, y, color=polygon_color, linewidth=2, transform=projection)

        if title is not None:
            ax.set_title(title, fontdict={'fontsize': 27, 'fontweight': 'bold'})

        # Set extent if provided
        if extends is not None:
            ax.set_extent(extends, crs=projection)

        # Custom colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm._A = []

        if hasattr(norm, "boundaries") and norm.boundaries is not None:
            ticks = norm.boundaries
            tick_labels = [int(b) for b in ticks]
        else:
            if cmap.name == "anomaly_cmap" and value_col in ["tp"]:
                ticks = np.linspace(norm.vmin, norm.vmax, len(ANOMALY_COLORS))
                for t in [0, -0.5]:
                    if t not in ticks:
                        ticks = np.sort(np.append(ticks, t))  # ensure 0 is in there
                tick_labels = [int(t) if abs(t) >= 1 else round(t, 1) for t in ticks]
            else:
                ticks = np.linspace(norm.vmin, norm.vmax, 11)
                tick_labels = [int(t) if abs(t) >= 1 else round(t, 1) for t in ticks]

        cbar = fig.colorbar(
            sm, ax=ax, orientation='vertical', location="right",
            fraction=0.04, pad=.04, aspect=25, ticks=ticks, shrink=0.82
        )

        if cmap.name == "anomaly_cmap" and value_col in ["tp"]:
            cbar.ax.set_ylim(-0.5, None)
                
        legend_title = legend_title if legend_title else value_col
        cbar.set_label(legend_title, labelpad=10, fontsize=20, weight='bold', color='#364563')
        cbar.set_ticklabels(tick_labels)
        plt.setp(plt.getp(cbar.ax.axes, 'xticklabels'),
                family=FONT_ROBOTO_CONDENSED_REGULAR.get_name(), fontsize=13)

        if add_logos:
            plt.close(fig)
            fig, img_ax = Plot.add_image_below(fig=fig, image_path=LOGO_HORIZON_PATH, pad_frac=0)
            return fig, ax, img_ax
        else:
            return fig, ax


    # adjust this so:
    ## shared colorbar is title + legend_title
    ## individual colorbars have legend_title and complete fig has title
    @staticmethod
    def subplot_gdf(gdfs:gpd.GeoDataFrame,
                    value_col:str,
                    legend_title:str, 
                    datetime_col:str='valid_time',
                    polygons:list[Polygon]=None,
                    ncols:int=5,
                    figsize:tuple[int, int]=(20, 12),
                    cmap:str=None,
                    borders:bool=True,
                    coastlines:bool=True,
                    gridlines:bool=True,
                    subtitle:str=None,
                    projection:cartopy.crs=ccrs.PlateCarree(),
                    extends:tuple[float, float, float, float]=None,
                    dpi:int=100,
                    flatten_empty_plots:bool=True,
                    marker:str='s',
                    shared_colorbar:bool=True,
                    add_logos:bool=True,
                    polygon_color='cyan'
    ) -> tuple[plt.Figure, np.ndarray[plt.Axes]] | tuple[plt.Figure, np.ndarray[plt.Axes], plt.Axes]:
        '''
        Generates a multi-panel subplot visualization of a GeoDataFrame, typically for time series data.

        The GeoDataFrame is grouped by unique dates in the `datetime_col`, and each resulting
        subset is plotted on its own subplot with shared or individual color scales.

        Parameters:
            gdfs (gpd.GeoDataFrame, required):
                The GeoDataFrame containing the data to plot, with a temporal column.
            value_col (str, required):
                The column name for the values to be colored.
            legend_title (str, required):
                The title for the shared or individual colorbar.
            datetime_col (str, optional):
                The column containing date/time information for grouping. Defaults to 'valid_time'.
            polygons (list[Polygon], optional):
                A list of shapely Polygon objects to overlay on each map (e.g., study area boundaries).
                Defaults to None.
            ncols (int, optional):
                The number of columns in the subplot grid. Defaults to 5.
            figsize (tuple[int, int], optional):
                Matplotlib figure size (width, height) in inches. Defaults to (20, 12).
            cmap (str, optional):
                The desired colormap type (e.g., "t2m", "tp", "anomaly") or a standard
                Matplotlib colormap name. Defaults to None (inferred from `value_col`).
            borders (bool, optional):
                Whether to draw country borders on each subplot. Defaults to True.
            coastlines (bool, optional):
                Whether to draw coastlines on each subplot. Defaults to True.
            gridlines (bool, optional):
                Whether to draw latitude/longitude gridlines and labels. Defaults to True.
            subtitle (str, optional):
                A main title for the entire figure (suptitle). Defaults to None.
            projection (cartopy.crs, optional):
                The Cartopy coordinate reference system for the map. Defaults to ccrs.PlateCarree().
            extends (tuple[float, float, float, float], optional):
                The map extent as [lon_min, lon_max, lat_min, lat_max]. Defaults to None (auto-extent).
            dpi (int, optional):
                Dots per inch for the figure resolution. Defaults to 100.
            flatten_empty_plots (bool, optional):
                If True, hides unused axes at the end of the grid. Defaults to True.
            marker (str, optional):
                The marker style for plotting point data. Defaults to 's' (square).
            shared_colorbar (bool, optional):
                If True, uses a single colorbar for the whole figure;
                otherwise, each subplot gets its own colorbar. Defaults to True.
            add_logos (bool, optional):
                Whether to add a custom image/logo below the plot. Defaults to True.
            polygon_color (str, optional):
                Color for the overlaid polygons. Defaults to 'cyan'.

        Returns:
            tuple[matplotlib.figure.Figure, np.ndarray[matplotlib.axes.Axes]] | tuple[matplotlib.figure.Figure, np.ndarray[matplotlib.axes.Axes], matplotlib.axes.Axes]:
                A tuple containing:
                - fig: The generated Matplotlib Figure.
                - axes: A flattened NumPy array of Matplotlib Axes objects for all subplots.
                - img_ax: The Axes object containing the added logo (only returned if `add_logos` is True).
        '''
        gdfs_local = gdfs.copy()
        # set cmap type
        cmap = cmap if cmap else value_col

        # Ensure datetime column is datetime type
        gdfs_local[datetime_col] = pd.to_datetime(gdfs_local[datetime_col])

        # Unique days sorted
        unique_days = sorted(gdfs_local[datetime_col].dt.date.unique())
        n_plots = len(unique_days)
        nrows = math.ceil(n_plots / ncols)

        # Create subplots with Cartopy projection
        fig, axes = plt.subplots(
            nrows, ncols, figsize=figsize, dpi=dpi,
            subplot_kw={'projection': projection},
        )
        axes = axes.flatten()

        if cmap == "anomaly" and value_col in ["tp"]:
            gdfs_local[value_col] = gdfs_local[value_col].clip(lower=-0.5, upper=None)

        # Shared color scale (only if shared_colorbar=True)
        if shared_colorbar:
            vmin = gdfs_local[value_col].min()
            vmax = gdfs_local[value_col].max()
            cmap, norm = Plot.get_colormap(cmap, vmin, vmax, value_col=value_col)


        for i, day in enumerate(unique_days):
            ax = axes[i]
            day_gdf = gdfs_local[gdfs_local[datetime_col].dt.date == day]

            # Individual scale if not shared
            if not shared_colorbar:
                vmin = day_gdf[value_col].min()
                vmax = day_gdf[value_col].max()
                cmap, norm = Plot.get_colormap(cmap, vmin, vmax, value_col=value_col)

            # Plot
            cell_size = 0.25  # degrees
            day_gdf["geometry"] = day_gdf.geometry.apply(
                    lambda p: box(p.x - cell_size/2, p.y - cell_size/2,
                                p.x + cell_size/2, p.y + cell_size/2)
                )
            
            day_gdf.plot(
                ax=ax, column=value_col, cmap=cmap,
                legend=False, vmin=vmin, vmax=vmax,
                marker=marker, norm=norm
            )

            if not shared_colorbar:
                # Add colorbar per axis
                #norm = Plot.cmap_norm_boundary(vmin, vmax, 11)

                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm._A = []
                cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.04, pad=0.04, ticks=norm.boundaries)
                cbar.set_label(legend_title)

            if gridlines:
                ax.gridlines(
                    crs=projection,
                    linewidth=0.5,
                    color='black',
                    draw_labels=["bottom", "left"],
                    alpha=0.2
                )
            if coastlines:
                ax.coastlines()
            if borders:
                ax.add_feature(cartopy.feature.BORDERS, lw=1, alpha=0.7, ls="--")

            if polygons is not None:
                for poly in polygons:
                    x, y = poly.exterior.xy
                    ax.plot(x, y, color=polygon_color, linewidth=2, transform=projection)

            ax.set_title(f"{day}", fontsize=18, weight='medium')

            if extends is not None:
                ax.set_extent(extends, crs=projection)

        fig.subplots_adjust(wspace=0.4)
        
        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(not flatten_empty_plots)

        # Shared colorbar if requested
        if shared_colorbar:
            #norm = Plot.cmap_norm_boundary(vmin, vmax, 11)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm._A = []

            # Handle any normalization type
            if hasattr(norm, "boundaries") and norm.boundaries is not None:
                # BoundaryNorm (discrete bins)
                ticks = norm.boundaries
                tick_labels = [int(b) for b in ticks]
            else:
                if cmap.name == "anomaly_cmap" and value_col in ["tp"]:
                    # Generate ticks that always include 0
                    ticks = np.linspace(norm.vmin, norm.vmax, len(ANOMALY_COLORS))
                    for t in [0, -0.5]:
                        if t not in ticks:
                            ticks = np.sort(np.append(ticks, t))  # ensure 0 is in there
                    tick_labels = [int(t) if abs(t) >= 1 else round(t, 1) for t in ticks]
                else:
                    # Normal case (temperature anomaly, etc.)
                    if vmin >= 0:
                        n_ticks = temperature_positive_cmap.N + 1
                    elif vmax <= 0:
                        n_ticks = temperature_negative_cmap.N + 1
                    else:
                        n_ticks = temperature_cmap.N + 1

                    ticks = np.linspace(norm.vmin, norm.vmax, n_ticks)
                    tick_labels = [round(t, 1) for t in ticks]

            cbar = fig.colorbar(
                sm, ax=axes.tolist(), 
                orientation='horizontal', location="top",
                fraction=0.01, pad=.07, aspect=60, ticks=ticks)
            
            if cmap.name == "anomaly_cmap" and value_col in ["tp"]:
                cbar.ax.set_xlim(-0.5, None)
            # Label for colorbar
            cbar.set_label(legend_title, labelpad=10, fontsize=27, weight='bold', color='#364563')
            # cbar.set_ticks(np.round(norm.boundaries).astype(int))
            # Make tick labels clean integers
        
            cbar.set_ticklabels(tick_labels)
            # Font settings
            plt.setp(plt.getp(cbar.ax.axes, 'xticklabels'), family=FONT_ROBOTO_CONDENSED_REGULAR.get_name(), fontsize=13)
            plt.show()
        
        if subtitle:
            fig.suptitle(subtitle)

        if add_logos:
            plt.close(fig)
            fig, img_ax = Plot.add_image_below(fig=fig, image_path=LOGO_HORIZON_PATH, pad_frac=-0.1)
            return fig, axes, img_ax
        else:
            return fig, axes




    @staticmethod
    def plot_poly(polygons:list[Polygon],
                  coords:list[list[float]],
                  layer:xr.DataArray=None,
                  cmap=None,
                  norm=None,
                  layer_type:str="elevation",
                  projection:cartopy.crs=ccrs.PlateCarree()
                ) -> tuple[plt.Figure, plt.Axes]:
        '''
        Plots geographical polygon boundaries over a base map, optionally displaying background raster data.

        This function is designed to visualize a selected region defined by a set of coordinates,
        along with specific polygons overlaid on a Cartopy-enabled map, using an Xarray DataArray
        for background visualization (e.g., elevation or climate classification).

        Parameters:
            polygons (list[Polygon]):
                A list of shapely Polygon objects to be plotted and highlighted.
            coords (list[list[float]]):
                A list of [longitude, latitude] pairs that define the overall extent of
                the region of interest for setting the map bounds and subsetting the layer.
            layer (xr.DataArray, optional):
                The 2D raster data to plot in the background (e.g., elevation, climate). Defaults to None.
            cmap (matplotlib.colors.Colormap, optional):
                Colormap object to use for the background layer, if applicable
                (especially for 'koppen' type). Defaults to None.
            norm (matplotlib.colors.Normalize, optional):
                Normalization object to use for the background layer, if applicable. Defaults to None.
            layer_type (str, optional):
                A key defining the type of background data ('elevation' or 'koppen')
                to determine default styling. Defaults to "elevation".
            projection (cartopy.crs, optional):
                The Cartopy coordinate reference system for the map. Defaults to ccrs.PlateCarree().

        Returns:
            tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
                A tuple containing:
                - fig: The generated Matplotlib Figure.
                - ax: The Matplotlib Axes object with the map.
        '''

        lons, lats = zip(*coords)
        min_lon, max_lon = min(lons), max(lons)
        min_lat, max_lat = min(lats), max(lats)

        if layer is not None:
            layer_subset = layer.sel(
                lon=slice(min_lon-3, max_lon+3),
                lat=slice(min_lat-3, max_lat+3)
            )

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': projection})
        ax.set_extent([min_lon - 3, max_lon + 3, min_lat - 3, max_lat + 3], crs=projection)
        ax.add_feature(cfeature.BORDERS, linestyle=':', alpha=0.5)
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.LAND, edgecolor='black')
        ax.gridlines(draw_labels=True)

        cax = inset_axes(
            ax, width="3%", height="100%", loc='center left',
            bbox_to_anchor=(1.1, 0., 1, 1), bbox_transform=ax.transAxes, borderpad=0
        )

        if layer is not None:
            if layer_type == "elevation":
                layer_subset.plot(
                    ax=ax, transform=projection, cmap="terrain",
                    cbar_ax=cax, cbar_kwargs={"label": "Elevation (m)"},
                    add_labels=False
                )
            elif layer_type == "koppen":
                layer_subset.plot(
                    ax=ax, transform=projection, cmap=cmap, norm=norm,
                    cbar_ax=cax, cbar_kwargs={"label": "Köppen-Geiger class"},
                    add_labels=False
                )

        ax.set_title("Selected region")

        # Plot selected polygons
        for polygon in polygons:
            x, y = polygon.exterior.xy
            ax.plot(x, y, color='red', linewidth=2, transform=projection)
            ax.fill(x, y, color='red', alpha=0.3, transform=projection)

        return fig, ax


    @staticmethod
    def plot_geometry(geom:Polygon | MultiPolygon | GeometryCollection,
                      ax:plt.Axes,
                      color:str='green',
                      alpha:float=0.3,
                      projection:cartopy.crs=ccrs.PlateCarree()):
        '''
        Recursively plots a single Shapely geometry (Polygon, MultiPolygon, or GeometryCollection) onto a Cartopy axis.

        This function handles various geometry types by drawing the exterior line and filling the interior,
        making it suitable for visualizing area boundaries on maps.

        Parameters:
            geom (shapely.Polygon | shapely.MultiPolygon | shapely.GeometryCollection, required):
                The Shapely geometry object to plot. Supports Polygon, MultiPolygon,
                and GeometryCollection containing these.
            ax (matplotlib.axes.Axes, required):
                The Matplotlib Axes object (must have a Cartopy projection).
            color (str, optional):
                The color for the boundary line and fill. Defaults to 'green'.
            alpha (float, optional):
                The transparency level for the fill color (0.0 to 1.0). Defaults to 0.3.
            projection (cartopy.crs, optional):
                The Cartopy coordinate reference system for the data to ensure correct plotting.
                Defaults to ccrs.PlateCarree().

        Raises:
            RecursionError:
                If the GeometryCollection contains geometries that lead to excessive
                recursive calls (unlikely for standard geographic data).
            TypeError:
                If an unsupported geometry type is provided.
        '''
        if isinstance(geom, Polygon):
            x, y = geom.exterior.xy
            ax.plot(x, y, color=color, linewidth=2, transform=projection)
            ax.fill(x, y, color=color, alpha=alpha, transform=projection)
        elif isinstance(geom, MultiPolygon):
            for poly in geom.geoms:
                Plot.plot_geometry(poly, ax, color=color, alpha=alpha)
        elif isinstance(geom, GeometryCollection):
            for subgeom in geom.geoms:
                if isinstance(subgeom, (Polygon, MultiPolygon, GeometryCollection)):
                    Plot.plot_geometry(subgeom, ax, color=color, alpha=alpha)
                else:
                    # Optionally handle or ignore other geometry types
                    pass
        else:
            raise TypeError(f"Unsupported geometry type: {type(geom)}")


    @staticmethod
    def elevation_region(data:dict,
                         polygons:list[Polygon],
                         elevation:xr.DataArray,
                         threshold:int=None, 
                         projection:cartopy.crs=ccrs.PlateCarree()
                         ) -> tuple[plt.Figure, plt.Axes, list[Polygon]]:
        '''
        Adjusts input regions by an elevation threshold and visualizes the result on a map.

        This function takes geographical regions (polygons defined in a GeoJSON-like dictionary),
        filters them to only include areas where the background elevation is below a given
        threshold, and plots the original and adjusted regions over the elevation map.

        Parameters:
            data (dict):
                A GeoJSON-like dictionary containing feature geometries (polygons) under the 'features' key.
            polygons (list[Polygon]):
                A list of original shapely Polygon objects derived from `data`.
            elevation (xr.DataArray):
                An Xarray DataArray containing elevation data (lon, lat coordinates)
                to be used for thresholding and background plotting.
            threshold (int, optional):
                The maximum elevation value (in meters) to retain in the adjusted regions.
                Defaults to 100000 (effectively no threshold).
            projection (cartopy.crs, optional):
                The Cartopy coordinate reference system for plotting. Defaults to ccrs.PlateCarree().

        Returns:
            tuple[matplotlib.figure.Figure, matplotlib.axes.Axes, list[Polygon]]:
                A tuple containing:
                - fig: The generated Matplotlib Figure.
                - ax: The Matplotlib Axes object with the map.
                - adjusted_polygons: A list of new shapely Polygon objects representing the
                areas of the original polygons that are below the elevation threshold.
        '''
        threshold = threshold if threshold else 100000

        all_coords = []
        adjusted_polygons = []

        for feature in data["features"]:

            coords = feature['geometry']['coordinates'][0]
            all_coords.extend(coords)
            poly = Polygon(coords)  
            minx, miny, maxx, maxy = poly.bounds
            elev_subset = elevation.sel(
                lon=slice(minx-0.5, maxx+0.5),
                lat=slice(miny-0.5, maxy+0.5)  
            )   
            elev_vals = elev_subset.squeeze().values

            if elev_subset.lat.values[0] < elev_subset.lat.values[-1]:
                elev_vals = elev_vals[::-1, :]  
                lat = elev_subset.lat.values[::-1]
            else:
                lat = elev_subset.lat.values

            lon = elev_subset.lon.values    
            lon2d, lat2d = np.meshgrid(lon, lat)    
            below_thresh = elev_vals <= threshold   
            inside_poly = contains_xy(poly, lon2d, lat2d)   
            final_mask = below_thresh & inside_poly

            transform = rasterio.transform.from_bounds(
                lon.min(), lat.min(),   
                lon.max(), lat.max(),   
                len(lon), len(lat)
            )   
            from shapely.geometry import shape

            for geom, val in rasterio.features.shapes(
                    final_mask.astype(np.uint8),
                    mask=final_mask,
                    transform=transform):
                
                if val == 1:
                    new_poly = shape(geom)
                    clipped_poly = new_poly.intersection(poly)

                    if not clipped_poly.is_empty:
                        if clipped_poly.geom_type == "Polygon":
                            adjusted_polygons.append(clipped_poly)
                        elif clipped_poly.geom_type == "MultiPolygon":
                            adjusted_polygons.extend(list(clipped_poly.geoms))

        lons, lats = zip(*all_coords)
        min_lon = min(lons)
        max_lon = max(lons)
        min_lat = min(lats)
        max_lat = max(lats)

        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': projection})

        ax.set_title(f"Selected regions under {threshold} m elevation")
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.LAND, edgecolor='black')
        ax.set_extent([min_lon - 3, max_lon + 3, min_lat - 3, max_lat + 3], crs=projection)
        ax.gridlines(draw_labels=True)

        cax = inset_axes(
            ax,
            width="3%", height="100%",
            loc='center left',
            bbox_to_anchor=(1.1, 0., 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0
        )

        elev_plot = elevation.plot(
            ax=ax,
            transform=projection,
            cmap="terrain",
            cbar_ax=cax,
            cbar_kwargs={"label": "Elevation (m)"},
            add_colorbar=True,
            add_labels=False,
            vmin=0, 
            vmax=threshold  
        )

        for poly in polygons:
            x, y = poly.exterior.xy
            ax.plot(x, y, color='red', linewidth=2, transform=projection)

        for geom in adjusted_polygons:
            Plot.plot_geometry(geom, ax)

        return fig, ax, adjusted_polygons




    @staticmethod
    def plot_timeserie(data, value_col:str, title:str, x_label:str, y_label:str, datetime_col:str='valid_time', 
                    fig_size:tuple=(12,6), dpi:int=100, show_grid:bool=True, line_style:str=':', marker_style:str=None, 
                    draw_style:str='default', label_rotation:int=0, line_width:float=1.5, labelticks:list[str]=None, labels:list[str]=None,
                    add_logos:bool=True, center_month_labels:bool=False, full_month_names:bool=False, ax=None, ci:bool = False):
        '''
        Plots a time series from a DataFrame column.

        The function sets up a Matplotlib figure/axis and plots the specified value column
        against the datetime column, applying various styling and formatting options.

        Parameters:
            data (pd.DataFrame, required):
                The DataFrame containing the time series data.
            value_col (str, required):
                The column name containing the values to plot on the y-axis.
            title (str, required):
                The title of the plot.
            x_label (str, required):
                The label for the x-axis (time/date).
            y_label (str, required):
                The label for the y-axis (value_col).
            datetime_col (str, optional):
                The column name containing datetime objects. Defaults to 'valid_time'.
            fig_size (tuple, optional):
                Matplotlib figure size (width, height) in inches. Defaults to (12, 6).
            dpi (int, optional):
                Dots per inch for the figure resolution. Defaults to 100.
            show_grid (bool, optional):
                Whether to display a grid on the plot. Defaults to True.
            line_style (str, optional):
                Matplotlib line style (e.g., '-', '--', ':'). Defaults to ':'.
            marker_style (str or dict, optional):
                Matplotlib marker style or a dict of keyword arguments for plotting markers. Defaults to None.
            draw_style (str, optional):
                Matplotlib draw style (e.g., 'default', 'steps'). Defaults to 'default'.
            label_rotation (int, optional):
                Rotation angle for x-axis tick labels. Defaults to 0.
            line_width (float, optional): 
                The width of the plotted line. Defaults to 1.5.
            labelticks (list[str], optional):
                Custom locations for x-axis ticks. Defaults to None.
            labels (list[str], optional):
                Custom labels for the x-axis ticks. Defaults to None.
            add_logos (bool, optional):
                Whether to add a custom image/logo below the plot. Defaults to True.
            center_month_labels (bool, optional):
                If True, centers month labels under the 15th of the month.
                Requires x-axis to be datetime. Defaults to False.
            full_month_names (bool, optional):]
                If True, uses full month names (e.g., 'January') when
                `center_month_labels` is True. Defaults to False.
            ax (matplotlib.axes.Axes, optional):
                An existing Matplotlib Axes object to plot onto. Defaults to None.
            ci (bool):
                Plot confidence interval shading if True. Expects columns {value_col}_ci_lower
                and {value_col}_ci_upper

        Returns:
            tuple[matplotlib.figure.Figure, matplotlib.axes.Axes, matplotlib.axes.Axes | None]:
                A tuple containing:
                - fig: The generated Matplotlib Figure.
                - ax: The Matplotlib Axes object with the time series plot.
                - img_ax: The Axes object containing the added logo, or None if `add_logos` is False.
        '''

        # plot_df = data.copy()
        # plot_df[datetime_col] = pd.to_datetime(plot_df[datetime_col])
        # plot_df["plot_time"] = plot_df[datetime_col]

        # start_month, end_month = month_range

        # # Determine if the period crosses the year boundary
        # crosses_year = (end_month < start_month)

        # # Adjust months so they plot in correct chronological order
        # if crosses_year:
        #     # For ranges like (7,6) or (9,3): shift early months (those before start_month) forward by one year
        #     early_mask = plot_df["plot_time"].dt.month < start_month
        #     plot_df.loc[early_mask, "plot_time"] += pd.DateOffset(years=1)

        # # Sort chronologically after shifting
        # plot_df = plot_df.sort_values("plot_time").reset_index(drop=True)

        # # ----- Create label ticks -----
        # # Define the logical start month (the first month of the period)
        # if crosses_year:
        #     # e.g. (7,6) → start in July 2024 and wrap to June 2025
        #     label_start = pd.Timestamp(f"2024-{start_month:02d}-01")
        # else:
        #     # e.g. (1,6) or (3,9): simple one-year span
        #     label_start = pd.Timestamp(f"2024-{start_month:02d}-01")

        # # Always 12 months long
        # labelticks = pd.date_range(label_start, periods=12, freq="MS")
        # labels = labelticks.strftime("%b")


        #set font family globally
        if ax is None:
            fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
        else:
            fig = ax.figure

        ax.plot(data[datetime_col], data[value_col], 
                color='darkblue', 
                linewidth=line_width, 
                linestyle=line_style, 
                drawstyle=draw_style,
                **(marker_style if marker_style is not None else {})
                )
        
        if ci:
            ax.fill_between(
                    data[datetime_col],
                    data[f"{value_col}_ci_lower"],
                    data[f"{value_col}_ci_upper"],
                    color='lightblue',  # or customize per variable
                    alpha=0.3,
                    label="(95% CI)"
                )

        ax.set_title(label=title)
        ax.set_xlabel(xlabel=x_label)
        ax.set_ylabel(ylabel=y_label)

        if labelticks is not None:
            ax.set_xticks(labelticks)
        if labels is not None:
            ax.set_xticklabels(labels)

        # Center month labels
        if center_month_labels:
            ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=15))
            fmt = "%B" if full_month_names else "%b"
            ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))

        if show_grid:
            ax.grid(True)

        # Format x-axis with date labels
        #ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        fig.autofmt_xdate(rotation=label_rotation)

        # plt.tight_layout()
        # plt.show()
        if add_logos:
            plt.close(fig)
            fig, img_ax = Plot.add_image_below(fig=fig, image_path=LOGO_HORIZON_PATH, pad_frac=-.1)
            return fig, ax, img_ax
        else:
            img_ax = None
            return fig, ax, img_ax





    @staticmethod
    def plot_n_days(
        rolled_data_list:list[gpd.GeoDataFrame], value_col:str, parameter:str, event_date:datetime,
        labelticks:list[int], labels:list[any], days:list[int], title:str,
        datetime_col:str="valid_time", add_logos:bool=True, fig_height:int=3, xtick_rotation:int=0, ncols:int=0
    ):
        '''
        Plots multiple time series showing rolling data accumulations over different day windows.

        Each subplot displays the time series of the specified variable aggregated over a rolling
        window (`ndays`). All historical years are plotted faintly, while the current (event)
        year is plotted prominently up to the event date.

        Parameters:
            rolled_data_list (list[gpd.GeoDataFrame], required):
                A list where each element is a GeoDataFrame containing the rolling
                accumulation data for a specific time window.
            value_col (str, required):
                The column name containing the accumulated values to plot on the y-axis.
            parameter (str, required):
                The name of the parameter being plotted (e.g., 'Precipitation').
                Used internally for logic, but not directly in the docstring.
            event_date (datetime, required):
                The date of the event, used to highlight the current year's data up to this point.
            labelticks (list[int], required):
                Day-of-year integers to use as x-axis tick locations.
            labels (list[any], required):
                Labels corresponding to `labelticks` (e.g., month abbreviations).
            days (list[int], required):
                A list of rolling window sizes (e.g., [3, 7, 14]) corresponding to the data in `rolled_data_list`.
            title (str, required):
                The base title for the subplots (e.g., 'Accumulation').
            datetime_col (str, optional):
                The column name containing datetime objects. Defaults to "valid_time".
            add_logos (bool, optional):
                Whether to add a custom image/logo below the plot. Defaults to True.
            fig_height (int, optional):
                The height (in inches) of each row of subplots. Defaults to 3.
            xtick_rotation (int, optional):
                Rotation angle for x-axis tick labels. Defaults to 0.
            ncols (int, optional):
                The number of columns in the subplot grid. If 0, it defaults to
                the total number of plots. Defaults to 0.

        Returns:
            tuple[matplotlib.figure.Figure, matplotlib.axes.Axes, matplotlib.axes.Axes | None]:
                A tuple containing:
                - fig: The generated Matplotlib Figure.
                - ax: The last Matplotlib Axes object used for plotting (or the single Axes if only one plot).
                - img_ax: The Axes object containing the added logo, or None if `add_logos` is False.
        '''

        # fig size
        nplots = len(rolled_data_list)
        ncols = ncols if ncols else nplots
        nrows = math.ceil(nplots/ncols)

        if value_col == 'tp':
            fig, axs = plt.subplots(
                ncols=ncols,
                nrows=nrows,
                figsize=(5 * ncols, fig_height * nrows),
                dpi=100,
                sharey=False
            )
        else:
            fig, axs = plt.subplots(
                ncols=ncols,
                nrows=nrows,
                figsize=(5 * ncols, fig_height * nrows),
                dpi=100,
                sharey=True
            )

        axs = np.array(axs).reshape(-1)

        if len(rolled_data_list) == 1:
            axs = [axs]  # make iterable if only one axis

        event_date = pd.to_datetime(event_date)
        event_year = event_date.year

        for ax, data_nday, ndays in zip(axs, rolled_data_list, days):
            # Plot all years EXCEPT event year in blue
            for y in data_nday[datetime_col].dt.year.unique():
                if y == event_year:
                    continue
                data_y = data_nday[data_nday[datetime_col].dt.year == y]
                ax.plot(
                    data_y[datetime_col].dt.dayofyear,
                    data_y[value_col],
                    color="tab:blue",
                    alpha=0.3
                )

            # Plot event year only up to event_date in black
            data_event = data_nday[
                (data_nday[datetime_col].dt.year == event_year) &
                (data_nday[datetime_col] <= event_date)
            ]
            ax.plot(
                data_event[datetime_col].dt.dayofyear,
                data_event[value_col],
                color="k"
            )

            # Style
            ax.set_xticks(labelticks)
            ax.set_xticklabels(labels, rotation=xtick_rotation)
            ax.grid(axis="x", color="k", alpha=0.2) # set vertical grid lines
            ax.set_title(f"{ndays}-day {title}", fontsize=18)

            # Highlight date window
            ylim = ax.get_ylim()
            dayofyear = event_date.dayofyear
            ax.add_patch(Rectangle((dayofyear, ylim[0]), -15, 10000, color="gold", alpha=0.3))
            ax.set_ylim(ylim)


        fig.subplots_adjust(hspace=0.4)

        for ax in axs[nplots:]:
            fig.delaxes(ax)
        axs = axs[:nplots]

        if add_logos:
            plt.close(fig)
            fig, img_ax = Plot.add_image_below(fig=fig, image_path=LOGO_HORIZON_PATH, pad_frac=0)
            return fig, ax, img_ax
        else:
            return fig, ax



    @staticmethod
    def subplot_contours(contour_gdf:gpd.GeoDataFrame,
                         gdf:gpd.GeoDataFrame,
                         value_col:str,
                         contour_col:str,
                         legend_title:str=None,
                         datetime_col:str="valid_time",
                         polygons:list[Polygon]=None,
                         ncols:int=5,
                         figsize:tuple[int,int]=(13,10),
                         cmap:str|None=None,
                         borders:bool=True,
                         coastlines:bool=True,
                         gridlines:bool=True,
                         subtitle:str=None,
                         extends:tuple[float,float,float,float]=None,
                         dpi:int=100,
                         flatten_empty_plots:bool=True,
                         marker:str='s',
                         shared_colorbar:bool=True,
                         add_logos:bool=False,
                         polygon_color:str='cyan',
                         contour_steps:int=200,
                         projection:cartopy.crs=ccrs.PlateCarree(),
                         grid_line_col:str='gray',
                         grid_line_size:float=.4,
                         grid_line_alpha:float=.5
    ) -> tuple[plt.Figure, np.ndarray[plt.Axes]] | tuple[plt.Figure, np.ndarray[plt.Axes], plt.Axes]:
        '''
        Plots daily GeoDataFrame values in a multi-panel grid, overlaid with atmospheric height contours (e.g., Z500).

        The function groups the point data (`gdf`) and the contour data (`contour_gdf`) by date,
        creating one subplot per day. It uses a custom map projection (Lambert Conformal) centered
        on the data and supports shared or individual colorbars.

        Parameters:
            contour_gdf (gpd.GeoDataFrame, required):
                GeoDataFrame containing the contour data (e.g., Z500 heights) on a regular lat/lon grid.
            gdf (gpd.GeoDataFrame, required):
                GeoDataFrame containing the primary point data to be colored.
            value_col (str, required):
                The column in `gdf` whose values determine the color.
            contour_col (str, required):
                The column in `contour_gdf` whose values are used for contours (e.g., 'z').
            legend_title (str, optional):
                The title for the shared colorbar. Defaults to the value_col name.
            datetime_col (str, optional):
                The column containing date/time information for grouping. Defaults to "valid_time".
            polygons (list[Polygon], optional):
                A list of shapely Polygon objects to overlay on the map. Defaults to None.
            ncols (int, optional):
                The number of columns in the subplot grid. Defaults to 5.
            figsize (tuple[int, int], optional):
                Matplotlib figure size (width, height) in inches. Defaults to (13, 10).
            cmap (str | None, optional):
                The desired colormap type (e.g., "t2m", "tp", "anomaly") or a
                standard Matplotlib colormap name. Defaults to None (inferred from `value_col`).
            borders (bool, optional):
                Whether to draw country borders. Defaults to True.
            coastlines (bool, optional):
                Whether to draw coastlines. Defaults to True.
            gridlines (bool, optional):
                Whether to draw lat/lon gridlines. Defaults to True.
            subtitle (str, optional):
                A main title for the entire figure (suptitle). Defaults to None.
            extends (tuple[float, float, float, float], optional):
                The map extent as [lon_min, lon_max, lat_min, lat_max]. Defaults to None (auto-extent).
            dpi (int, optional):
                Dots per inch for the figure resolution. Defaults to 100.
            flatten_empty_plots (bool, optional):
                If True, hides unused axes at the end of the grid. Defaults to True.
            marker (str, optional):
                The marker style for plotting point data. Defaults to 's' (square).
            shared_colorbar (bool, optional):
                If True, uses a single colorbar for the whole figure. Defaults to True.
            add_logos (bool, optional):
                Whether to add a custom image/logo below the plot. Defaults to False.
            polygon_color (str, optional):
                Color for the overlaid polygons. Defaults to 'cyan'.
            contour_steps (int, optional):
                The interval between contour lines (e.g., 200 meters for Z500). Defaults to 200.
            projection (cartopy.crs, optional):
                The Cartopy CRS for plotting data. Defaults to ccrs.PlateCarree().
            grid_line_col (str, optional):
                Color of the gridlines. Defaults to 'gray'.
            grid_line_size (float, optional):
                Linewidth of the gridlines. Defaults to 0.4.
            grid_line_alpha (float, optional):
                Transparency of the gridlines. Defaults to 0.5.

        Returns:
            tuple[matplotlib.figure.Figure, np.ndarray[matplotlib.axes.Axes]] | tuple[matplotlib.figure.Figure, np.ndarray[matplotlib.axes.Axes], matplotlib.axes.Axes]:
                A tuple containing:
                - fig: The generated Matplotlib Figure.
                - axes: A flattened NumPy array of Matplotlib Axes objects for all subplots.
                - img_ax: The Axes object containing the added logo (only returned if `add_logos` is True).
        '''

        # set cmap type
        cmap = cmap if cmap else value_col

        # Ensure datetime
        gdf[datetime_col] = pd.to_datetime(gdf[datetime_col])
        contour_gdf[datetime_col] = pd.to_datetime(contour_gdf[datetime_col])

        # Unique sorted days
        unique_days = sorted(contour_gdf[datetime_col].dt.date.unique())
        n_plots = len(unique_days)
        nrows = math.ceil(n_plots / ncols)

        # Projection fixed to LambertConformal
        mid_lon = contour_gdf["longitude"].mean()
        mid_lat = contour_gdf["latitude"].mean()
        proj = ccrs.LambertConformal(central_longitude=mid_lon, central_latitude=mid_lat)

        # Figure
        fig, axes = plt.subplots(
            nrows, ncols, figsize=figsize, dpi=dpi,
            subplot_kw={"projection": proj}
        )
        axes = axes.flatten()

        # Shared color scale
        if shared_colorbar:
            vmin = gdf[value_col].min()
            vmax = gdf[value_col].max()
            cmap, norm = Plot.get_colormap(cmap, vmin, vmax, value_col=value_col)
        else:
            norm = None

        # contour contour levels
        # Get raw min and max
        zmin_raw, zmax_raw = contour_gdf[contour_col].min(), contour_gdf[contour_col].max()

        # Round outward to nearest step
        zmin = np.floor(zmin_raw / contour_steps) * contour_steps
        zmax = np.ceil(zmax_raw / contour_steps) * contour_steps

        # Create clean range
        z_lev = np.arange(zmin, zmax + contour_steps, contour_steps)

        for i, day in enumerate(unique_days):
            ax = axes[i]
            day_gdf = gdf[gdf[datetime_col].dt.date == day]

            if not shared_colorbar:
                vmin = day_gdf[value_col].min()
                vmax = day_gdf[value_col].max()
                cmap, norm = Plot.get_colormap(cmap, vmin, vmax, value_col=value_col)

            if not day_gdf.empty:
                cell_size = 0.25  # degrees
                day_gdf["geometry"] = day_gdf.geometry.apply(
                    lambda p: box(p.x - cell_size/2, p.y - cell_size/2,
                                p.x + cell_size/2, p.y + cell_size/2)
                )
                day_gdf.plot(
                    ax=ax, column=value_col, cmap=cmap,
                    legend=False, vmin=vmin, vmax=vmax,
                    norm=norm, marker=marker,
                    transform=projection
                )

            # --- Z500 contours ---
            contour_day = contour_gdf[contour_gdf[datetime_col].dt.date == day]
            if not contour_day.empty:
                pivot = contour_day.pivot_table(index="latitude", columns="longitude", values=contour_col)
                lon, lat, Z = pivot.columns.values, pivot.index.values, pivot.values
                cn = ax.contour(lon, lat, Z, z_lev, colors="black", linewidths=0.4, transform=projection)
                ax.clabel(cn, inline=1)

            if not shared_colorbar:
                # Add colorbar per axis
                #norm = Plot.cmap_norm_boundary(vmin, vmax, 11)
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm._A = []
                cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.04, pad=0.04, ticks=norm.boundaries)
                cbar.set_label(legend_title)

            # Map features

            if gridlines:
                gl = ax.gridlines(
                    draw_labels=False, x_inline=False, y_inline=False,
                    linewidth=grid_line_size, color=grid_line_col, alpha=grid_line_alpha
                )
                gl.right_labels = gl.top_labels = False
                # gl.xlabel_style = {"size": 8, "color": grid_line_col}
                # gl.ylabel_style = {"size": 8, "color": grid_line_col}

            if coastlines:
                ax.coastlines(resolution="50m", color="black", linewidth=0.5, alpha=0.7)
            if borders:
                ax.add_feature(cfeature.BORDERS.with_scale('50m'), edgecolor="black", linewidth=0.5)

            if polygons is not None:
                for poly in polygons:
                    x, y = poly.exterior.xy
                    ax.plot(x, y, color=polygon_color, linewidth=2, transform=projection)

            ax.set_title(f"{day}", fontsize=18, weight='medium')

            if extends is not None:
                ax.set_extent(extends, crs=projection)


        fig.subplots_adjust(hspace=.25, wspace=.05)
        fig.tight_layout(pad=0.5)

        # Hide empty plots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(not flatten_empty_plots)

        # Shared colorbar
        if shared_colorbar:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm._A = []        
            cbar = fig.colorbar(sm, ax=axes.tolist(), orientation='horizontal', location="top",
                                fraction=0.01, pad=.07, aspect=60, ticks=norm.boundaries)
            cbar.set_label(legend_title if legend_title else value_col,
                        labelpad=10, fontsize=27, weight='bold', color='#364563')
            plt.setp(plt.getp(cbar.ax.axes, 'xticklabels'), family=FONT_ROBOTO_CONDENSED_REGULAR.get_name(), fontsize=13)
            plt.show()

        if subtitle:
            fig.suptitle(subtitle, y=1.05)

        # fig.subplots_adjust(wspace=0.25, hspace=.45, top=.85)
        fig.tight_layout()

        if add_logos:
            plt.close(fig)
            fig, img_ax = Plot.add_image_below(fig=fig, image_path=LOGO_HORIZON_PATH, pad_frac=-0.1)
            return fig, axes, img_ax
        else:
            return fig, axes
    
    @staticmethod
    def plot_cordex_map(gdf, domains_dict, bbox, study_region, mapproj, selected_domain=None, add_logos=True):
        """
        Plots CORDEX domains, highlighting the selected domain,
        the bounding box, and the study region.
        """
        # Setup the figure
        fig, ax = plt.subplots(figsize=(16, 9), dpi=100, 
                            subplot_kw={"projection": mapproj})

        ax.set_global() 
        ax.coastlines(resolution='110m', color='black', linewidth=0.5)
        ax.stock_img()

        legend_handles = []

        # Plot ALL domains
        for r in domains_dict.keys():
            dom_row = gdf.loc[[r]]
            
            # Style logic
            is_selected = (r == selected_domain)
            line_width = 4 if is_selected else 1.2
            alpha_val = 1.0 if is_selected else 0.3
            z_order = 5 if is_selected else 3
            
            native_projection = domains_dict[r]["projection"]
            color = domains_dict[r]["colour"]
            
            # Plot the boundary
            boundary = dom_row.to_crs(native_projection).boundary
            
            # Base plot for all domains
            boundary.plot(ax=ax, 
                        transform=native_projection, 
                        color=color, 
                        linewidth=2 if not is_selected else line_width, 
                        alpha=0.5 if not is_selected else alpha_val,
                        zorder=z_order)

            # Update legend
            label_text = f"{r}: {domains_dict[r]['long_name']}"
            if is_selected:
                label_text += " (Recommended)"
            
            legend_handles.append(mpatches.Patch(color=color, alpha=alpha_val, label=label_text))

        # Plot the Bounding Box (bbox) with the marker 'o'
        ax.plot([bbox[0], bbox[2], bbox[2], bbox[0], bbox[0]], 
                [bbox[1], bbox[1], bbox[3], bbox[3], bbox[1]], 
                color="darkred", lw=2, alpha=1, marker='o', transform=mapproj, zorder=10)

        # Plot the Study Region
        study_region.boundary.plot(ax=ax, transform=mapproj, color="green", lw=2, alpha=1.0, zorder=11)

        # Add Custom Line Legend Items
        legend_handles.append(mlines.Line2D([], [], color='green', lw=2, label='Study Region'))
        legend_handles.append(mlines.Line2D([], [], color='darkred', lw=2, marker='o', label='Bounding Box'))

        # Dynamic Zoom Logic
        if selected_domain:
            bounds = gdf.loc[selected_domain, 'geometry'].bounds
            pad = 10
            ax.set_extent([bounds[0]-pad, bounds[2]+pad, bounds[1]-pad, bounds[3]+pad], crs=mapproj)
        else:
            buffer = 30 
            ax.set_extent([bbox[0]-buffer, bbox[2]+buffer, bbox[1]-buffer, bbox[3]+buffer], crs=mapproj)

        # Final Formatting
        gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.2)
        gl.top_labels = False
        gl.right_labels = False

        
        leg = plt.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.02, 1), 
                        title="CORDEX Domains", fontsize='small')
        for text in leg.get_texts():
            if "(Recommended)" in text.get_text():
                text.set_weight("bold")
                text.set_size("medium")

        plt.title("CORDEX Domains & Study Area", fontsize=18, pad=20)
        plt.tight_layout()

        # Logos and Return
        if add_logos:
            plt.close(fig)
            fig, img_ax = Plot.add_image_below(fig=fig, image_path=LOGO_HORIZON_PATH, pad_frac=-0.02)
            return fig, ax, img_ax
        else:
            return fig, ax

    @staticmethod
    def plot_koppen_geiger(
        kg_da: xr.DataArray,
        polygons: list[Polygon],
        coords: list[list[float]],
        legend_path: str,
        projection=ccrs.PlateCarree(),
        fontsize: int = 8,
        figsize=(10, 10),
        extra_polygons: list[Polygon] = None
    ) -> tuple[plt.Figure, plt.Axes]:
        '''
        Plots Köppen–Geiger climate classifications for a selected region and polygons.

        The function subsets the climate data based on the provided coordinates, plots the
        classified climate as a background raster, overlays the input polygons (and optionally
        extra/original polygons), and includes a custom, grouped Köppen-Geiger legend at the bottom.

        Parameters:
            kg_da (xr.DataArray, required):
                The Xarray DataArray containing the Köppen–Geiger codes (raster data).
            polygons (list[Polygon], required):
                A list of shapely Polygon objects representing the primary region(s)
                to be highlighted/outlined (e.g., the final, filtered region).
            coords (list[list[float]], required):
                A list of [longitude, latitude] pairs that define the overall
                bounding box for the plot extent and raster subsetting.
            legend_path (str, required):
                The file path to the CSV or text file containing the Köppen–Geiger
                legend mapping (codes, names, and RGB colors).
            projection (cartopy.crs, optional):
                The Cartopy coordinate reference system for the map. Defaults to ccrs.PlateCarree().
            fontsize (int, optional):
                Base font size for plot elements, including the legend. Defaults to 8.
            figsize (tuple, optional):
                Matplotlib figure size (width, height) in inches. Defaults to (10, 10).
            extra_polygons (list[Polygon], optional):
                A list of additional polygons to plot, typically representing the original, unfiltered region.
                These are filled and outlined in green (green fill, green line). Defaults to None.

        Returns:
            tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
                A tuple containing:
                - fig: The generated Matplotlib Figure.
                - ax: The Matplotlib Axes object with the climate map.
        '''
        
        kg_legend = KoppenGeiger.load_kg_legend(legend_path)
        kg_da_masked = kg_da.where(kg_da >= 1)


        # Build listed colormap & norm
        kg_colors = [tuple(rgb) for rgb in kg_legend["rgb"]]
        kg_cmap = mcolors.ListedColormap(kg_colors)
        kg_norm = mcolors.BoundaryNorm(list(kg_legend["code"]) + [31], kg_cmap.N)

        fig, ax = Plot.plot_poly(polygons, coords, layer=kg_da_masked, cmap=kg_cmap, norm=kg_norm, layer_type="koppen")
        if extra_polygons is not None and len(extra_polygons) > 0:
            for poly in extra_polygons:
                x, y = poly.exterior.xy
                ax.fill(
                    x, y,
                    color="green", alpha=0.1,
                    transform=projection,
                    zorder=1  # lower -> below main polygons
                )
                ax.plot(
                    x, y,
                    color="green", linewidth=2,
                    transform=projection,
                    label="Original region",
                    zorder=1
                )
        handles = [
            mpatches.Rectangle((0, 0), 1, 1, facecolor="white", linewidth=2, edgecolor="green", label="Original region"),
            mpatches.Rectangle((0, 0), 1, 1, facecolor="white", linewidth=2, edgecolor="red", label="Filtered region")
        ]

        ax.legend(
            handles=handles,
            loc="upper right",
            frameon=True,
            fontsize=fontsize,
            title="Regions",
            title_fontsize=fontsize + 1
        )

        KoppenGeiger.draw_koppen_legend(fig, kg_legend)

        return fig, ax
    
    @staticmethod
    def plot_seasonal_cycles(seasonal_cycles, obs_seasonal_cycle, value_col:str,
                          legend_title:str=None, title:str=None, cmap:str=None, add_logos:bool=True,
                          projection:cartopy.crs=ccrs.PlateCarree(), dpi:int=100,
                          subtitle:bool=True):
        
        '''
        Plots seasonal cycles for multiple models alongside observational data.
        Each subplot displays the seasonal cycle of a specific model compared to
        the ERA5 observational seasonal cycle.

        Parameters:
            seasonal_cycles (dict, required):
                A dictionary where keys are model names and values are xarray DataArrays
                containing the seasonal cycle data for each model.
            obs_seasonal_cycle (xarray.Dataset, required):
                An xarray Dataset containing the observational seasonal cycle data (ERA5).
            value_col (str, required):
                The variable name/column in the datasets to plot (e.g., 't2m', 'tp').
            legend_title (str | None, optional):
                The title for the overall figure legend. Defaults to None.
            add_logs (bool, optional):
                Whether to add a custom image/logo below the plot. Defaults to True.
            subtitle (bool, optional):
                Whether to add a subtitle to the figure. Defaults to True.

        Returns:
            tuple[matplotlib.figure.Figure, matplotlib.axes.Axes, matplotlib.axes.Axes | None]:
                A tuple containing:
                - fig: The generated Matplotlib Figure.
                - ax: The Matplotlib Axes object with the seasonal cycle plots.
                - img_ax: The Axes object containing the added logo, or None if `add_logos` is False.
        '''

        n_models = len(seasonal_cycles)
        fig, axs, axs_flat, nrows, ncols = Plot.create_grid(n_models, sharex=True, sharey=True)

        ticks, labels, days = Plot.month_ticks()

        for i, (model_name, da) in enumerate(seasonal_cycles.items()):
            ax = axs_flat[i]

            ax.plot(da.values, label="model")
            ax.plot(obs_seasonal_cycle[value_col].values, color="k", label="ERA5")

            ax.set_title(model_name.replace("_", "-"), weight="medium", fontsize=16)
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels)
            if i % ncols == 0:
                ax.set_ylabel(legend_title)

            ax.grid(alpha=0.1)

            for d in range(365):
                if days[d].day == 1:
                    ax.axvline(d, color="k", alpha=0.05)
            
            ax.legend()

        for j in range(i + 1, len(axs.flatten())):
            axs.flatten()[j].set_axis_off()

        if subtitle:
            fig.suptitle(title, y=1.2, fontsize=20, weight='medium')

        if add_logos:
            plt.close(fig)
            fig, img_ax = Plot.add_image_below(fig=fig, image_path=LOGO_HORIZON_PATH, pad_frac=-.01)
            return fig, ax, img_ax
        else:
            img_ax = None
            return fig, ax, img_ax
    

    @staticmethod
    def plot_spatial_maps(obs:xr.Dataset,
                          spatial_maps:dict,
                          value_col:str,
                          legend_title:str|None=None,
                          ncols:int=4,
                          cmap:str|None=None,
                        #   borders:bool=True,
                        #   coastlines:bool=True,
                        #   gridlines:bool=True,
                          projection:cartopy.crs=ccrs.PlateCarree(),
                        #   dpi:int=100,
                          add_logos:bool=True
    ) -> tuple[plt.Figure, plt.Axes, plt.Axes | None]:
        '''
        Plots ERA5 (obs) in the top-left corner and CORDEX models in subsequent rows.

        Parameters:
            obs (xarray.Dataset, required):
                The xarray Dataset containing the observational data (ERA5).
            spatial_maps(dict, required):
                A dictionary where keys are model names and values are xarray DataArrays
                containing the spatial map data for each model.
            value_col (str, required):
                The variable name/column in the datasets to plot (e.g., 't2m', 'tp').
            legend_title (str | None, optional):
                The title for the overall figure legend. Defaults to None.
            ncols (int, optional):
                The number of columns in the subplot grid. Defaults to 4.
            cmap (str | matplotlib.colors.Colormap | None, optional):
                The desired colormap type (e.g., "t2m", "tp", "anomaly") or a
                standard Matplotlib colormap name. Defaults to None (inferred from `value_col`).
            projection (cartopy.crs, optional);
                The Cartopy coordinate reference system for the map. Defaults to ccrs.PlateCarree().
            add_logos (bool, optional);
                Whether to add a custom image/logo below the plot. Defaults to True.

        Returns:
            tuple[matplotlib.figure.Figure, matplotlib.axes.Axes, matplotlib.axes.Axes | None]:
                A tuple containing:
                - fig: The generated Matplotlib Figure.
                - ax: The Matplotlib Axes object with the spatial maps.
                - img_ax: The Axes object containing the added logo, or None if `add_logos` is False.
        '''

        cmap = cmap if cmap else value_col
        n_models = len(spatial_maps)
        total_slots = ncols + n_models

        fig, axs, axs_flat, nrows, ncols = Plot.create_grid(
            n_panels=total_slots, 
            projection=projection
        )

        # Calculate Global Limits 
        data_obs = obs.mean(dim="valid_time") if "valid_time" in obs.dims else obs
        vmin, vmax = data_obs.min().values, data_obs.max().values
        cmap, norm = Plot.get_colormap(cmap, vmin, vmax, value_col=value_col)

        # Observational data
        ax_obs = axs_flat[0]

        ax_obs.pcolormesh(
            data_obs.longitude, data_obs.latitude, data_obs,
            cmap=cmap, norm=norm, transform=ccrs.PlateCarree()
        )

        ax_obs.set_title("ERA5", fontsize=18, weight='medium')

        # Apply standard plot.py map features
        ax_obs.coastlines()
        ax_obs.add_feature(cartopy.feature.BORDERS, lw=1, alpha=0.7, ls="--")
        ax_obs.gridlines(draw_labels=False, linewidth=0.5, color='black', alpha=0.2)


        # Turn off the rest of the first row (empty space)
        for i in range(1, ncols):
            axs_flat[i].set_axis_off()

        # Cordex models
        start_index = ncols

        for i, (name, da_clim) in enumerate(spatial_maps.items()):
            ax_idx = start_index + i
            ax = axs_flat[ax_idx]

            # Plot
            ax.pcolormesh(
                da_clim.longitude, da_clim.latitude, da_clim,
                cmap=cmap, norm=norm, transform=ccrs.PlateCarree()
            )

            ax.set_title(name, fontsize=16, weight='medium')

            # Standard features
            ax.coastlines()
            ax.add_feature(cartopy.feature.BORDERS, lw=1, alpha=0.7, ls="--")
            ax.gridlines(draw_labels=False, linewidth=0.5, color='black', alpha=0.2)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm._A = []

        # Handle any normalization type to determine ticks
        if hasattr(norm, "boundaries") and norm.boundaries is not None:
            ticks = norm.boundaries
            tick_labels = [int(b) for b in ticks]

        else:
            if cmap.name == "anomaly_cmap" and value_col in ["tp"]:
                ticks = np.linspace(norm.vmin, norm.vmax, len(ANOMALY_COLORS))
                for t in [0, -0.5]:
                    if t not in ticks:
                        ticks = np.sort(np.append(ticks, t))
                tick_labels = [int(t) if abs(t) >= 1 else round(t, 1) for t in ticks]
            else:
                if vmin >= 0:
                    n_ticks = temperature_positive_cmap.N + 1
                elif vmax <= 0:
                    n_ticks = temperature_negative_cmap.N + 1
                else:
                    n_ticks = temperature_cmap.N + 1

                ticks = np.linspace(norm.vmin, norm.vmax, n_ticks)
                tick_labels = [round(t, 1) for t in ticks]


        cbar = fig.colorbar(
            sm, ax=axs_flat.tolist(), 
            orientation='horizontal', location="top",
            fraction=0.01, pad=.08, aspect=100, ticks=ticks)

        if cmap.name == "anomaly_cmap" and value_col in ["tp"]:
            cbar.ax.set_xlim(-0.5, None)

        cbar.set_label(legend_title, labelpad=10, fontsize=27, weight='bold', color='#364563')
        cbar.set_ticklabels(tick_labels)

        plt.setp(plt.getp(cbar.ax.axes, 'xticklabels'), family=FONT_ROBOTO_CONDENSED_REGULAR.get_name(), fontsize=13)


        # Hide unused axes in the bottom rows
        last_used_index = start_index + n_models
        for i in range(last_used_index, len(axs_flat)):
            axs_flat[i].set_axis_off()

        if add_logos:
            plt.close(fig)
            fig, img_ax = Plot.add_image_below(fig=fig, image_path=LOGO_HORIZON_PATH, pad_frac=-0.2)
            return fig, axs_flat, img_ax
        else:
            plt.tight_layout()
            return fig, axs_flat, None

    @staticmethod
    def month_ticks() -> tuple[list[int], list[str], pd.DatetimeIndex]:
        '''
        Generates tick positions and labels for months based on the 15th day of each month.

        Returns:
            tuple[list[int], list[str], pd.DatetimeIndex];
                A tuple containing:
                - ticks: A list of day-of-year integers representing the 15th of each month.
                - labels: A list of single-character month abbreviations corresponding to the ticks.
                - days: A Pandas DatetimeIndex covering the full year from Jan 1 to Dec 31.
        '''

        import pandas as pd
        days = pd.date_range(start="2020-01-01", end="2021-01-01")
        ticks = [i for i in range(365) if days[i].day == 15]
        labels = [days[i].strftime("%b")[0] for i in range(365) if days[i].day == 15]
        return ticks, labels, days
    
    @staticmethod
    def create_grid(n_panels, ncols=4, projection=None, sharex=False, sharey=False):
        '''
        Creates a grid of subplots based on the number of panels required.
        '''
        nrows = int(np.ceil(n_panels / ncols))
        if n_panels < ncols:
            ncols = n_panels

        fig, axs = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(20, 4 * nrows), # Slightly increased height per row for breathing room
            dpi=100,
            subplot_kw={"projection": projection} if projection else None,
            sharex=sharex,
            sharey=sharey,
        )

        axs = np.array(axs)
        axs_flat = axs.flatten()

        return fig, axs, axs_flat, nrows, ncols
    
    @staticmethod
    def plot_rolling_window_comparison(
        model_dfs: dict,
        obs_df: pd.DataFrame,
        value_col: str,
        time_col: str = "year",
        model_value_col: str = "value", # The column name in model_dfs (often standardized to 'value')
        legend_title: str = None,
        figsize: tuple = None,
        dpi: int = 100,
        add_logos: bool = True,
        subtitle: bool = True,
        yaxis_label: str = None,
    ):
        '''
        Plots a grid comparing rolling window statistics of models vs observations with Confidence Intervals.
        Reuses create_grid and adds logos.
        '''

        n_models = len(model_dfs)

        # Create Grid
        fig, axs, axs_flat, nrows, ncols = Plot.create_grid(
            n_panels=n_models,
            sharex=True,
            sharey=True
        )

        # Iterate and Plot
        for i, (model_name, model_df) in enumerate(model_dfs.items()):
            ax = axs_flat[i]


            # Plot OBSERVATIONS
            ax.plot(
                obs_df[time_col], obs_df[value_col], 
                color='black', 
                linewidth=1.5, 
                linestyle='--', 
                label='ERA 5 Observations'
            )

            # Plot CI 
            obs_lower = f"{value_col}_ci_lower"
            obs_upper = f"{value_col}_ci_upper"

            if obs_lower in obs_df.columns and obs_upper in obs_df.columns:
                ax.fill_between(
                    obs_df[time_col],
                    obs_df[obs_lower],
                    obs_df[obs_upper],
                    color='gray',
                    alpha=0.2
                )


            # Plot MODELS
            ax.plot(
                model_df[time_col], model_df[model_value_col], 
                color='darkblue', 
                linewidth=1.5, 
                linestyle='-', 
                label='Cordex Model'
            )

            # Plot CI (Model)
            mod_lower = f"{model_value_col}_ci_lower"
            mod_upper = f"{model_value_col}_ci_upper"

            if mod_lower in model_df.columns and mod_upper in model_df.columns:
                ax.fill_between(
                    model_df[time_col],
                    model_df[mod_lower],
                    model_df[mod_upper],
                    color='lightblue',
                    alpha=0.3
                )

            # Formatting
            ax.set_title(model_name.replace("_", " "), weight='bold', fontsize=14)

            if i % ncols == 0 and yaxis_label:
                ax.set_ylabel(yaxis_label)

            # Only set xlabel on bottom row
            if i >= (nrows - 1) * ncols:
                ax.set_xlabel(time_col.capitalize())

            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize='small')

        # Hide Unused Panels
        for j in range(i + 1, len(axs_flat)):
            axs_flat[j].set_axis_off()

        # Subtitle and Layout
        if subtitle and legend_title:
            fig.suptitle(
                legend_title, 
                fontsize=20, 
                weight='bold', 
                color="#364563",
                y=1.02
            )

        fig.tight_layout()

        # Logos and Return
        if add_logos:
            plt.close(fig)
            fig, img_ax = Plot.add_image_below(fig=fig, image_path=LOGO_HORIZON_PATH, pad_frac=-0.02)
            return fig, axs_flat, img_ax
        else:
            return fig, axs_flat, None
    
    
class KoppenGeiger:

    @staticmethod
    def load_kg_legend(path):
        '''
        Loads the Köppen–Geiger climate classification legend from a text file.

        The legend file is expected to have a specific format (e.g., '1: Af Tropical rainforest [102 0 0]'),
        which is parsed using a regular expression to extract the classification code, class abbreviation,
        description, and normalized RGB color values.

        Parameters:
            path (str): The file path to the Köppen–Geiger legend text file.

        Returns:
            pd.DataFrame: A DataFrame with columns 'code' (int), 'class' (str), 'description' (str),
                and 'rgb' (tuple of normalized floats).
        '''
        rows = []
        pattern = re.compile(r"^\s*(\d+):\s+(\w+)\s+(.*?)\s+\[(.*?)\]")

        with open(path, "r") as f:
            for line in f:
                match = pattern.match(line)
                if match:
                    code   = int(match.group(1))
                    klass  = match.group(2)
                    desc   = match.group(3).strip()
                    rgb    = tuple(int(v)/255 for v in match.group(4).split())
                    rows.append((code, klass, desc, rgb))

        return pd.DataFrame(rows, columns=["code", "class", "description", "rgb"])

    @staticmethod
    def draw_koppen_legend(fig, kg_legend, fontsize=8):
        '''
        Draws a custom, grouped Köppen-Geiger climate classification legend onto a Matplotlib figure.

        The legend is placed in an inset Axes at the bottom of the figure, organized into main
        climate categories (Tropical, Arid, Temperate, Cold, Polar) across four columns.



        Parameters:
            fig (matplotlib.figure.Figure): The Matplotlib figure object to add the legend to.
            kg_legend (pd.DataFrame): A DataFrame containing the Köppen-Geiger legend data,
                including 'rgb' color tuples and 'class' descriptions.
            fontsize (int, optional): The font size for the legend text. Defaults to 8.

        Returns:
            matplotlib.axes.Axes: The newly created Axes object containing the legend.
        '''
        
        KG_GROUPS = {
            "Tropical": [1,2,3],
            "Arid": [4,5,6,7],
            "Temperate": list(range(8,17)),
            "Cold": list(range(17,29)),
            "Polar": [29,30]
        }

        # Bottom inset axes for legend
        fig.subplots_adjust(bottom=0.25)
        ax = fig.add_axes([0.08, 0.04, 0.85, 0.3])
        ax.axis("off")
        ax.set_frame_on(False)

        grouped_rows = []
        for group, codes in KG_GROUPS.items():
            # Group header (rgb=None)
            grouped_rows.append((None, group))

            # Rows for each class in group
            subset = kg_legend[kg_legend["code"].isin(codes)]
            for _, r in subset.iterrows():
                text = f"{r['class']} — {r['description']}"
                grouped_rows.append((r["rgb"], text))
        
        # Split into 4 columns
        num_cols = 4
        rows_per_col = int(np.ceil(len(grouped_rows) / num_cols))

        for col in range(num_cols):
            col_data = grouped_rows[col * rows_per_col:(col + 1) * rows_per_col]
            if col == 1:
                x0 = 0.02 + col * 0.20
                y = 0.5
            else:
                x0 = 0.02 + col * 0.28
                y = 0.5

            for rgb, text in col_data:

                if rgb is None:  # Group header
                    ax.text(x0, y, text, fontweight="bold", fontsize=fontsize+5, va="top")
                    y -= 0.065
                    continue

                # Small color box
                ax.add_patch(mpatches.Rectangle(
                    (x0, y - 0.03), 0.02, 0.03,
                    color=rgb, ec="black", lw=0.3
                ))

                ax.text(x0 + 0.03, y, text, fontsize=fontsize, va="top")
                y -= 0.05

        return ax