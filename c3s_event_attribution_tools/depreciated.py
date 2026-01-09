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
import numpy as np
from shapely import contains_xy
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from datetime import datetime, timedelta
from typing import Union, Literal




# Plot
#===============================================================================================================================================================

# add standard deviation
def depreciated_n_day_accumulations_gdf(data, value_col, parameter, event_date, labelticks, labels, centering=False, datetime_col="valid_time", days=None, ylimit=None):

    fig, axs = plt.subplots(ncols=4, figsize=(20, 3), dpi=100, sharey=True)

    # Ensure datetime and sorted
    data = data.copy()
    data[datetime_col] = pd.to_datetime(data[datetime_col])
    data = data.sort_values(datetime_col)

    for i in range(4):
        ax = axs[i]

        # Determine n-day window
        if days is not None:
            ndays = days[i]
        elif value_col == 't2m':
            ndays = [1, 3, 7, 14][i]
        elif value_col == 'tp':
            ndays = [1, 3, 5, 10][i]
        else:
            ndays = [1, 3, 5, 11][i]

        if value_col == "tp":
            data_nday = (
                data.set_index(datetime_col)
                    [value_col]
                    .rolling(ndays, min_periods=1, center=centering)
                    .sum()
                    .reset_index()
            )
        else:
            data_nday = (
                data.set_index(datetime_col)
                    [value_col]
                    .rolling(ndays, min_periods=1, center=centering)
                    .mean()
                    .reset_index()
            )


        # Plot each year in blue
        for y in data_nday[datetime_col].dt.year.unique():
            data_y = data_nday[data_nday[datetime_col].dt.year == y]
            ax.plot(
                data_y[datetime_col].dt.dayofyear,
                data_y[value_col],
                color="tab:blue",
                alpha=0.3
            )

        # Style the plot
        ax.set_xticks(labelticks)
        ax.set_xticklabels(labels)
        ax.grid(axis="x", color="k", alpha=0.2)
        ax.set_title(f"{ndays}-day accumulated {parameter}")

        # Highlight date window
        ylim = ax.get_ylim()
        print(ylim)

        dayofyear = pd.to_datetime(event_date).dayofyear
        ax.add_patch(Rectangle((dayofyear, ylim[0]), -15, 10000,
                               color="gold", alpha=0.3))
        ax.set_ylim(ylim)

        # Highlight selected year
        year2 = pd.to_datetime(event_date)
        data_y = data_nday[data_nday[datetime_col] <= year2]
        ax.plot(data_y[datetime_col].dt.dayofyear, data_y[value_col], color="k")

    if ylimit is not None:
        ax.set_ylim(0, ylimit)

    return fig, axs


def depreciated_plot_n_day_accumulations(rolled_data_list, value_col, parameter, event_date, labelticks, labels, days, ylimit=None, datetime_col="valid_time"):
    """
    Plot n-day rolling accumulations for different windows.
    
    Parameters
    ----------
    rolled_data_list : list of DataFrames
        List of results from n_day_accumulations_gdf(), one per window.
    value_col : str
        Column that was rolled.
    parameter : str
        Parameter name for titles.
    event_date : str or datetime
        Highlight date.
    labelticks : list
        X-axis tick positions.
    labels : list
        X-axis tick labels.
    days : list
        List of window sizes.
    ylimit : int or None
        Upper limit for y-axis.
    datetime_col : str
        Column with datetimes.
    """
    fig, axs = plt.subplots(ncols=len(rolled_data_list), figsize=(5 * len(rolled_data_list), 3), dpi=100, sharey=True)

    if len(rolled_data_list) == 1:
        axs = [axs]  # make iterable if only one axis

    for ax, data_nday, ndays in zip(axs, rolled_data_list, days):
        # Plot each year in blue
        for y in data_nday[datetime_col].dt.year.unique():
            data_y = data_nday[data_nday[datetime_col].dt.year == y]
            ax.plot(
                data_y[datetime_col].dt.dayofyear,
                data_y[value_col],
                color="tab:blue",
                alpha=0.3
            )

        # Style the plot
        ax.set_xticks(labelticks)
        ax.set_xticklabels(labels)
        ax.grid(axis="x", color="k", alpha=0.2)
        ax.set_title(f"{ndays}-day accumulated {parameter}")

        # Highlight date window
        ylim = ax.get_ylim()
        dayofyear = pd.to_datetime(event_date).dayofyear
        ax.add_patch(Rectangle((dayofyear, ylim[0]), -15, 10000, color="gold", alpha=0.3))
        ax.set_ylim(ylim)

        # Highlight selected year (all up to event_date)
        year2 = pd.to_datetime(event_date)
        data_y = data_nday[data_nday[datetime_col] <= year2]
        ax.plot(data_y[datetime_col].dt.dayofyear, data_y[value_col], color="k")

        if ylimit is not None:
            ax.set_ylim(0, ylimit)

    return fig, axs



def depreciated_subplot_gdf(gdfs:gpd.GeoDataFrame, value_col:str, datetime_col:str='valid_time',
                polygons:list[Polygon]=None, ncols:int=5, figsize:tuple[int, int]=(20, 12),
                cmap:str='coolwarm', legend_title:str='Temperature (°C)', borders:bool=True,
                coastlines:bool=True, gridlines:bool=True, subtitle:str=None,
                projection:cartopy.crs=ccrs.PlateCarree(), extends:tuple[float, float, float, float]=None,
                dpi:int=100, flatten_empty_plots:bool=True
                ):
    
    # Ensure datetime column is datetime type
    gdfs[datetime_col] = pd.to_datetime(gdfs[datetime_col])

    # Unique days sorted
    unique_days = sorted(gdfs[datetime_col].dt.date.unique())
    n_plots = len(unique_days)
    nrows = math.ceil(n_plots / ncols)

    # Create subplots with Cartopy projection
    fig, axes = plt.subplots(
        nrows, ncols, figsize=figsize, dpi=dpi,
        subplot_kw={'projection': projection}
    )
    axes = axes.flatten()

    # Normalize color scale across all data
    vmin = math.floor(gdfs[value_col].min())
    vmax = math.ceil(gdfs[value_col].max())

    for i, day in enumerate(unique_days):
        ax = axes[i]

        # Filter GeoDataFrame for this day
        day_gdf = gdfs[gdfs[datetime_col].dt.date == day]

        # Plot data on this subplot
        day_gdf.plot(
            ax=ax,
            column=value_col,
            cmap=cmap,
            legend=False,  # legend handled once globally
            vmin=vmin,
            vmax=vmax,
            marker='s'
        )

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

        # Draw polygons if provided
        if polygons is not None:
            for poly in polygons:
                x, y = poly.exterior.xy
                ax.plot(x, y, color='red', linewidth=2, transform=projection)

        ax.set_title(f"{day}", fontsize=10)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(not flatten_empty_plots)

    # Add shared colorbar to the top
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []
    cbar = fig.colorbar(sm, ax=axes.tolist(), orientation='horizontal', location="top", fraction=0.01, pad=.07, aspect=40)
    cbar.set_label(legend_title, labelpad=10, fontsize=12)

    if subtitle:
        fig.suptitle(subtitle, fontsize=16)
    
    # Set extent if provided
    if extends is not None:
      ax.set_extent(extends, crs=projection)

    #plt.tight_layout(rect=[0, 0, 0.95, 0.95])  # leave room for suptitle and colorbar
    return fig, axes


# plots multiple subplots of a GeoDataFrame in a single figure
def depreciated_subplot_gdf2(gdfs:gpd.GeoDataFrame, value_col:str, datetime_col:str='valid_time',
                polygons:list[Polygon]=None, ncols:int=5, figsize:tuple[int, int]=(20, 12),
                cmap:str='coolwarm', legend_title:str='Temperature (°C)', borders:bool=True,
                coastlines:bool=True, gridlines:bool=True, subtitle:str=None,
                projection:cartopy.crs=ccrs.PlateCarree(), extends:tuple[float, float, float, float]=None,
                dpi:int=100, flatten_empty_plots:bool=True, marker:str='o'
                ):

    # Ensure datetime column is datetime type
    gdfs[datetime_col] = pd.to_datetime(gdfs[datetime_col])

    # Unique days sorted
    unique_days = sorted(gdfs[datetime_col].dt.date.unique())
    n_plots = len(unique_days)
    nrows = math.ceil(n_plots / ncols)

    # Create subplots with Cartopy projection
    fig, axes = plt.subplots(
        nrows, ncols, figsize=figsize, dpi=dpi,
        subplot_kw={'projection': projection},
    )
    axes = axes.flatten()

    # Normalize color scale across all data
    vmin = math.floor(gdfs[value_col].min())
    vmax = math.ceil(gdfs[value_col].max())

    for i, day in enumerate(unique_days):
        ax = axes[i]

        # Filter GeoDataFrame for this day
        day_gdf = gdfs[gdfs[datetime_col].dt.date == day]

        # Plot data on this subplot
        day_gdf.plot(
            ax=ax,
            column=value_col,
            cmap=cmap,
            legend=False,  # legend handled once globally
            vmin=vmin,
            vmax=vmax,
            marker=marker
        )

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

        # Draw polygons if provided
        if polygons is not None:
            for poly in polygons:
                x, y = poly.exterior.xy
                ax.plot(x, y, color='red', linewidth=2, transform=projection)

        ax.set_title(f"{day}", fontsize=18, color='darkblue', weight='medium')

            # Set extent if provided
        if extends is not None:
            ax.set_extent(extends, crs=projection)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(not flatten_empty_plots)

    # Add shared colorbar to the top
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []
    cbar = fig.colorbar(sm, ax=axes.tolist(), orientation='horizontal', location="top", fraction=0.01, pad=.07, aspect=40)
    cbar.set_label(legend_title, labelpad=10, fontsize=27, weight='bold', color='darkblue')
    # set colorbar ticklabels
    cbar.ax.xaxis.set_tick_params(color='darkgrey') # dont work
    plt.setp(plt.getp(cbar.ax.axes, 'xticklabels'), color='darkgrey') # dont work

    if subtitle:
        fig.suptitle(subtitle)

    #plt.tight_layout(rect=[0, 0, 0.95, 0.95])  # leave room for suptitle and colorbar
    return fig, axes




# Process
#===============================================================================================================================================================

def depreciated_calculate_anomaly(final_gdf:gpd.GeoDataFrame, event_gdf:gpd.GeoDataFrame, value_col:str, datetime_col:str='valid_time'):
    # Ensure datetime columns
    final_gdf = final_gdf.copy()
    event_gdf = event_gdf.copy()
    final_gdf[datetime_col] = pd.to_datetime(final_gdf[datetime_col])
    event_gdf[datetime_col] = pd.to_datetime(event_gdf[datetime_col])
    
    # Prepare list for adjusted results
    adjusted_list = []
    
    for target_date in event_gdf[datetime_col].unique():
        # --- Step 1: get mean around date for all years ---
        results = []
        for year in final_gdf[datetime_col].dt.year.unique():
            center_date = datetime(year, target_date.month, target_date.day)
            start_window = center_date - timedelta(days=15)
            end_window = center_date + timedelta(days=15)
            
            subset = final_gdf[
                (final_gdf[datetime_col] >= start_window) &
                (final_gdf[datetime_col] <= end_window)
            ]
            results.append(subset)
        
        combined = pd.concat(results, ignore_index=True)
        mean_gdf = combined.groupby("geometry")[value_col].mean().reset_index()
        
        # --- Step 2: subtract from gdfs for this target date ---
        current_day_gdf = event_gdf[event_gdf[datetime_col] == target_date]
        
        # Merge on geometry so subtraction matches locations
        merged = current_day_gdf.merge(mean_gdf, on="geometry", suffixes=("", "_mean"))

        if value_col == "t2m":
            merged[value_col] = merged[value_col] - merged[f"{value_col}_mean"]
        elif value_col == "tp":
            merged[value_col] = merged[value_col] / merged[f"{value_col}_mean"]
        else:
            raise ValueError("value_col must be 't2m' or 'tp'")
    
        merged.drop(columns=[f"{value_col}_mean"], inplace=True)
        
        adjusted_list.append(merged)
    
    # Combine adjusted daily frames back together
    final_adjusted_gdf = gpd.GeoDataFrame(
        pd.concat(adjusted_list, ignore_index=True),
        crs=event_gdf.crs
    )
    
    return final_adjusted_gdf



def depreciated_n_day_accumulations_gdf1(gdf, value_col:str, padding:int, centering:bool=False, datetime_col:str="valid_time"):
    """
    Compute rolling n-day accumulation (sum for 'tp', mean otherwise).
    
    Parameters
    ----------
    data : GeoDataFrame or DataFrame
        Input data.
    value_col : str
        Column to roll (e.g. 't2m', 'tp').
    days : int
        Window size in days.
    centering : bool
        Center the window.
    datetime_col : str
        Column with datetimes.

    Returns
    -------
    DataFrame
        Rolled values with same columns.
    """
    gdf = gdf.copy()
    gdf[datetime_col] = pd.to_datetime(gdf[datetime_col])
    #data = data.sort_values(datetime_col)

    if value_col == "tp":
        data_nday = (
            gdf.set_index(datetime_col)[value_col]
            .rolling(padding, min_periods=1, center=centering)
            .sum()
            .reset_index()
        )
    else:
        data_nday = (
            gdf.set_index(datetime_col)[value_col]
            .rolling(padding, min_periods=1, center=centering)
            .mean()
            .reset_index()
        )

    return data_nday

def depreciated_n_day_accumulations_gdf2(
    gdf:Union[pd.DataFrame, gpd.GeoDataFrame], value_col:str, padding:int,
    centering:bool=False, datetime_col:str="valid_time",
    method:Literal["sum", "mean", "std", "quantile"]="mean", quantile:float=0.9
):
    """
    Compute rolling n-day statistics (sum, mean, std, quantile).
    
    Parameters
    ----------
    gdf : GeoDataFrame or DataFrame
        Input data.
    value_col : str
        Column to roll (e.g. 't2m', 'tp').
    padding : int
        Window size in days.
    centering : bool
        Center the window.
    datetime_col : str
        Column with datetimes.
    method : {"sum", "mean", "std", "quantile"}
        Rolling aggregation method.
    q : float, optional
        Quantile to compute if method="quantile". Default is 0.5 (median).

    Returns
    -------
    DataFrame
        Rolled values with same columns.
    """

    gdf = gdf.copy()
    gdf[datetime_col] = pd.to_datetime(gdf[datetime_col])
    
    roller = gdf.set_index(datetime_col)[value_col].rolling(
        padding, min_periods=1, center=centering
    )

    if method == "sum":
        result = roller.sum()
    elif method == "mean":
        result = roller.mean()
    elif method == "std":
        result = roller.std()
    elif method == "quantile":
        result = roller.quantile(quantile)
    else:
        raise ValueError(f"Unsupported method: {method}")

    return result.reset_index()

# Util
#===============================================================================================================================================================






# Data
#===============================================================================================================================================================





