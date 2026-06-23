from datetime import datetime
import xarray as xr
import geopandas as gpd

class CordexClient:
    def __init__(self, cordex_token: str):
        """
        Placeholder for CordexClient
        """
        self.cordex_token = cordex_token

    def fetch_cordex_xr(
        self,
        variable: str,
        model_url: str,
        bbox: tuple[float, float, float, float],
        time_range: tuple[datetime, datetime],
        temp_res: str = "daily"
    ) -> xr.Dataset:
        """
        Fetch CORDEX data as an xarray Dataset for a given variable, model, bounding box, and time range.

        Parameters:
            variable (str): The variable to fetch (e.g., 'tasmax').
            model_url (str): The model URL segment to access the specific dataset. (eg. eur11-hist-day-cccma_canesm2-clmcom_clm_cclm4_8_17-r1i1p1)
            bbox (tuple): A tuple defining the bounding box (min_lon, min_lat, max_lon, max_lat).
            time_range (tuple): A tuple defining the time range (start_time, end_time).
            temp_res (str): Temporal resolution (e.g. "daily", "monthly").
        """
        headers = {
            "Authorization": f"Bearer {self.cordex_token}",
        }
        
        ds = xr.open_zarr(model_url, consolidated=True, storage_options={"headers": headers})
        variable_ds = ds[variable]
        bbox_filtered_ds = variable_ds.sel(
            longitude=slice(bbox[0], bbox[2]),
            latitude=slice(bbox[1], bbox[3]),
            time=slice(time_range[0], time_range[1]),
        )
        
        out_ds = bbox_filtered_ds.to_dataset()
        # CORDEX Zarr stores are daily. Resample to monthly only when requested.
        if temp_res == "monthly":
            out_ds = self._resample_to_monthly(out_ds, variable)
        elif temp_res != "daily":
            raise ValueError(f"temp_res must be 'daily' or 'monthly', got '{temp_res}'")

        return out_ds
    
    def fetch_cordex_gpd(
        self,
        variable: str,
        model_url: str,
        bbox: tuple[float, float, float, float],
        time_range: tuple[datetime, datetime],
    ) -> gpd.GeoDataFrame:
        """
        Fetch CORDEX data as a GeoDataFrame for a given variable, model, GeoDataFrame, and time range.

        Parameters:
            variable (str): The variable to fetch (e.g., 'tasmax').
            model_url (str): The model URL segment to access the specific dataset. (eg. eur11-hist-day-cccma_canesm2-clmcom_clm_cclm4_8_17-r1i1p1)
            bbox (tuple): A tuple defining the bounding box (min_lat, min_lon, max_lat, max_lon).
            time_range (tuple): A tuple defining the time range (start_time, end_time).
        """
        ds = self.fetch_cordex_xr(variable, model_url, bbox, time_range)

        df = ds.to_dataframe().reset_index()
        # create geometry
        df['geometry'] = gpd.points_from_xy(df['longitude'], df['latitude'])
        gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')

        return gdf
    
    def list_available_models(self) -> list[str]:
        """
        Placeholder method to list available CORDEX models.
        In a real implementation, this would query the CORDEX data store.
        """
        # This is a placeholder implementation.
        return [
            "eur11-hist-day-cccma_canesm2-clmcom_clm_cclm4_8_17-r1i1p1",
            # Add more models as needed
        ]
    
    @staticmethod
    def _resample_to_monthly(ds: xr.Dataset, variable: str) -> xr.Dataset:
        """
        Aggregate a daily CORDEX dataset to monthly, choosing the operator per variable
        so the result matches the CMIP6 native-monthly product.
        """
        # "MS" = month-start timestamps, consistent with CMIP6 monthly outputs
        resampler = ds.resample(time="MS")

        if variable in ("tas", "tasmin", "tasmax"):
            # CMIP6 monthly tas/tasmax/tasmin = time-mean of the daily values
            return resampler.mean()
        elif variable == "pr":
            # pr is a flux (kg m-2 s-1); monthly MEAN keeps the rate convention
            # (downstream multiplies by 86400 -> mm/day). Use .sum() only if you
            # switch the whole pipeline to monthly accumulations.
            return resampler.mean()
        else:
            return resampler.mean()