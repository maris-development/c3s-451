from datetime import datetime
import xarray as xr
import geopandas as gpd

class CordexClient:
    def __init__(self, cordex_token: str):
        """
        Placeholder for CordexClient
        """
        self.cordex_token = cordex_token
        self.cordex_arco_base_url = "https://arco.datastores.ecmwf.int/cadl-arco-geo-014/arco/projections_cordex_domains_single_levels"

    def fetch_cordex_xr(
        self,
        variable: str,
        model_url: str,
        bbox: tuple[float, float, float, float],
        time_range: tuple[datetime, datetime],
    ) -> xr.Dataset:
        """
        Fetch CORDEX data as an xarray Dataset for a given variable, model, bounding box, and time range.

        Parameters:
            variable (str): The variable to fetch (e.g., 'tasmax').
            model_url (str): The model URL segment to access the specific dataset. (eg. eur11-hist-day-cccma_canesm2-clmcom_clm_cclm4_8_17-r1i1p1)
            bbox (tuple): A tuple defining the bounding box (min_lat, min_lon, max_lat, max_lon).
            time_range (tuple): A tuple defining the time range (start_time, end_time).
        """
        zarr_url = f"{self.cordex_arco_base_url}/{model_url}/geoChunked.zarr"
        headers = {
            "Authorization": f"Bearer {self.cordex_token}",
        }
        
        ds = xr.open_zarr(zarr_url, consolidated=True, storage_options={"headers": headers})
        variable_ds = ds[variable]
        bbox_filtered_ds = variable_ds.sel(
            longitude=slice(bbox[1], bbox[3]),
            latitude=slice(bbox[0], bbox[2]),
            time=slice(time_range[0], time_range[1]),
        )
        
        out_ds = bbox_filtered_ds.to_dataset()
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