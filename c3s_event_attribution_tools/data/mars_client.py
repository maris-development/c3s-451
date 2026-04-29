from ecmwfapi import ECMWFService
from datetime import datetime, timedelta
import shutil, subprocess
import xarray as xr
import iris  # type: ignore
import tempfile

from .variable import MarsVariable
from ..utils import Utils

class MarsClient:
    def __init__(self, key: str, email: str):
        """
        Initialize the MarsClient with credentials for the ECMWF MARS archive.

        This constructor sets up the ECMWF service for MARS (Meteorological Archival
        and Retrieval System) and performs a system check to ensure the CDO
        (Climate Data Operators) tool is available in the environment.

        Parameters:
            key (str):
                The API key required to authenticate with the ECMWF API.
        """
        self.key = key
        self.email = email
        self.server = ECMWFService("mars", key=key, url="https://api.ecmwf.int/v1", email=email)
        self.check_cdo()

    def check_cdo(self):
        """
        Check if the CDO (Climate Data Operators) utility is installed and functional.

        This method verifies the presence of the 'cdo' executable in the system
        PATH and attempts to retrieve its version information.

        Raises:
            EnvironmentError: If the 'cdo' command is not found or fails to
                execute correctly.
        """
        if shutil.which("cdo") is None:
            raise EnvironmentError(
                "❌ CDO not found. Please install it (e.g. `sudo apt install cdo`)."
            )

        try:
            v = subprocess.run(
                ["cdo", "-V"], capture_output=True, text=True, check=True
            )
            Utils.print(f"✅ {v.stdout.splitlines()[0]}")
        except Exception as e:
            raise EnvironmentError(f"⚠️ CDO check failed: {e}")

    def get_date_list(self, min_date: datetime, max_date: datetime) -> list[str]:
        """
        Generate a list of formatted date strings within a specified range.

        This method produces a list of dates in "YYYY-MM-DD" format. It includes
        logic to cap the range at two days before the current UTC date. This
        ensures the generated list does not include dates for which operational
        data may still be incomplete.

        Parameters:
            min_date (datetime):
                The starting date of the range.
            max_date (datetime):
                The requested end date of the range.

        Returns:
            list[str]: A list of date strings, capped at the most recent
            available daily data (UTC date minus two days).
        """
        # Use UTC date minus two days as the latest available date.
        given_date = datetime.utcnow() - timedelta(days=2)

        # Ensure given_date is within min_date and max_date
        if max_date > given_date:
            max_date = given_date

        # Generate list of days before the current date
        date_list = []
        while min_date <= max_date:
            date_list.append(min_date.strftime("%Y-%m-%d"))
            min_date += timedelta(days=1)

        return date_list

    def get_temp_path(self) -> str:
        """
        Create a temporary file and return its absolute file path.

        This helper method initializes a named temporary file with a '.nc'
        extension. The file is created with 'delete=False' to ensure it
        persists on disk for subsequent processing by external tools or
        libraries.

        Returns:
            str: The absolute path to the newly created temporary NetCDF file.
        """
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".nc")
        return temp_file.name

    def _normalize_dataset_and_filter_bbox(
        self,
        ds: xr.Dataset,
        min_lon: float,
        max_lon: float,
        min_lat: float,
        max_lat: float,
    ) -> xr.Dataset:
        """
        Normalize time and longitudes, then filter by bounding box.

        Parameters:
            ds (xr.Dataset): Dataset containing longitude and latitude coordinates.
            min_lon (float): Minimum longitude of the bounding box.
            max_lon (float): Maximum longitude of the bounding box.
            min_lat (float): Minimum latitude of the bounding box.
            max_lat (float): Maximum latitude of the bounding box.

        Returns:
            xr.Dataset: Time/longitude-normalized and spatially filtered dataset.
        """
        ds = self._normalize_time(ds)

        if "longitude" in ds.coords:
            ds = ds.assign_coords(longitude=((ds.longitude + 180) % 360) - 180)

        return ds.where(
            (ds["longitude"] >= min_lon)
            & (ds["longitude"] <= max_lon)
            & (ds["latitude"] >= min_lat)
            & (ds["latitude"] <= max_lat),
            drop=True
        )

    def _normalize_time(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Normalize datetime coordinates to midnight (00:00:00).

        This ensures daily data always uses day-boundary timestamps rather
        than midday timestamps (e.g. 12:00:00).

        Parameters:
            ds (xr.Dataset): Dataset potentially containing time coordinates.

        Returns:
            xr.Dataset: Dataset with normalized `valid_time`/`time` coordinates.
        """
        for coord_name in ("valid_time", "time"):
            if coord_name in ds.coords:
                ds = ds.assign_coords({coord_name: ds[coord_name].dt.floor("D")})

        return ds
    
    def fetch_forecast_data(
        self,
        variable: MarsVariable,
        min_date: datetime,
        max_date: datetime,
        min_lon: float,
        max_lon: float,
        min_lat: float,
        max_lat: float,
    ) -> xr.Dataset:
        """
        Fetch forecast MARS data for a specific variable.

        Parameters:
            variable (MarsVariable): Variable selector used to route to the
                corresponding forecast data retrieval method.
            min_date (datetime): Start date of the request window.
            max_date (datetime): End date of the request window.
            min_lon (float): Minimum longitude of the bounding box.
            max_lon (float): Maximum longitude of the bounding box.
            min_lat (float): Minimum latitude of the bounding box.
            max_lat (float): Maximum latitude of the bounding box.

        Returns:
            xr.Dataset: Requested forecast dataset.

        Raises:
            NotImplementedError: If the variable is known but not yet supported.
            ValueError: If an unknown variable is provided.
        """
        if variable == MarsVariable.t2m:
            return self.fetch_t2m_mean_forecast_data(
                min_date, max_date, min_lon, max_lon, min_lat, max_lat
            )

        if variable == MarsVariable.t2m_min:
            return self.fetch_t2m_min_forecast_data(
                min_date, max_date, min_lon, max_lon, min_lat, max_lat
            )

        if variable == MarsVariable.t2m_max:
            return self.fetch_t2m_max_forecast_data(
                min_date, max_date, min_lon, max_lon, min_lat, max_lat
            )

        if variable == MarsVariable.tp:
            return self.fetch_total_precipitation_forecast_data(
                min_date, max_date, min_lon, max_lon, min_lat, max_lat
            )

        if variable == MarsVariable.mslp:
            return self.fetch_mslp_forecast_data(
                min_date, max_date, min_lon, max_lon, min_lat, max_lat
            )

        if variable == MarsVariable.z500:
            return self.fetch_z500_forecast_data(
                min_date, max_date, min_lon, max_lon, min_lat, max_lat
            )

        raise ValueError(f"Unsupported MarsVariable: {variable}")

    
    def fetch_t2m_mean_forecast_data(
        self,
        min_date: datetime,
        max_date: datetime,
        min_lon: float,
        max_lon: float,
        min_lat: float,
        max_lat: float
    ) -> xr.Dataset:
        # Fetch the current date -7 days as a list of dates
        current_date = datetime.utcnow() - timedelta(days=1) # Use yesterday's date to compensate for forecast delay
        date_str = current_date.strftime("%Y-%m-%d")
        print(f"Fetching t2m forecast data from MARS for date: {date_str}")
        request = {
            "class": "od",
            "date": date_str,
            "step": "6/12/18/24/30/36/42/48/54/60/66/72/78/84/90/96/102/108/114/120/126/132/138/144/150/156/162/168/174/180/186",
            "expver": "1",
            "param": self.find_param_code("t2m"), # 2m temperature
            "grid": "0.25/0.25", # 0.25 degree grid
            "time": "00:00:00",
            "class": "od",
            "levtype": "sfc",
            "stream": "oper",
            "type": "fc",
            "target": "output",
            "format": "netcdf",
        }
        temp_path = self.get_temp_path()
        self.server.execute(request,target=temp_path)
        
        temp_path = self.get_temp_path()
        self.server.execute(request, target=temp_path)

        # USE CDO to compute daily mean
        out_daily = temp_path.replace(".nc", "_daily.nc")
        subprocess.run(
            [
                "cdo",
                "-O",
                "-r",
                "-f",
                "nc4",
                "-s",
                f"-daymean",
                f"-shifttime,3hour",
                temp_path,
                out_daily,
            ],
            check=True,
        )

        ds = xr.open_dataset(out_daily, chunks=-1)
        ds = ds.rename({"time": "valid_time"})
        return self._normalize_dataset_and_filter_bbox(
            ds, min_lon, max_lon, min_lat, max_lat
        )
        
    def fetch_t2m_max_forecast_data(
        self,
        min_date: datetime,
        max_date: datetime,
        min_lon: float,
        max_lon: float,
        min_lat: float,
        max_lat: float
    ) -> xr.Dataset:
        # Fetch the current date -7 days as a list of dates
        current_date = datetime.utcnow() - timedelta(days=1) # Use yesterday's date to compensate for forecast delay
        date_str = current_date.strftime("%Y-%m-%d")
        print(f"Fetching tmax forecast data from MARS for date: {date_str}")
        request = {
            "class": "od",
            "date": date_str,
            "step": "6/12/18/24/30/36/42/48/54/60/66/72/78/84/90/96/102/108/114/120/126/132/138/144/150/156/162/168/174/180/186",
            "expver": "1",
            "param": self.find_param_code("tmax"), 
            "grid": "0.25/0.25", # 0.25 degree grid
            "time": "00:00:00",
            "class": "od",
            "levtype": "sfc",
            "stream": "oper",
            "type": "fc",
            "target": "output",
            "format": "netcdf",
        }
        temp_path = self.get_temp_path()
        self.server.execute(request,target=temp_path)
        
        temp_path = self.get_temp_path()
        self.server.execute(request, target=temp_path)

        # USE CDO to compute daily mean
        out_daily = temp_path.replace(".nc", "_daily.nc")
        subprocess.run(
            [
                "cdo",
                "-O",
                "-r",
                "-f",
                "nc4",
                "-s",
                f"-daymean",
                f"-shifttime,3hour",
                temp_path,
                out_daily,
            ],
            check=True,
        )

        ds = xr.open_dataset(out_daily, chunks=-1)
        ds = ds.rename({"time": "valid_time"})
        return self._normalize_dataset_and_filter_bbox(
            ds, min_lon, max_lon, min_lat, max_lat
        )
        
    def fetch_t2m_min_forecast_data(
        self,
        min_date: datetime,
        max_date: datetime,
        min_lon: float,
        max_lon: float,
        min_lat: float,
        max_lat: float
    ) -> xr.Dataset:
        # Fetch the current date -7 days as a list of dates
        current_date = datetime.utcnow() - timedelta(days=1) # Use yesterday's date to compensate for forecast delay
        date_str = current_date.strftime("%Y-%m-%d")
        print(f"Fetching tmin forecast data from MARS for date: {date_str}")
        request = {
            "class": "od",
            "date": date_str,
            "step": "6/12/18/24/30/36/42/48/54/60/66/72/78/84/90/96/102/108/114/120/126/132/138/144/150/156/162/168/174/180/186",
            "expver": "1",
            "param": self.find_param_code("tmin"), 
            "grid": "0.25/0.25", # 0.25 degree grid
            "time": "00:00:00",
            "class": "od",
            "levtype": "sfc",
            "stream": "oper",
            "type": "fc",
            "target": "output",
            "format": "netcdf",
        }
        temp_path = self.get_temp_path()
        self.server.execute(request,target=temp_path)
        
        temp_path = self.get_temp_path()
        self.server.execute(request, target=temp_path)

        # USE CDO to compute daily mean
        out_daily = temp_path.replace(".nc", "_daily.nc")
        subprocess.run(
            [
                "cdo",
                "-O",
                "-r",
                "-f",
                "nc4",
                "-s",
                f"-daymean",
                f"-shifttime,3hour",
                temp_path,
                out_daily,
            ],
            check=True,
        )

        ds = xr.open_dataset(out_daily, chunks=-1)
        ds = ds.rename({"time": "valid_time"})
        return self._normalize_dataset_and_filter_bbox(
            ds, min_lon, max_lon, min_lat, max_lat
        )
        
    def fetch_mslp_forecast_data(
        self,
        min_date: datetime,
        max_date: datetime,
        min_lon: float,
        max_lon: float,
        min_lat: float,
        max_lat: float
    ) -> xr.Dataset:
        # Fetch the current date -7 days as a list of dates
        current_date = datetime.utcnow() - timedelta(days=1) # Use yesterday's date to compensate for forecast delay
        date_str = current_date.strftime("%Y-%m-%d")
        print(f"Fetching mslp forecast data from MARS for date: {date_str}")
        request = {
            "class": "od",
            "date": date_str,
            "step": "6/12/18/24/30/36/42/48/54/60/66/72/78/84/90/96/102/108/114/120/126/132/138/144/150/156/162/168/174/180/186",
            "expver": "1",
            "param": self.find_param_code("mslp"),
            "grid": "0.25/0.25", # 0.25 degree grid
            "time": "00:00:00",
            "class": "od",
            "levtype": "sfc",
            "stream": "oper",
            "type": "fc",
            "target": "output",
            "format": "netcdf",
        }
        temp_path = self.get_temp_path()
        self.server.execute(request,target=temp_path)
        
        temp_path = self.get_temp_path()
        self.server.execute(request, target=temp_path)

        # USE CDO to compute daily mean
        out_daily = temp_path.replace(".nc", "_daily.nc")
        subprocess.run(
            [
                "cdo",
                "-O",
                "-r",
                "-f",
                "nc4",
                "-s",
                f"-daymean",
                f"-shifttime,3hour",
                temp_path,
                out_daily,
            ],
            check=True,
        )

        ds = xr.open_dataset(out_daily, chunks=-1)
        ds = ds.rename({"time": "valid_time"})
        return self._normalize_dataset_and_filter_bbox(
            ds, min_lon, max_lon, min_lat, max_lat
        )
        
    def fetch_total_precipitation_forecast_data(
        self,
        min_date: datetime,
        max_date: datetime,
        min_lon: float,
        max_lon: float,
        min_lat: float,
        max_lat: float
    ) -> xr.Dataset:
        # Fetch the current date -7 days as a list of dates
        current_date = datetime.utcnow() - timedelta(days=1) # Use yesterday's date to compensate for forecast delay
        date_str = current_date.strftime("%Y-%m-%d")
        print(f"Fetching total precipitation forecast data from MARS for date: {date_str}")
        request = {
            "class": "od",
            "date": date_str,
            "step": "6/12/18/24/30/36/42/48/54/60/66/72/78/84/90/96/102/108/114/120/126/132/138/144/150/156/162/168/174/180/186",
            "expver": "1",
            "param": self.find_param_code("tp"),
            "grid": "0.25/0.25", # 0.25 degree grid
            "time": "00:00:00",
            "class": "od",
            "levtype": "sfc",
            "stream": "oper",
            "type": "fc",
            "target": "output",
            "format": "netcdf",
        }
        temp_path = self.get_temp_path()
        self.server.execute(request,target=temp_path)
        
        temp_path = self.get_temp_path()
        self.server.execute(request, target=temp_path)

        # USE CDO to compute daily mean
        out_daily = temp_path.replace(".nc", "_daily.nc")
        subprocess.run(
            [
                "cdo",
                "-O",
                "-r",
                "-f",
                "nc4",
                "-s",
                f"-daymean",
                f"-shifttime,3hour",
                temp_path,
                out_daily,
            ],
            check=True,
        )

        ds = xr.open_dataset(out_daily, chunks=-1)
        ds = ds.rename({"time": "valid_time"})
        return self._normalize_dataset_and_filter_bbox(
            ds, min_lon, max_lon, min_lat, max_lat
        )
        
    def fetch_z500_forecast_data(
        self,
        min_date: datetime,
        max_date: datetime,
        min_lon: float,
        max_lon: float,
        min_lat: float,
        max_lat: float
    ) -> xr.Dataset:
        # Fetch the current date -7 days as a list of dates
        current_date = datetime.utcnow() - timedelta(days=1) # Use yesterday's date to compensate for forecast delay
        date_str = current_date.strftime("%Y-%m-%d")
        print(f"Fetching z500 forecast data from MARS for date: {date_str}")
        request = {
            "class": "od",
            "date": date_str,
            "step": "6/12/18/24/30/36/42/48/54/60/66/72/78/84/90/96/102/108/114/120/126/132/138/144/150/156/162/168/174/180/186",
            "expver": "1",
            "param": self.find_param_code("z500"),
            "grid": "0.25/0.25", # 0.25 degree grid
            "time": "00:00:00",
            "class": "od",
            "levtype": "pl",
            "levelist": "500",
            "stream": "oper",
            "type": "fc",
            "target": "output",
            "format": "netcdf",
        }
        
        temp_path = self.get_temp_path()
        self.server.execute(request,target=temp_path)
        
        temp_path = self.get_temp_path()
        self.server.execute(request, target=temp_path)

        # USE CDO to compute daily mean
        out_daily = temp_path.replace(".nc", "_daily.nc")
        subprocess.run(
            [
                "cdo",
                "-O",
                "-r",
                "-f",
                "nc4",
                "-s",
                f"-daymean",
                f"-shifttime,3hour",
                temp_path,
                out_daily,
            ],
            check=True,
        )

        ds = xr.open_dataset(out_daily, chunks=-1)
        ds = ds.rename({"time": "valid_time"})
        return self._normalize_dataset_and_filter_bbox(
            ds, min_lon, max_lon, min_lat, max_lat
        )
        
    def fetch_operational_data(
        self,
        variable: MarsVariable,
        min_date: datetime,
        max_date: datetime,
        min_lon: float,
        max_lon: float,
        min_lat: float,
        max_lat: float,
    ) -> xr.Dataset:
        """
        Fetch operational MARS data for a specific variable.

        Parameters:
            variable (MarsVariable): Variable selector used to route to the
                corresponding operational data retrieval method.
            min_date (datetime): Start date of the request window.
            max_date (datetime): End date of the request window.
            min_lon (float): Minimum longitude of the bounding box.
            max_lon (float): Maximum longitude of the bounding box.
            min_lat (float): Minimum latitude of the bounding box.
            max_lat (float): Maximum latitude of the bounding box.

        Returns:
            xr.Dataset: Requested operational dataset.

        Raises:
            NotImplementedError: If the variable is known but not yet supported.
            ValueError: If an unknown variable is provided.
        """
        if variable == MarsVariable.t2m:
            return self.fetch_t2m_mean_operational_data(
                min_date, max_date, min_lon, max_lon, min_lat, max_lat
            )

        if variable == MarsVariable.t2m_min:
            return self.fetch_t2m_min_operational_data(
                min_date, max_date, min_lon, max_lon, min_lat, max_lat
            )

        if variable == MarsVariable.t2m_max:
            return self.fetch_t2m_max_operational_data(
                min_date, max_date, min_lon, max_lon, min_lat, max_lat
            )

        if variable == MarsVariable.tp:
            return self.fetch_total_precipitation_operational_data(
                min_date, max_date, min_lon, max_lon, min_lat, max_lat
            )

        if variable == MarsVariable.mslp:
            return self.fetch_mslp_operational_data(
                min_date, max_date, min_lon, max_lon, min_lat, max_lat
            )

        if variable == MarsVariable.z500:
            return self.fetch_z500_operational_data(
                min_date, max_date, min_lon, max_lon, min_lat, max_lat
            )

        raise ValueError(f"Unsupported MarsVariable: {variable}")

    def fetch_t2m_mean_operational_data(
        self,
        min_date: datetime,
        max_date: datetime,
        min_lon: float,
        max_lon: float,
        min_lat: float,
        max_lat: float,
    ) -> xr.Dataset:
        """
        Fetch mean 2m temperature operational data from MARS and return an xarray Dataset.

        This method retrieves analysis data from the ECMWF operational stream,
        processes the raw 6-hourly data into daily means using CDO, and
        applies longitude normalization and spatial filtering on the resulting Dataset.

        Parameters:
            min_date (datetime):
                The start date for the data retrieval.
            max_date (datetime):
                The end date for the data retrieval.
            min_lon (float):
                The minimum longitude for spatial filtering.
            max_lon (float):
                The maximum longitude for spatial filtering.
            min_lat (float):
                The minimum latitude for spatial filtering.
            max_lat (float):
                The maximum latitude for spatial filtering.

        Returns:
            xr.Dataset: A Dataset containing daily mean 2m temperature data
            with UTC timestamps.
        """
        time = "00:00:00/06:00:00/12:00:00/18:00:00"
        Utils.print(
            f"Fetching t2m data from MARS: {self.get_date_list(min_date, max_date)}"
        )
        request = {
            "class": "od",
            "date": self.get_date_list(min_date, max_date),
            "expver": "1",
            "param": self.find_param_code("t2m"),  # 2m temperature
            "grid": "0.25/0.25",  # 0.25 degree grid
            "time": time,
            "class": "od",
            "levtype": "sfc",
            "stream": "oper",
            "type": "an",
            "target": "output",
            "format": "netcdf",
        }
        temp_path = self.get_temp_path()
        self.server.execute(request, target=temp_path)

        # USE CDO to compute daily mean
        out_daily = temp_path.replace(".nc", "_daily.nc")
        subprocess.run(
            [
                "cdo",
                "-O",
                "-r",
                "-f",
                "nc4",
                "-s",
                f"-daymean",
                f"-shifttime,3hour",
                temp_path,
                out_daily,
            ],
            check=True,
        )

        ds = xr.open_dataset(out_daily, chunks=-1)
        ds = ds.rename({"time": "valid_time"})
        return self._normalize_dataset_and_filter_bbox(
            ds, min_lon, max_lon, min_lat, max_lat
        )

    def fetch_mslp_operational_data(
        self,
        min_date: datetime,
        max_date: datetime,
        min_lon: float,
        max_lon: float,
        min_lat: float,
        max_lat: float,
    ) -> xr.Dataset:
        """
        Fetch daily mean sea-level pressure operational data from MARS.

        This method retrieves 6-hourly analysis sea-level pressure data,
        computes daily means with CDO, and returns an xarray Dataset.
        """
        time = "00:00:00/06:00:00/12:00:00/18:00:00"
        Utils.print(
            f"Fetching mslp data from MARS: {self.get_date_list(min_date, max_date)}"
        )
        request = {
            "class": "od",
            "date": self.get_date_list(min_date, max_date),
            "expver": "1",
            "param": self.find_param_code("mslp"),
            "grid": "0.25/0.25",
            "time": time,
            "class": "od",
            "levtype": "sfc",
            "stream": "oper",
            "type": "an",
            "target": "output",
            "format": "netcdf",
        }
        temp_path = self.get_temp_path()
        self.server.execute(request, target=temp_path)

        out_daily = temp_path.replace(".nc", "_daily.nc")
        subprocess.run(
            [
                "cdo",
                "-O",
                "-r",
                "-f",
                "nc4",
                "-s",
                "-daymean",
                "-shifttime,3hour",
                temp_path,
                out_daily,
            ],
            check=True,
        )

        ds = xr.open_dataset(out_daily, chunks=-1)
        rename_map = {}
        if "time" in ds.coords:
            rename_map["time"] = "valid_time"
        if rename_map:
            ds = ds.rename(rename_map)

        return self._normalize_dataset_and_filter_bbox(
            ds, min_lon, max_lon, min_lat, max_lat
        )

    def fetch_z500_operational_data(
        self,
        min_date: datetime,
        max_date: datetime,
        min_lon: float,
        max_lon: float,
        min_lat: float,
        max_lat: float,
    ) -> xr.Dataset:
        """
        Fetch daily mean 500 hPa geopotential operational data from MARS.

        This method retrieves 6-hourly analysis geopotential data at 500 hPa,
        computes daily means with CDO, and returns an xarray Dataset.
        """
        time = "00:00:00/06:00:00/12:00:00/18:00:00"
        Utils.print(
            f"Fetching z500 data from MARS: {self.get_date_list(min_date, max_date)}"
        )
        request = {
            "class": "od",
            "date": self.get_date_list(min_date, max_date),
            "expver": "1",
            "param": self.find_param_code("z500"),
            "grid": "0.25/0.25",
            "time": time,
            "class": "od",
            "levtype": "pl",
            "levelist": "500",
            "stream": "oper",
            "type": "an",
            "target": "output",
            "format": "netcdf",
        }
        temp_path = self.get_temp_path()
        self.server.execute(request, target=temp_path)

        out_daily = temp_path.replace(".nc", "_daily.nc")
        subprocess.run(
            [
                "cdo",
                "-O",
                "-r",
                "-f",
                "nc4",
                "-s",
                "-daymean",
                "-shifttime,3hour",
                temp_path,
                out_daily,
            ],
            check=True,
        )

        ds = xr.open_dataset(out_daily, chunks=-1)
        rename_map = {}
        if "time" in ds.coords:
            rename_map["time"] = "valid_time"
        if rename_map:
            ds = ds.rename(rename_map)

        return self._normalize_dataset_and_filter_bbox(
            ds, min_lon, max_lon, min_lat, max_lat
        )

    def fetch_t2m_min_operational_data(
        self,
        min_date: datetime,
        max_date: datetime,
        min_lon: float,
        max_lon: float,
        min_lat: float,
        max_lat: float,
    ) -> xr.Dataset:
        """
        Fetch daily minimum 2m temperature operational data from MARS.

        This method retrieves 6-hourly forecast steps for minimum temperature,
        processes them into daily minimums using CDO with a time shift,
        and returns an xarray Dataset with spatial and temporal filtering applied.

        Parameters:
            min_date (datetime):
                The start date for the data retrieval.
            max_date (datetime):
                The end date for the data retrieval.
            min_lon (float):
                The minimum longitude for spatial filtering.
            max_lon (float):
                The maximum longitude for spatial filtering.
            min_lat (float):
                The minimum latitude for spatial filtering.
            max_lat (float):
                The maximum latitude for spatial filtering.

        Returns:
            xr.Dataset: A Dataset containing daily minimum 2m
            temperature data.
        """
        time = "00:00:00/12:00:00"
        Utils.print(
            f"Fetching t2m min data from MARS: {self.get_date_list(min_date, max_date)}"
        )
        request = {
            "class": "od",
            "step": "6/12",
            "date": self.get_date_list(min_date, max_date),
            "expver": "1",
            "param": self.find_param_code("tmin"),  # 2m temperature
            "grid": "0.25/0.25",  # 0.25 degree grid
            "time": time,
            "class": "od",
            "levtype": "sfc",
            "stream": "oper",
            "type": "fc",
            "target": "output",
            "format": "netcdf",
        }
        temp_path = self.get_temp_path()
        self.server.execute(request, target=temp_path)

        # USE CDO to compute daily min
        out_daily = temp_path.replace(".nc", "_daily.nc")
        subprocess.run(
            [
                "cdo",
                "-O",
                "-r",
                "-f",
                "nc4",
                "-s",
                f"-daymin",
                f"-shifttime,-3hour",
                temp_path,
                out_daily,
            ],
            check=True,
        )

        ds = xr.open_dataset(out_daily, chunks=-1)
        Utils.print(ds)
        ds = ds.rename({"mn2t6": "t2m", "time": "valid_time"})
        return self._normalize_dataset_and_filter_bbox(
            ds, min_lon, max_lon, min_lat, max_lat
        )

    def fetch_t2m_max_operational_data(
        self,
        min_date: datetime,
        max_date: datetime,
        min_lon: float,
        max_lon: float,
        min_lat: float,
        max_lat: float,
    ) -> xr.Dataset:
        """
        Fetch daily maximum 2m temperature operational data from MARS.

        This method retrieves 6-hourly forecast steps for maximum temperature,
        processes them into daily maximums using CDO, and returns a
        xarray Dataset with normalized longitudes.

        Parameters:
            min_date (datetime):
                The start date for the data retrieval.
            max_date (datetime):
                The end date for the data retrieval.
            min_lon (float):
                The minimum longitude for spatial filtering.
            max_lon (float):
                The maximum longitude for spatial filtering.
            min_lat (float):
                The minimum latitude for spatial filtering.
            max_lat (float):
                The maximum latitude for spatial filtering.

        Returns:
            xr.Dataset: A Dataset containing daily maximum 2m
            temperature data.
        """
        time = "00:00:00/12:00:00"
        # Fetch the current date -7 days as a list of dates
        Utils.print(
            f"Fetching t2m max data from MARS: {self.get_date_list(min_date, max_date)}"
        )
        request = {
            "class": "od",
            "step": "6/12",
            "date": self.get_date_list(min_date, max_date),
            "expver": "1",
            "param": self.find_param_code("tmax"),  # 2m temperature
            "grid": "0.25/0.25",  # 0.25 degree grid
            "time": time,
            "class": "od",
            "levtype": "sfc",
            "stream": "oper",
            "type": "fc",
            "target": "output",
            "format": "netcdf",
        }
        temp_path = self.get_temp_path()
        self.server.execute(request, target=temp_path)

        # USE CDO to compute daily max
        out_daily = temp_path.replace(".nc", "_daily.nc")
        subprocess.run(
            [
                "cdo",
                "-O",
                "-r",
                "-f",
                "nc4",
                "-s",
                f"-daymax",
                f"-shifttime,-3hour",
                temp_path,
                out_daily,
            ],
            check=True,
        )

        ds = xr.open_dataset(out_daily, chunks=-1)
        ds = ds.rename({"mx2t6": "t2m", "time": "valid_time"})
        return self._normalize_dataset_and_filter_bbox(
            ds, min_lon, max_lon, min_lat, max_lat
        )

    def fetch_total_precipitation_operational_data(
        self,
        min_date: datetime,
        max_date: datetime,
        min_lon: float,
        max_lon: float,
        min_lat: float,
        max_lat: float,
    ) -> xr.Dataset:
        """
        Fetch total daily precipitation operational data from MARS.

        This method retrieves accumulated precipitation data, calculates the
        daily sum via CDO, and returns a Dataset filtered to the specified
        bounding box with longitude coordinates translated to the -180 to 180 range.

        Parameters:
            min_date (datetime):
                The start date for the data retrieval.
            max_date (datetime):
                The end date for the data retrieval.
            min_lon (float):
                The minimum longitude for spatial filtering.
            max_lon (float):
                The maximum longitude for spatial filtering.
            min_lat (float):
                The minimum latitude for spatial filtering.
            max_lat (float):
                The maximum latitude for spatial filtering.

        Returns:
            xr.Dataset: A Dataset containing total daily
            precipitation data.
        """
        time = "00:00:00/12:00:00"
        Utils.print(
            f"Fetching tp data from MARS: {self.get_date_list(min_date, max_date)}"
        )
        request = {
            "class": "od",
            "step": "6/12",
            "date": self.get_date_list(min_date, max_date),
            "expver": "1",
            "param": self.find_param_code("tp"),  # total precipitation
            "grid": "0.25/0.25",  # 0.25 degree grid
            "time": time,
            "class": "od",
            "levtype": "sfc",
            "stream": "oper",
            "type": "fc",
            "target": "output",
            "format": "netcdf",
        }
        temp_path = self.get_temp_path()
        self.server.execute(request, target=temp_path)

        # USE CDO to compute daily sum
        out_daily = temp_path.replace(".nc", "_daily.nc")
        subprocess.run(
            ["cdo", "-O", "-r", "-f", "nc4", "-s", f"-daysum", temp_path, out_daily],
            check=True,
        )

        ds = xr.open_dataset(out_daily, chunks=-1)
        ds = ds.rename({"time": "valid_time"})
        return self._normalize_dataset_and_filter_bbox(
            ds, min_lon, max_lon, min_lat, max_lat
        )

    def find_param_code(self, name: str) -> str | None:
        """
        Find the ECMWF MARS parameter code for a given variable name.

        Maps human-readable variable names to their specific ECMWF GRIB
        parameter codes used for MARS archive queries.

        Parameters:
            name (str):
                The short name of the variable (e.g., "t2m", "tp", "tmin").

        Returns:
            str | None: The corresponding MARS parameter code string, or None
            if the variable name is not mapped.
        """
        param_codes = {
            "t2m": "167.128",
            "z500": "129.128",
            "mslp": "151.128",
            "tp": "228.128",
            "tmin": "122.128",
            "tmax": "121.128",
            # Add more mappings as needed
        }
        return param_codes.get(name)
