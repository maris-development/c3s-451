# Functions for analogues

# import subprocess
from typing import Any

from matplotlib.colors import ListedColormap
import numpy as np
import cartopy.crs as ccrs
import os
import scipy.stats as stats
import calendar
import numpy.ma as ma
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import datetime
import warnings
from shapely.geometry import Polygon, MultiPolygon
import matplotlib
import pandas as pd

from .utils import Utils

# iris doesnt work on windows
try:
    import iris
    import iris.analysis
    import iris.coord_categorisation # type: ignore
    import iris.util # type: ignore
except ImportError:
    Utils.print("cant import iris, are you on windows?")
    



class Analogues:

    ERA5FILESUFFIX : str = "_daily" 

    def __init__(self) -> None:
        Analogues.ERA5FILESUFFIX : str = "_daily"    

    @staticmethod
    def reanalysis_file_location() -> str:
        '''
        Return the location of the ERA5 data

        Returns:
            str:
                location of the ERA5 NetCDF data files
        '''
        #CEX path 
        #return os.path.join(os.environ["CLIMEXP_DATA"],"ERA5")
        #local wrkstation path 
        return '/net/pc230042/nobackup/users/sager/nobackup_2_old/ERA5-CX-READY/'

    @staticmethod
    def find_reanalysis_filename(var:str, daily:bool=True) -> str:
        '''
        Return the field filename for a given variable

        Parameters:
            var (str):
                Variable choice
            daily (bool):
                Daily values yes/no?

        Returns:
            str:
                File path
        '''

        suffix : str = Analogues.ERA5FILESUFFIX
        path : str = os.path.join(Analogues.reanalysis_file_location(),"era5_{0}{1}.nc".format(var,suffix))
        return path

    @staticmethod
    def event_data_era(event_data:str, date: list, ana_var: str) -> list:

        '''
        Get ERA data for a defined list of variables on a given event date

        Parameters:
            event_data (str): Daily or extended
            date (list): Selected date
            ana_var (str): Variable to import

        Returns
            list: List of cubes for selected date
        '''

        event_list = []
        # TODO: Convert variable list into a non-local (possibly with a class?)
        if event_data == 'extended':
            for variable in [ana_var, 'tp', 't2m', 't2m']:
                event_list.append(Analogues.reanalysis_data_single_date(variable, date))
        else:
            for variable in [ana_var, 'tp', 't2m', 'sfcWind']:
                event_list.append(Analogues.reanalysis_data_single_date(variable, date))
        return event_list

    @staticmethod
    def composite_dates_anomaly(cube: iris.cube.Cube, date_list: list) -> iris.cube.Cube:
        '''
        Returns single composite of all dates
        
        Parameters:
            cube (iris.cube.Cube):
                list of cubes, 1 per year - as used to calc D/date_list
            date_list (list):
                list of events to composite
        
        Returns:
            iris.cube.Cube:
                Mean composite anomaly field averaged over all valid dates in date_list.
        '''
        cube = cube - cube.collapsed(['latitude', 'longitude'], iris.analysis.MEAN)
        n = len(date_list)
        FIELD = 0
        for each in range(n):
            year = int(date_list[each][:4])
            month = calendar.month_abbr[int(date_list[each][4:-2])]
            day = int(date_list[each][-2:])
            NEXT_FIELD = Analogues.pull_out_day_era(cube, year, month, day)
            if NEXT_FIELD == None:
                Utils.print('Field failure for: ',+each)
                n = n-1
            else:
                if FIELD == 0:
                    FIELD = NEXT_FIELD
                else:
                    FIELD = FIELD + NEXT_FIELD
        return FIELD/n

    

    @staticmethod
    def regrid(original:iris.cube.Cube, new:iris.cube.Cube) -> iris.cube.Cube:
        '''
        Regrids onto a new grid
        
        Parameters:
            original (iris.cube.Cube):
                Original cube
            new (iris.cube.Cube):
                New grid
        
        Returns:
            iris.cube.Cube:
                Regridded cube
        '''

        mod_cs = original.coord_system(iris.coord_systems.CoordSystem)
        new.coord(axis='x').coord_system = mod_cs
        new.coord(axis='y').coord_system = mod_cs
        new_cube = original.regrid(new, iris.analysis.Linear())

        return new_cube

    @staticmethod
    def extract_region(
        cube_list: iris.cube.Cube|iris.cube.CubeList,
        region: list[float], 
        lat: str='latitude', 
        lon: str='longitude'
    ) -> tuple[iris.cube.Cube|iris.cube.CubeList, str, str]:
        '''
        Extract region using boundering box (defaults to Europe)

        Parameters:
            cube_list (iris.cube.Cube|iris.cube.Cube):
                Iris cube or cubelist
            region (list[float]):
                Region to extract
            lat (str):
                Latitude column
            lon (str):
                Longitude column
                reg_cubes (iris.cube.Cube|iris.cube.CubeList):
                
        Returns:
            reg_cubes (iris.cube.Cube|iris.cube.CubeList):
                Iris cube or cubelist consisting of only the extracted region
            lat (str):
                Latitude column
            lon (str):
                Longitude column
        '''

        const_lat = iris.Constraint(latitude = lambda cell:region[1] < cell < region[0])
        if isinstance(cube_list, iris.cube.Cube):
            reg_cubes_lat = cube_list.extract(const_lat)
            reg_cubes = reg_cubes_lat.intersection(longitude=(region[3], region[2]))
        elif isinstance(cube_list, iris.cube.CubeList):
            reg_cubes = iris.cube.CubeList([])
            for each in range(len(cube_list)):
                # Utils.print(each)
                subset = cube_list[each].extract(const_lat)
                reg_cubes.append(subset.intersection(longitude=(region[3], region[2])))
                
        return Analogues.guess_bounds(reg_cubes, lat=lat, lon=lon)

    @staticmethod
    def euclidean_distance(field:iris.cube.Cube, event:iris.cube.Cube) -> list:
        '''
        Returns list of euclidean distances
        BOTH MUST HAVE SAME DIMENSIONS FOR LAT/LON
        AREA WEIGHTING APPLIED

        Parameters:
            field (iris.cube.Cube):
                single cube of analogues field.
            event (iris.cube.Cube):
                cube of single day of event to match.
            
        Returns:
            list:
                List of euclidean distances
        '''

        D= [] # to be list of all euclidean distances
        if event.coord('latitude').has_bounds():
            pass
        else:
            event.coord('latitude').guess_bounds()
        if event.coord('longitude').has_bounds():
            pass
        else:
            event.coord('longitude').guess_bounds()
        weights=iris.analysis.cartography.area_weights(event)
        event=event*weights
        a, b, c =np.shape(field) 
        field=field*np.array([weights]*a) 
        XA= event.data.reshape(b*c,1)
        XB = field.data.reshape(np.shape(field.data)[0],b*c,1)
        for Xb in XB:
            D.append(np.sqrt(np.sum(np.square(XA-Xb)))) 
        return D

    @staticmethod
    def reanalysis_data(var:str, Y1:int=1950, Y2:int=2023, months:list[str]=['Jan']) -> iris.cube.Cube:
        '''
        Loads in reanalysis daily data

        Parameters:
            var (str):
                VAR can be psi250, msl, or tp (to add more)
            Y1 (int):
                Start year
            Y2 (int):
                End year
            months (list[str]):
                Months to subset

        Returns:
            iris.cube.Cube:
                Iris cube containing the selected variable for months from Y1 to Y2
        '''
        cubes = iris.load(Analogues.find_reanalysis_filename(var), var)
        try:
            cube = cubes[0]
        except:
            Utils.print("Error reading cubes for %s", var)
            raise FileNotFoundError
        iris.coord_categorisation.add_year(cube, 'time')
        cube = cube.extract(iris.Constraint(year=lambda cell: Y1 <= cell <= Y2))
        iris.coord_categorisation.add_month(cube, 'time')
        cube = cube.extract(iris.Constraint(month=months))
        return cube

    @staticmethod
    def top_separate_analogues_df(df, N, sep=5):
        '''
        Input data is a dataframe of dates and distance
        '''
        # ensure datetime
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        # remove NaN and zero distances
        df = df.dropna(subset=["distance"]) 
        df = df[df["distance"] != 0]
        # sort by ascending distance (best analogues first)
        df = df.sort_values("distance")  
        selected_dates = []
        for d in df["date"]:
            if not selected_dates:
                selected_dates.append(d)
                continue
            # check separation condition
            if all(abs((d - s).days) >= sep for s in selected_dates):
                selected_dates.append(d)
    
        # filter original dataframe
        result = df[df["date"].isin(selected_dates)]
    
        result_top = result.nsmallest(N, "distance")
        # sort back by date (optional, nicer output)
        result_top = result_top.sort_values("date")
    
        return result_top

    @staticmethod
    def anomaly_period_outputs(Y1:int, Y2:int, ana_var:str, N:int,
                               date:list, months:list[str], region:list[float]
                               ) -> list:
        '''
        Identify the N closest analogues of for event

        Parameters:
            Y1 (int):
                Start year
            Y2 (int):
                End year
            ana_var (str):
                Analogue variable used to identify similar days.
            N (int):
                Number of analogues
            date (list):
                date format: [YYYY, 'Mon', DD], e.g. [2021, 'Jul', 14])
            months (list[str]):
                List of month abbreviations used to restrict the seasonal window.
            region (list[float]):
                Spatial region used for analogue selection.
        
        Returns:
            list:
                List of the N closest analogue dates, given as strings in YYYYMMDD format.
        '''
        P1_msl = Analogues.reanalysis_data(ana_var, Y1, Y2, months) # Get ERA5 data, Y1 to Y2, for var and season chosen. Global.
        P1_field = Analogues.extract_region(P1_msl, region) # Extract the analogues domain (R1) from global field
        ### difference for anomaly version
        P1_spatialmean = P1_field.collapsed(['latitude', 'longitude'], iris.analysis.MEAN)   # Calculate spatial mean for each day
        P1_field = P1_field - P1_spatialmean # Remove spatial mean from each day
        event = Analogues.reanalysis_data_single_date(ana_var, date)
        E = Analogues.extract_region(event, region) # Extract domain for event field
        E = E - E.collapsed(['latitude', 'longitude'], iris.analysis.MEAN) # remove spatial mean for event field
        ###
        P1_dates = Analogues.analogue_dates_v2(E, P1_field,N) # calculate the closest analogues
        
        return P1_dates

    @staticmethod
    def cube_date(cube:iris.cube.Cube) -> tuple[int, str, int, Any]:
        '''
        Returns date of cube (assumes cube single day)

        Parameters:
            cube (iris.cube.Cube):
                Iris cube

        Returns:
            year (int):
                Year
            month (int|str):
                Month
            day (int):
                Day
            time (Any):
                Time
        '''
        if len(cube.coords('year')) > 0:
           pass
        else:
           iris.coord_categorisation.add_year(cube, 'time')
        if len(cube.coords('month')) > 0:
           pass
        else:
           iris.coord_categorisation.add_month(cube, 'time')
        if len(cube.coords('day_of_month')) > 0:
           pass
        else:
           iris.coord_categorisation.add_day_of_month(cube, 'time')
        if len(cube.coords('day_of_year')) > 0:
           pass
        else:
           iris.coord_categorisation.add_day_of_year(cube, 'time')
        year = cube.coord('time').units.num2date(cube.coord('time').points)[0].year
        month = cube.coord('time').units.num2date(cube.coord('time').points)[0].month
        day = cube.coord('time').units.num2date(cube.coord('time').points)[0].day
        time = cube.coord('time').points[0]
        return year, month, day, time

    @staticmethod
    def date_list_checks(date_list:list, days_apart:int=5) -> list:
        '''
        Takes date_list and removes:
        1) the original event (if present)
        2) any days within 5 days of another event

        Parameters:
            date_list (list):
                List of dates
            days_apart (int):
                Minimum number of days required between any two retained dates.
        
        Returns:
            list:
                Filtered list of dates in YYYYMMDD format, with dates closer than days_apart days removed.
        '''

        dates = []
        for each in date_list:
            dates.append(datetime.date(int(each[:4]), int(each[4:6]), int(each[6:])))
        new_dates = dates.copy()
        for i, each in enumerate(dates): # for each date in turn
            for other in dates[i+1:]: # go through all the rest of the dates
                if other in new_dates:
                    if abs((each-other).days) < days_apart:
                        new_dates.remove(other)
        new_dates_list = []
        for each in new_dates:
            new_dates_list.append(str(each.year)+str("{:02d}".format(each.month))+str("{:02d}".format(each.day)))
        return new_dates_list

    @staticmethod
    def eucdist_of_datelist(event:iris.cube.Cube, reanalysis_cubelist:iris.cube.Cube|iris.cube.CubeList,
                            date_list:list, region:list[float]) -> list:
        '''
        Calculate Euclidean distances between an event field and a list of dates.

        Parameters:
            event (iris.cube.Cube):
                Cube containing the event field to be compared.
            reanalysis_cubelist (iris.cube.Cube|iris.cube.CubeList):
                Daily reanalysis data from which fields corresponding to date_list are extracted.
            date_list (list):
                List of dates to compare against the event, given as strings in YYYYMMDD format.
            region (list[float]):
                Spatial region used to subset all fields prior to comparison.

        Returns:
            list:
                List of Euclidean distance values, one for each date in date_list,
                representing dissimilarity from the event field.
        '''

        ED_list = []
        E = Analogues.extract_region(event, region)
        if E.coord('latitude').has_bounds():
            pass
        else:
            E.coord('latitude').guess_bounds()
        if E.coord('longitude').has_bounds():
            pass
        else:
            E.coord('longitude').guess_bounds()
        weights = iris.analysis.cartography.area_weights(E)
        E = E*weights
        for i, each in enumerate(date_list):
            year = int(date_list[i][:4])
            month = calendar.month_abbr[int(date_list[i][4:-2])]
            day = int(date_list[i][-2:])
            field = Analogues.extract_region(Analogues.pull_out_day_era(reanalysis_cubelist, year, month, day), region)
            field = field*weights
            b, c = np.shape(field)
            XA = E.data.reshape(b*c,1)
            XB = field.data.reshape(b*c, 1)
            D = np.sqrt(np.sum(np.square(XA - XB)))
            ED_list.append(D)
        return ED_list

    @staticmethod
    def analogue_dates_v2(event:iris.cube.Cube, reanalysis_cube:iris.cube.Cube,
                          region:list[float], N:int) -> list:
        
        '''
        Identify analogue dates based on minimum Euclidean distance to an event field.

        The function extracts a specified region from both the event field and the
        reanalysis dataset, computes the Euclidean distance between the event and
        each reanalysis time slice, and returns the dates corresponding to the
        N closest analogue fields. A minimum temporal separation between returned
        dates is enforced.
    
        Parameters:
            event (iris.cube.Cube):
                Event field used as the reference for analogue selection.
            reanalysis_cube (iris.cube.Cube):
                Reanalysis dataset containing candidate analogue fields.
            region (list[float]):
                Spatial region to extract, typically given as
            N (int):
                Number of analogue dates to identify.
    
        Returns:
            date_list2 (list[str]):
                List of analogue dates in YYYYMMDD string format, filtered to ensure
                a minimum separation in time between events and minimum euclidian distance.
        '''
        
        var_e = Analogues.extract_region(event, region)
        reanalysis_cube = Analogues.extract_region(reanalysis_cube, region)
        var_d = Analogues.euclidean_distance(reanalysis_cube, var_e)
       
        time_coord = reanalysis_cube.coord("time")
        dates = time_coord.units.num2date(time_coord.points)
        time_str = [f"{d.year:04d}-{d.month:02d}-{d.day:02d}" for d in dates]

        ec_df=pd.DataFrame({'date':time_str,'distance':var_d})
        top_N_analogues=Analogues.top_separate_analogues_df(ec_df,N)
        final_date_list = top_N_analogues["date"].dt.strftime("%Y%m%d").tolist()

        return final_date_list

    @staticmethod
    def pull_out_day_era(psi: iris.cube.Cube, sel_year: int, sel_month: str|int, sel_day: int) -> iris.cube.Cube|None:

        """
        Extract a single daily field for a given date from ERA reanalysis data.

        The function accepts either a single Iris cube or an iterable of cubes.
        It searches for the specified year, month, and day, and returns the
        corresponding daily field if present.

        Parameters:
            psi (iris.cube.Cube or iterable of iris.cube.Cube):
                Reanalysis data from which to extract the requested day.
            sel_year (int):
                Year to extract.
            sel_month (str):
                Month abbreviation (e.g. 'Jan', 'Feb', ...).
            sel_day (int):
                Day of the month.

        Returns:
            iris.cube.Cube | None:
                Cube containing data for the selected date.
                Or None if the requested date is not present in the data.
                
        """
        
        psi_day = None
        
        if type(psi) == iris.cube.Cube:
            psi_day = Analogues.extract_date(psi, sel_year, sel_month, sel_day)
        else:
            for each in psi:
                if len(each.coords('year')) > 0:
                    pass
                else:
                    iris.coord_categorisation.add_year(each, 'time')
                if each.coord('year').points[0]==sel_year:
                    psi_day = Analogues.extract_date(each, sel_year, sel_month, sel_day)
                else:
                    pass
        try:
            return psi_day
        except NameError:
            Utils.print('ERROR: Date not in data')
            return

    @staticmethod
    def extract_date(cube: iris.cube.Cube, year: int, month: str|int, day: int) -> iris.cube.Cube:
        '''
        Extract specific day from cube of a single year

        Parameters:
            cube (iris.cube.Cube):
                Iris cube with daily values
            year (int):
                Year
            month (str|int):
                Month
            day (int):
                Day
        
        Returns:
            iris.cube.Cube:
                Iris cube of a single date
        '''
        if len(cube.coords('year')) > 0:
            pass
        else:
            iris.coord_categorisation.add_year(cube, 'time')
        if len(cube.coords('month')) > 0:
            pass
        else:
            iris.coord_categorisation.add_month(cube, 'time')
        if len(cube.coords('day_of_month')) > 0:
            pass
        else:
            iris.coord_categorisation.add_day_of_month(cube, 'time')
        return cube.extract(iris.Constraint(year=year, month=month, day_of_month=day))

    @staticmethod
    def composite_dates(psi:iris.cube.Cube, date_list:list) -> iris.cube.Cube:
        """
        Returns a single composite field from a list of event dates.

        The function extracts the daily field corresponding to each date in
        ``date_list`` from the input cube and computes the mean composite.
        Dates that are not present in the data are skipped.

        Parameters:
            psi(iris.cube.Cube):
                Reanalysis data cube containing daily fields spanning multiple years.
            date_list (list):
                List of event dates in YYYYMMDD string format to be composited.

        Returns:
            composite (iris.cube.Cube):
                Mean composite field averaged over all valid dates.
        """

        n = len(date_list)
        FIELD = 0
        for each in range(n):
            year = int(date_list[each][:4])
            month = calendar.month_abbr[int(date_list[each][4:-2])]
            day = int(date_list[each][-2:])
            NEXT_FIELD = Analogues.pull_out_day_era(psi, year, month, day)
            if NEXT_FIELD == None:
                Utils.print('Field failure for: ',+each)
                n = n-1
            else:
                if FIELD == 0:
                    FIELD = NEXT_FIELD
                else:
                    FIELD = FIELD + NEXT_FIELD
        return FIELD/n

    @staticmethod
    def reanalysis_data_single_date(var:str, date:list) -> iris.cube.Cube:
        '''
        Loads in reanalysis daily data for single date

        Parameters:
            var (str):
                Either 'z500', 'msl', 't2m', 'tp'.... more to be added
            date (list):
                Date to extract
        
        Returns:
            iris.cube.Cube:
                Iris cube containing a single date
        '''

        filename = Analogues.find_reanalysis_filename(var)
        # Utils.print("Read file: {} for date {}".format(filename,date))
        cube = iris.load(filename, var)[0]
        cube = Analogues.extract_date(cube,date[0],date[1],date[2])
        return cube

    @staticmethod
    def quality_analogs(field:iris.cube.Cube, date_list:list, N:int,
                        analogue_variable:str, region:list[float]
                        ) -> list:
        '''
        For the 30 closest analogs of the event day, calculates analogue quality

        Parameters;
            field (iris.cube.Cube):
                Iris cube
            date_list (list):
                List of cubes
            N (int):
                Number of analogues?
            analogue_variable (str):
                Variable
            region (list[float]):
                Region to subset the data on
        
        Returns:
            list:
                Quality list
        '''
        Q = []
        filename = Analogues. find_reanalysis_filename(analogue_variable)
        cube = iris.load(filename, analogue_variable)[0]
        for i, each in enumerate(date_list):
            year = int(each[:4])
            month = calendar.month_abbr[int(each[4:-2])]
            day = int(each[-2:])
            cube = Analogues.extract_date(cube,year,month,day)
            cube = Analogues.extract_region(cube,region)
            D = Analogues.euclidean_distance(field, cube) # calc euclidean distances
            Q.append(np.sum(np.sort(D)[:N]))
        return Q

    @staticmethod
    def diff_significance(cube_past:iris.cube.Cube, dates_past:list,
                          cube_prst:iris.cube.Cube, dates_prst:list) -> iris.cube.Cube:
        '''
        Returns single composite of all dates

        Parameters:
            cube_past (iris.cube.Cube):
                Iris cube with daily data of either 'z500', 'slp', t2m', or 'tp'
            dates_past (list):
                List of past dates used for cube_past
            cube_prst (iris.cube.Cube):
                Iris cube with daily data of either 'z500', 'slp', t2m', or 'tp'
            dates_prst (list):
                List of present dates used for cube_prst
            
        Returns
            iris.cube.Cube:
                Iris cube composite of all dates (dates_past & dates_prst)
        '''   

        n = len(dates_past)
        field_list1 = iris.cube.CubeList([])
        for each in range(n):
            year = int(dates_past[each][:4])
            month = calendar.month_abbr[int(dates_past[each][4:-2])]
            day = int(dates_past[each][-2:])
            field_list1.append(Analogues.pull_out_day_era(cube_past, year, month, day))
        n = len(dates_prst)
        field_list2 = iris.cube.CubeList([])
        for each in range(n):
            year = int(dates_prst[each][:4])
            month = calendar.month_abbr[int(dates_prst[each][4:-2])]
            day = int(dates_prst[each][-2:])
            field_list2.append(Analogues.pull_out_day_era(cube_prst, year, month, day))    
        sig_field = field_list1[0].data
        a, b = np.shape(field_list1[0].data)
        for i in range(a):
            # Utils.print(i)
            for j in range(b):
                loc_list1 = []; loc_list2 = []
                for R in range(n):
                    loc_list1.append(field_list1[R].data[i,j])
                    loc_list2.append(field_list2[R].data[i,j])
                    # Trap and avoid precision loss warning
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        u, p = stats.ttest_ind(loc_list1, loc_list2, equal_var=False, alternative='two-sided')
                if p < 0.05:
                    sig_field[i,j] = 1
                else:
                    sig_field[i,j] = 0
        result_cube = field_list1[0]
        result_cube.data = sig_field
        return result_cube

    @staticmethod
    def composite_dates_ttest(cube:iris.cube.Cube, date_list:list) -> iris.cube.Cube:
        '''
        Returns single composite of all dates
        
        Parameters:
            cube (iris.cube.Cube):
                Iris cube with daily data of either 'z500', 'slp', t2m', or 'tp'
            date_list (list):
                List of dates

        Returns:
            iris.cube.Cube:
                Iris cube of all dates
        '''

        n = len(date_list)
        field_list = iris.cube.CubeList([])
        for each in range(n):
            year = int(date_list[each][:4])
            month = calendar.month_abbr[int(date_list[each][4:-2])]
            day = int(date_list[each][-2:])
            field_list.append(Analogues.pull_out_day_era(cube, year, month, day))
        sig_field = field_list[0].data
        a, b = np.shape(field_list[0].data)
        for i in range(a):
            # Utils.print(i)
            for j in range(b):
                loc_list = []
                for R in range(n):
                    loc_list.append(field_list[R].data[i,j])
                t_stat, p_val = stats.ttest_1samp(loc_list, 0)
                if p_val < 0.05:
                    sig_field[i,j] = 1
                else:
                    sig_field[i,j] = 0
        result_cube = field_list[0]
        result_cube.data = sig_field
        return result_cube

    @staticmethod
    def impact_index(cube:iris.cube.Cube, region:list[float]) -> np.ma.MaskedArray:
        '''
        Calculates the impact index over the cube

        Parameters:
            cube (iris.cube.Cube):
                Iris cube with daily data of either 'z500', 'slp', t2m', or 'tp'
            region (list[float]):
                Region used for subsetting
        
        Returns:
            np.ma.MaskedArray:
                Impact index iris cube
        '''

        cube = Analogues.extract_region(cube, region)
        cube.coord('latitude').guess_bounds()
        cube.coord('longitude').guess_bounds()
        grid_areas = iris.analysis.cartography.area_weights(cube)
        cube.data = ma.masked_invalid(cube.data)
        return cube.collapsed(('longitude','latitude'),iris.analysis.MEAN,weights=grid_areas).data

    @staticmethod
    def analogues_list(cube:iris.cube.Cube, date_list:list) -> list[iris.cube.Cube]:
        '''
        Takes single cube of single variable that includes dates in date_list
        Pulls out single days of date
        Returns as list of cubes

        Parameters:
            cube (iris.cube.Cube):
                single cube of a single variable
            date_list (list):
                list of dates in format 'YYYYMMDD'
        
        Returns:
            list[iris.cube.Cube]:
                A list of cubes
        '''

        n = len(date_list)
        x = []
        for each in range(n):
            year = int(date_list[each][:4])
            month = calendar.month_abbr[int(date_list[each][4:-2])]
            day = int(date_list[each][-2:])
            NEXT_FIELD = Analogues.pull_out_day_era(cube, year, month, day)
            if NEXT_FIELD == None:
                Utils.print('Field failure for: ',+each)
                n = n-1
            x.append(NEXT_FIELD)
        return x

    @staticmethod
    def plot_analogue_months(PAST:list, PRST:list) -> None:
        '''
        Produces histogram of number of analogues in each calendar month

        Parameters:
            PAST (list):
                List of dates of past period analogues, format ['19580308', ...
            PRST (list):
                List of dates of present period analogues, format ['19580308', ...
        
        Returns:
            None
        '''

        # List of months (as number)
        PAST_MONTH = []
        for each in PAST:
            PAST_MONTH.append(int(each[4:6]))
        PRST_MONTH = []
        for each in PRST:
            PRST_MONTH.append(int(each[4:6]))
        plt.hist(PAST_MONTH, np.arange(.5, 13, 1), alpha=.5, label='Past (1950-1980)')
        plt.hist(PRST_MONTH, np.arange(.5, 13, 1), color='r', alpha=.5, label='Present (1994-2024)')
        plt.xticks(np.arange(1, 13, 1),['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
        plt.legend()
        plt.xlim([0.5, 12.5])
        #plt.yticks(np.arange(0, 25, 5))
        plt.ylabel('Frequency')
        plt.xlabel('Month')
        plt.title('Monthly distribution of top circulation analogues')

    @staticmethod
    def set_coord_system(cube:iris.cube.Cube,
                         chosen_system:iris.analysis.cartography=iris.analysis.cartography.DEFAULT_SPHERICAL_EARTH_RADIUS
                         ) -> iris.cube.Cube:
        '''
        This is used to prevent warnings that no coordinate system defined
        Defaults to DEFAULT_SPHERICAL_EARTH_RADIUS

        Parameters:
            cube (iris.cube.Cube):
                Iris cube
            chosen_system (iris.analysis.cartography):
                Coordinate system
        
        Returns:
            iris.cube.Cube:
                Iris cube with applied coordinate system
        '''
        cube.coord('latitude').coord_system = iris.coord_systems.GeogCS(chosen_system)
        cube.coord('longitude').coord_system = iris.coord_systems.GeogCS(chosen_system)
        return cube

    ######################################################################################
    # MARIS added functions
    # most of these are repeated functions in the analogues
    ######################################################################################

    # Analogue rewrites
    ######################################################################################

    @staticmethod
    def analogue_months(event_date:list) -> list[str]:
        '''
        Return the months surrounding the event month
        
        Parameters:
            event_date: [year, month, day] e.g. [2024, 'Mar', 10]

        Returns:
            list[str]: Three month abbreviations [previous, current, next] e.g. ['Feb', 'Mar', 'Apr'].
        '''

        X = list(calendar.month_abbr)
        i = event_date if type(event_date) is int else X.index(event_date)
        if 1<i<12:
            months = [X[i-1], X[i], X[i+1]]
        elif i == 1:
            months = [X[12], X[i], X[i+1]]
        elif i == 12:
            months = [X[i-1], X[i], X[1]]

        return months

    @staticmethod
    def number_of_analogues(Y1:int, Y2:int, months:list[str]) -> int:
        '''
        Return the number of analogues in period Y1 to Y2 for the months given

        Parameters:
            Y1 (int): First year period
            Y2 (int): Last year period
            months (list[str]): Three month abbreviations [previous, current, next] e.g. ['Feb', 'Mar', 'Apr']
        
        Returns:
            int: Number of analouges to be used for calculations
        '''

        return int(((Y2-Y1)*len(months)*30)/100)

    @staticmethod
    def find_reanalysis_filename_v2(var:str, daily:bool = True) -> str:
        '''
        Return the field filename for a given variable

        Parameters:
            var (str): Variable
            daily (bool): ???

        Returns:
            str: Relative file location
        '''

        suffix : str = Analogues.ERA5FILESUFFIX
        path : str = os.path.join("", "era5_{0}{1}.nc".format(var,suffix))
        return path

    @staticmethod
    def reanalysis_data_v2(var:str, Y1:int=1950, Y2:int=2023, months:list[str]='[Jan]') -> iris.cube.Cube:
        '''
        Loads in reanalysis daily data

        Parameters:
            var (str): can be msl, or tp (to add more)
            Y1 (int): First year period
            Y2 (int): Last year period
            months (list[str]): Three month abbreviations [previous, current, next] e.g. ['Feb', 'Mar', 'Apr']

        Returns
            iris.cube.Cube: Loaded iris cube from NetCDF from Y1 to Y2 for selected months
        '''

        cubes = iris.load(Analogues.find_reanalysis_filename_v2(var), var)
        try:
            cube = cubes[0]
        except:
            Utils.print("Error reading cubes for %s", var)
            raise FileNotFoundError
        iris.coord_categorisation.add_year(cube, 'time')
        cube = cube.extract(iris.Constraint(year=lambda cell: Y1 <= cell <= Y2))
        iris.coord_categorisation.add_month(cube, 'time')
        cube = cube.extract(iris.Constraint(month=months))
        return cube
    
    @staticmethod
    def nc_to_cube(path:str) -> iris.cube.Cube:
        '''
        Load in NetCDF file to iris cube

        Parameters:
            path (str): Relative file location

        Returns:
            iris.cube.Cube: Loaded iris cube from NetCDF
        '''

        cubes = iris.load(path)
        try:
            cube = cubes[0]
        except:
            Utils.print("Error reading cubes for %s", path)
            raise FileNotFoundError
        return cube
    
    def extract_months(cube:iris.cube.Cube, months:list[str]) -> iris.cube.Cube:
        '''
        Extract months from cube

        Parameters:
            cube (iris.cube.Cube): Iris cube
            months (list[str]): Three month abbreviations [previous, current, next] e.g. ['Feb', 'Mar', 'Apr']

        Returns:
            iris.cube.Cube: Extracted iris cube for selected months
        '''

        if not any(coord.long_name == 'month' for coord in cube.aux_coords):
            iris.coord_categorisation.add_month(cube, 'time')

        cube = cube.extract(iris.Constraint(month=months))
        return cube

    @staticmethod
    def reanalysis_data_single_date_v2(var:str, date:list) -> iris.cube.Cube:
        '''
        Loads in reanalysis daily data

        Parameters:
            var (str): Can be msl, or tp (to add more)
            date (list): Single date for extracting the cube
                [year, month, day] e.g. [2024, 'Mar', 10]

        Returns:
            iris.cube.Cube: Loaded iris cube from NetCDF for selected date
        '''

        filename = Analogues.find_reanalysis_filename_v2(var)
        # Utils.print("Read file: {} for date {}".format(filename,date))
        cube = iris.load(filename, var)[0]
        cube = Analogues.extract_date(cube,date[0],date[1],date[2])
        return cube

    @staticmethod
    def extract_region_shape(cube:iris.cube.Cube, shape:Polygon, lat:str='latitude', lon:str='longitude') -> iris.cube.Cube:
        '''
        Extract Region using a shape e.g. shapefile or polygon

        Parameters:
            cube (iris.cube.Cube): Single cube
            shape (Polygon): Polygon to mask cube with

        Returns
            iris.cube.Cube: Masked cube
        '''

        masked_cube = iris.util.mask_cube_from_shape(cube=cube, shape=shape)

        return Analogues.guess_bounds(masked_cube, lat=lat, lon=lon)

    @staticmethod
    def event_data_era_v2(event_data:str, date:list, ana_var:str) -> list:
        '''
        Get ERA data for a defined list of variables on a given event date

        Parameters:
            event_data (str): Daily or extended
            date (list): Selected date
            ana_var (str): Variable to import

        Returns
            list: List of cubes for selected date
        '''

        event_list = []
        # TODO: Convert variable list into a non-local (possibly with a class?)
        if event_data == 'extended':
            for variable in [ana_var, 'tp', 't2m', 't2m']:
                event_list.append(Analogues.reanalysis_data_single_date_v2(variable, date))
        else:
            for variable in [ana_var, 'tp', 't2m', 'sfcWind']:
                event_list.append(Analogues.reanalysis_data_single_date_v2(variable, date))
        return event_list

    @staticmethod
    def extract_year(cube:iris.cube.Cube, Y1:int, Y2:int) -> iris.cube.Cube:
        '''
        Subsets the cube for given year range

        Parameters:
            cube (iris.cube.Cube): Cube to subset
            Y1 (int): First year period
            Y2 (int): Last year period
        
        Returns:
            iris.cube.Cube: Subsetted cube
        '''

        if not any(coord.name() == "year" for coord in cube.coords()):
            iris.coord_categorisation.add_year(cube, 'time')

        # is this actually checking if the Y1 and Y2 are in cube year range?
        rCube = cube.extract(iris.Constraint(year=lambda cell: Y1 <= cell <= Y2))
        return rCube

    @staticmethod
    def extract_date_v2(cube:iris.cube.Cube, date:list) -> iris.cube.Cube:
        '''
        Subset cube for specific date (single day)

        Paramters:
            cube (iris.cube.Cube): Cube to subset
            date (list): Date
        
        Returns:
            iris.cube.Cube: Subsetted cube
        '''

        year = date[0]
        month = date[1]
        day = date[2]

        if len(cube.coords('year')) > 0:
            pass
        else:
            iris.coord_categorisation.add_year(cube, 'time')
        if len(cube.coords('month')) > 0:
            pass
        else:
            iris.coord_categorisation.add_month(cube, 'time')
        if len(cube.coords('day_of_month')) > 0:
            pass
        else:
            iris.coord_categorisation.add_day_of_month(cube, 'time')
        return cube.extract(iris.Constraint(year=year, month=month, day_of_month=day))

    @staticmethod
    def analogue_dates_v3(daily_cube:iris.cube.Cube, event_cube:iris.cube.Cube, N:int) -> list:
        '''
        Identify analogue dates for a given event field.

        The function computes the Euclidean distance between a single-day event
        field and all daily fields in the input cube, then returns the dates
        corresponding to the N closest analogue fields. A minimum temporal
        separation between returned dates is enforced.

        Parameters:
            daily_cube (iris.cube.Cube):
                Cube containing daily fields from which analogue dates are selected.
            event_cube (iris.cube.Cube):
                Cube containing a single-day event field used as the reference.
            N (int):
                Number of analogues
        
        Returns:
            List of analogue dates in YYYYMMDD string format, filtered to ensure
            a minimum separation in time between events.
        '''

        
        D = Analogues.euclidean_distance(daily_cube, event_cube)

        time_coord = daily_cube.coord("time")
        dates = time_coord.units.num2date(time_coord.points)
        time_str = [f"{d.year:04d}-{d.month:02d}-{d.day:02d}" for d in dates]

        ec_df=pd.DataFrame({'date':time_str,'distance':D})
        top_N_analogues=Analogues.top_separate_analogues_df(ec_df,N)
        date_list2 = top_N_analogues["date"].dt.strftime("%Y%m%d").tolist()

        return date_list2

    # anaomaly period output for cubes

    # cube with daily values, cube of just the event, variable used, date of the event, region, number of analogues
    @staticmethod
    def anomaly_period_outputs_v2(daily_cube:iris.cube.Cube, event_cube:iris.cube.Cube, date:list, N:int) -> list:
        '''
        Identify analogue dates for an event after removing spatial means.

        The function computes anomaly fields by removing the spatial mean from
        both the daily dataset and the event field, identifies analogue dates
        using Euclidean distance, and returns a list of analogue dates excluding
        the event date itself.

        Parameters:
            daily_cube (iris.cube.Cube):
                Cube with daily values
            event_cube (iris.cube.Cube):
                Cube of just a single day
            date (list):
                Date of the event [year, month abbrev, day]
            N (int):
                Number of analogues
        
        Returns
            list:
                List of analogue dates in YYYYMMDD string format, excluding the
                original event date.
        '''

        # cube_daily = cube_daily[:, 0, :, :]  # shape = (time, lat, lon)
        # multiple days
        daily_cube = daily_cube - daily_cube.collapsed(['latitude', 'longitude'], iris.analysis.MEAN) # event for anavar to plot (fig a)

        # cube_event = cube_event[:, 0, :, :]  # shape = (time, lat, lon)
        # single day
        event_cube = event_cube - event_cube.collapsed(['latitude', 'longitude'], iris.analysis.MEAN) # event for anavar to plot (fig a)

        P1_dates = Analogues.analogue_dates_v3(daily_cube, event_cube, N)


        return P1_dates

    # J: add return type
    @staticmethod
    def analogues_composite_anomaly_v2(cube:iris.cube.Cube, dates:list) -> iris.cube.Cube:
        '''
        Compute a composite anomaly field from analogue dates.

        The function removes the spatial mean from each daily field in the input
        cube and computes a mean composite anomaly over the specified analogue
        dates.

        Parameters:
            cube (iris.cube.Cube):
                Single cube with daily values
            dates (list):
                List of dates

        Returns:
            iris.cube.Cube:
                Composite anomaly field averaged over all analogue dates.
        '''

        P1_spatialmean = cube.collapsed(['latitude', 'longitude'], iris.analysis.MEAN)   # Calculate spatial mean for each day
        P1_field = cube - P1_spatialmean # Remove spatial mean from each day
        P1_comp = Analogues.composite_dates_anomaly(P1_field, dates) # composite analogues
        return P1_comp

    # J: add return type
    @staticmethod
    def analogues_composite_v2(cube:iris.cube.Cube, dates:list) -> iris.cube.Cube|float:
        '''
        Compute a composite field from analogue dates.

        The function extracts daily fields corresponding to the provided dates
        from the input cube and computes their mean composite. Dates that cannot
        be extracted are skipped.

        Parameters:
            cube (iris.cube.Cube):
                Single cube with daily values
            dates (list):
                List of dates

        Returns:
            iris.cube.Cube|float:
                Mean composite field averaged over all valid dates. A float may be
                returned if no valid composite could be constructed.
        '''

        P1_comp = Analogues.composite_dates(cube, dates) # composite analogues
        return P1_comp

    # J: add return type
    # correlation value
    @staticmethod
    def var_correlation(var_cube:iris.cube.Cube, correlation_cube:iris.cube.Cube
                        ) -> tuple[iris.cube.Cube, np.ndarray, np.ndarray]:
        '''
        Calculates the correlation between var_cube and correlation_cube

        Parameters:
            var_cube (iris.cube.Cube):
                Cube with daily values of variable t2m or tp
            correlation_cube (iris.cube.Cube):
                Cube with daily values of variable z500 or slp/msl

        Returns:
            z_data (iris.cube.Cube):
                Anomaly cube of the correlation variable with the spatial mean
                removed at each time step.
            corr_field (np.ndarray):
                Spatial field of Pearson correlation coefficients between the
                event time series and each grid point of the correlation cube.
            p_field (np.ndarray):
                Spatial field of p-values associated with the Pearson correlations.
                
        '''

        z_data = correlation_cube 
        z_data = z_data - z_data.collapsed(['latitude', 'longitude'], iris.analysis.MEAN) # event for anavar to plot (fig a)
        a, b, c = np.shape(z_data.data)
        corr_field = np.empty((b,c))
        p_field = np.empty((b,c))
        for i in np.arange(b):
            for j in np.arange(c):
                x, y = stats.pearsonr(var_cube.data, z_data.data[:,i,j])
                corr_field[i,j] = x
                p_field[i,j] = y

        # p_field does not seem to be used
        return z_data, corr_field, p_field

    @staticmethod
    def impact_index_v2(cube:iris.cube.Cube) -> np.ma.MaskedArray:
        '''
        Calculate the spatially averaged impact index over a cube.

        The function computes an area-weighted mean over latitude and longitude
        after masking invalid values, producing a single impact index value
        for each time step.

        Parameters:
            cube (iris.cube.Cube):
                Cube containing the spatial extent over which the impact index is calculated.

        Returns:
            np.ma.MaskedArray:
                Area-weighted mean impact index collapsed over latitude and longitude.
        '''

        grid_areas = iris.analysis.cartography.area_weights(cube)
        cube.data = ma.masked_invalid(cube.data)
        return cube.collapsed(('longitude','latitude'),iris.analysis.MEAN,weights=grid_areas).data

    # Processing
    ######################################################################################

    @staticmethod
    def blue_red_ratio(z_data:iris.cube.Cube, correlation_field:np.ndarray, domain:list[float]) -> tuple[float, float]:
        '''
        Calculate the blue/red ratio for a given correlation field and domain

        Parameters:
            z_data (iris.cube.Cube):
                Anomaly cube of the correlation variable with the spatial mean
                removed at each time step.
            correlation_field (np.ndarray):
                Spatial field of Pearson correlation coefficients between the event
                time series and each grid point of the correlation cube.
            domain (list[float]):
                Domain to subset the data on    

        Returns:
            blue, red (tuple[float, float]):
                Blue/red ratio and the percentage of significant grid points in the domain
        '''

        X = z_data[0,:,:] 
        X.data = correlation_field
        Y = Analogues.extract_region(X, domain)

        a,b = np.shape(Y.data)
        blue = 0
        red = 0
        for i in np.arange(a):
            for j in np.arange(b):
                if Y.data[i,j] < 0:
                    blue +=1
                else:
                    red +=1
        
        return (blue, red)



    # Plotting
    ######################################################################################

    @staticmethod
    def plot_box(axs:matplotlib.axes.Axes, bbox:list[float]) -> None:
        '''
        Draws bounding box on plot

        Parameters:
            axs (matplotlib.axes.Axes):
                Plot to draw boundingbox on
            bbox (list[float]):
                Bounding box

        Returns:
            None
        '''

        axs.plot([bbox[3], bbox[2]], [bbox[1], bbox[1]],'k')
        axs.plot([bbox[3], bbox[2]], [bbox[0], bbox[0]],'k')
        axs.plot([bbox[3], bbox[3]], [bbox[1], bbox[0]],'k')
        axs.plot([bbox[2], bbox[2]], [bbox[1], bbox[0]],'k')

    # adds background to plots
    @staticmethod
    def background(ax:matplotlib.axes.Axes) -> None:
        '''
        Adds background to given plot (ax)

        Parameters:
            ax (matplotlib.axes.Axes):
                Plot axes to add background to

        Returns:
            None
        '''

        ax.coastlines(linewidth=0.4)
        #ax.add_feature(cf.BORDERS, lw = 1, alpha = 0.7, ls = "--", zorder = 99)
        gl = ax.gridlines(draw_labels=True, x_inline=False, y_inline=False, linewidth=0.2, color='k',alpha=0.5,linestyle='--')
        gl.right_labels =gl.left_labels = gl.top_labels = gl.bottom_labels= False
        gl.xlabel_style = {'size': 5, 'color': 'gray'}
        gl.ylabel_style = {'size': 5, 'color': 'gray'}

    # Plot map of correlation
    @staticmethod
    def plot_correlation_map(z_data:iris.cube.Cube, z500_correlation:np.ndarray,
                             slp_correlation:np.ndarray,region:list[float],
                             z500_domain: list[float], slp_domain:list[float],
                             draw_labels:bool=True, fig_size:tuple[float, float]=(10,10)
                            ) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:

        '''
        Plots the correlation figures for z500 and slp

        Parameters:
            z_data (iris.cube.Cube):
                Event data
            z500_correlation (np.ndarray):
                z500 correlation data
            slp_correlation (np.npdarray):
                slp correlation data
            region (list[float]):
                Event region
            z500_domain (list[float]):
                z500 region
            slp_domain (list[float]):
                slp region
            draw_labels (bool):
                Draw labels?
            fig_size (tuple[float, float]):
                Figure size

        Returns:
            fig (matplotlib.figure.Figure):
                Correlation map figure
            ax (matplotlib.axes.Axes):
                Correlation map plot

        '''

        fig, ax = plt.subplots(2, 1, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=fig_size)
        lats = z_data.coord('latitude').points
        lons = z_data.coord('longitude').points
        con_lev = np.linspace(-1, 1, 20)

        c1 = ax[0].contourf(lons, lats, z500_correlation, levels=con_lev, cmap='RdBu_r', transform=ccrs.PlateCarree())
        ax[0].add_feature(cfeature.COASTLINE)
        ax[0].coastlines(linewidth=0.4)
        ax[0].set_title('Corr Z500')
        ax[0].gridlines(draw_labels=draw_labels)
        Analogues.plot_region(ax[0], region)
        Analogues.plot_region(ax[0], z500_domain)

        c1 = ax[1].contourf(lons, lats, slp_correlation, levels=con_lev, cmap='RdBu_r', transform=ccrs.PlateCarree())
        ax[1].add_feature(cfeature.COASTLINE)
        ax[1].coastlines(linewidth=0.4)
        ax[1].set_title('Corr SLP')
        ax[1].gridlines(draw_labels=draw_labels)
        Analogues.plot_region(ax[1], region)
        Analogues.plot_region(ax[1], slp_domain)

        fig.subplots_adjust(right=0.8)
        cax = fig.add_axes([0.85, 0.3, 0.01, 0.4])
        cbar = fig.colorbar(c1, cax=cax, ticks=[-1, 0, 1])
        cbar.ax.set_yticklabels(['-1', '0', '1'])
        plt.tight_layout()

        return fig, ax
    
    @staticmethod
    def plot_region(axs, region):
        """
        Draw region on plot.

        Parameters
        ----------
        axs : matplotlib.axes.Axes
            The axes to draw on.
        region : list[float], shapely.geometry.Polygon or shapely.geometry.MultiPolygon
            Either bounding box [min_lat, max_lat, min_lon, max_lon],
            a Polygon or a MultiPolygon.
        """
        if isinstance(region, list):
            # bounding box
            axs.plot([region[3], region[2]], [region[1], region[1]], 'k')
            axs.plot([region[3], region[2]], [region[0], region[0]], 'k')
            axs.plot([region[3], region[3]], [region[1], region[0]], 'k')
            axs.plot([region[2], region[2]], [region[1], region[0]], 'k')

        elif isinstance(region, Polygon):
            x, y = region.exterior.xy
            axs.plot(x, y, 'r', linewidth=2)

        elif isinstance(region, MultiPolygon):
            for poly in region.geoms:
                x, y = poly.exterior.xy
                axs.plot(x, y, 'r', linewidth=2)

    # Violin Plot (to visually check the result)
    @staticmethod
    def violin_plot(Haz:str, II_event:iris.cube.Cube, II_z500:list, II_slp:list,
                    fig_size:tuple[float, float]=(2.5, 2.5)
                    ) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:

        '''
        Violin plot to visually check the result

        Parameters:
            Haz (str):
                Event variable, either 't2m' or 'tp'
            II_event (iris.cube.Cube):
                Event value plotted as a horizontal reference line
            II_z500 (list):
                Distribution of z500 index values
            II_slp (list):
                Distribution of slp index values
            fig_size (tuple[float, float]):
                Figure size
        
        Returns:
            fig (matplotlib.figure.Figure):
                Figure of the violin plot
            axs (matplotlib.axes.Axes):
                Violin plot
        
        '''

        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=fig_size)

        plots = axs.violinplot([II_z500, II_slp], showmeans=True, showextrema=False, widths = .8)
        plots["bodies"][1].set_facecolor('green')

        axs.axhline(II_event, color='r', label = 'Event')
        axs.set_xticks([1,2], labels=['Z500', 'SLP'])
        axs.tick_params(axis='x', length=0)

        if Haz == 't2m': axs.set_ylabel('Temperature (K)')
        if Haz == 'tp': axs.set_ylabel('Daily Rainfall (mm)')

        t, p = stats.ttest_ind(II_z500, II_slp, equal_var=False, alternative='two-sided')
        if p < 0.05:
            axs.set_title(('%.2f'%t), pad=-20, loc='left', fontweight="bold")
        else:
            axs.set_title(('%.2f'%t), pad=-20, loc='left')

        return fig, axs

    # Shown for larger range of analogue proportions
    @staticmethod
    def plot_analogue_proportions(II_event:iris.cube.Cube, II_z500:list, II_slp:list, N:int,
                                  fig_size:tuple[float,float]=(2.5, 2.5), xlim:int=10
                                  ) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        
        '''
        Show the larger range of analogue proportions as a line graph

        Parameters:
            II_event (iris.cube.Cube):
                Event value plotted as a horizontal reference line
            II_z500 (list):
                Distribution of z500 index values
            II_slp (list):
                Distribution of slp index values
            N (int):
                Number of analogues, used as right horizontal plot limit
            fig_size (tuple[float, float]):
                Figure size
            xlim (int): 
                Left horizontal plot limit

        Returns:
            fig (matplotlib.figure.Figure):
                Figure of the line graph
            axs (matplotlib.axes.Axes):
                Line graph plot
        
        '''

        meanT = []
        for i in np.arange(len(II_z500)):
            meanT.append(np.mean(II_z500[:i]))

        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=fig_size)
        axs.plot(meanT, 'b', label = 'Z500')

        meanT = []
        for i in np.arange(len(II_slp)):
            meanT.append(np.mean(II_slp[:i]))

        axs.plot(meanT, 'g', label = 'SLP')
        axs.set_xlim([xlim, N])
        axs.axhline(II_event, color='r', label = 'Event')
        axs.legend() 
        axs.set_ylabel('Hazard')
        axs.set_xlabel('# of analogues')

        return fig, axs

    # Plot: Analogue variable
    @staticmethod
    def plot_analogue_variable(ana_var:str, event_cube:iris.cube.Cube, selected_daily_cube:iris.cube.Cube,
                               dates_past:list, dates_prst:list, event_date:list, region:list[float]
                               ) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        
        '''
        Plots the analogue variable for the event date, past, and present

        Parameters:
            ana_var (str):
                Analogue variable, either 'z500' or 'slp'
            event_cube (iris.cube.Cube):
                Single day cube of the selected variable, either 'z500' or 'slp'
            selected_daily_cube (iris.cube.Cube):
                Cube of the selected variable, either 'z500' or 'slp', with daily values
            dates_past (list):
                Past dates
            dates_prst (list):
                Present dates
            event_date (list):
                Date of the event
        
        Returns:
            fig (matplotlib.figure.Figure):
                Figure
            ax (matplotlib.axes.Axes):
                plot
        '''


        # EVENT FIELDS
        event_cube = event_cube - event_cube.collapsed(['latitude', 'longitude'], iris.analysis.MEAN)

        # ANALOGUE COMPOSITES
        PAST_comp = Analogues.analogues_composite_anomaly_v2(selected_daily_cube, dates_past)
        PRST_comp = Analogues.analogues_composite_anomaly_v2(selected_daily_cube, dates_prst)

        if ana_var == 'z500':
            PAST_comp = PAST_comp/10
            PRST_comp = PRST_comp/10
            event_cube = event_cube/10

        fig = plt.figure(figsize=(12,3),layout='constrained',dpi=200)

        lats=PRST_comp.coord('latitude').points
        lons=PRST_comp.coord('longitude').points

        x= np.round(np.arange(0, np.max([np.abs(PAST_comp.data), np.abs(PRST_comp.data), np.abs(event_cube.data)]), 2))
        con_lev=np.append(-x[::-1][:-1],x)

        ax= plt.subplot(1,3,1,projection=ccrs.PlateCarree())
        c1 = ax.contourf(lons, lats, event_cube.data, levels=con_lev, cmap="RdBu_r", transform=ccrs.PlateCarree(), extend='both')
        Analogues.plot_region(ax, region)
        cbar = plt.colorbar(c1,fraction=0.046, pad=0.04)
        cbar.ax.tick_params()
        ax.set_title('a) Event, '+str(event_date[2])+event_date[1]+str(event_date[0]), loc='left')
        Analogues.background(ax)

        ax= plt.subplot(1,3,2,projection=ccrs.PlateCarree())
        c1 = ax.contourf(lons, lats, PAST_comp.data, levels=con_lev, cmap="RdBu_r", transform=ccrs.PlateCarree(), extend='both')
        Analogues.plot_region(ax, region)
        cbar = plt.colorbar(c1,fraction=0.046, pad=0.04)
        cbar.ax.tick_params()
        ax.set_title('b) Past Analogues', loc='left')
        Analogues.background(ax)

        ax= plt.subplot(1,3,3,projection=ccrs.PlateCarree())
        c1 = ax.contourf(lons, lats, PRST_comp.data, levels=con_lev, cmap="RdBu_r", transform=ccrs.PlateCarree(), extend='both')
        Analogues.plot_region(ax, region)
        cbar = plt.colorbar(c1,fraction=0.046, pad=0.04)
        cbar.ax.tick_params()
        ax.set_title('c) Present Analogues', loc='left')
        Analogues.background(ax)

        fig.suptitle('Analogue Variable: '+ ana_var)

        return fig, ax

    @staticmethod
    def plot_z500_slp_t2m_tp(ana_var:str, var_list:list[str], cube_map:dict[str, iris.cube.Cube], 
                             R2:list[float], region:list[float], poly_region:list[float], sig_field:list,
                             event_date:list, dates_past:list, dates_prst:list,
                             fig_size:tuple[float, float]=(12,12), dpi:int=200,  parameter:str='Tmean'
                             ) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        
        '''
        Plot event, past analogue composite, present analogue composite,
        and their difference for multiple variables.

        For each variable in ``var_list`` (typically 'z500', 'slp', 't2m', 'tp'),
        four panels are produced:
            1. Event field
            2. Past analogue composite
            3. Present analogue composite
            4. Change (present minus past)

        The analogue variable (``ana_var``) is plotted as spatial anomalies
        with the spatial mean removed.

        Units:
        temperature (t2m): Kelvin;
        precipitation (tp): Meters;
        z500: m^2 s^-2;

        Parameters:
            ana_var (str):
                'z500' or 'slp'
            var_list (list[str]):
                List specifying the order of variables to plot, 'z500', 'slp', 't2m', and 'tp'
            cube_map (dict[str, iris.cube.Cube]):
                Dictionary with daily cube data for 'z500', 'slp', 't2m' and 'tp'
            R2 (list[float]):
                Region used for selecting the data
            region (list[float]):
                Region used for drawing the event
            sig_field (list):
                Single composite of all dates
            event_date (list):
                Date of the event
            dates_past (list):
                Past dates
            dates_prst (list):
                Present dates
            fig_size (tuple[float, float]):
                Figure size
            dpi (int):
                Dots per inch
        
        Returns:
            fig (matplotlib.figure.Figure):
                Figure of the plots
            ax (matplotlib.axes.Axes):
                Plot object

        '''

        # Plot: Z500, SLP, t2m, tp (ToDo: +winds when ERA5 not extended)
        # Z500 plotted as anomalies (to remove the influence of the longterm trend)
        fig = plt.figure(figsize=fig_size, layout='constrained', dpi=dpi)

        # J: replace msl with slp
        for i, var in enumerate(var_list):
            if var == 'z500' or (var == 'msl' or var == 'slp'): CMAP = ["RdBu_r", "RdBu_r"]
            if var == 'tp': CMAP = ["YlGnBu", "BrBG"]
            if var == 't2m': CMAP = ["YlOrRd", "RdYlBu_r"]

            # EVENT FIELDS
            # here we can just extract the date from the cubes we imported earlier
            # event_cube = my.extract_region(my.reanalysis_data_single_date_v2(var, event_date), R2)
            # replace with line below
            # can we just use the cube_map_event_reg?
            event_cube =  Analogues.extract_date_v2(Analogues.extract_region(cube_map[var], R2), event_date)

            if var == ana_var:
                event_cube = event_cube - event_cube.collapsed(['latitude', 'longitude'], iris.analysis.MEAN)
                    # ANALOGUE COMPOSITES
            if var == ana_var:
                PRST_comp = Analogues.analogues_composite_anomaly_v2(Analogues.extract_region(cube_map[var], R2), dates_prst)
                PAST_comp = Analogues.analogues_composite_anomaly_v2(Analogues.extract_region(cube_map[var], R2), dates_past)
            else:
                PRST_comp = Analogues.analogues_composite_v2(Analogues.extract_region(cube_map[var], R2), dates_prst)
                PAST_comp = Analogues.analogues_composite_v2(Analogues.extract_region(cube_map[var], R2), dates_past)


            # Unit conversions
            # ============== Check this because the units should already be correct =============
            if var == 'z500':
                PAST_comp = PAST_comp * 0.0980665 # adjust for gravity?
                PRST_comp = PRST_comp * 0.0980665
                event_cube = event_cube * 0.0980665
            if var == 'msl' or var == 'slp':
                PAST_comp = PAST_comp * .01   # adjust for? mm to cm?
                PRST_comp = PRST_comp * .01
                event_cube = event_cube * .01    
            if var == 't2m':
                PAST_comp = PAST_comp - 273.15    # adjust kelvin to celsius
                PRST_comp = PRST_comp - 273.15
                event_cube = event_cube - 273.15   
            #======================================================================================

            lats=PRST_comp.coord('latitude').points
            lons=PRST_comp.coord('longitude').points 

            if var == 'z500' or (var == 'msl' or var == 'slp'):
                vmin = np.nanmin([
                    np.nanmin(PAST_comp.data),
                    np.nanmin(PRST_comp.data),
                    np.nanmin(event_cube.data),
                ])
                vmax = np.nanmax([
                    np.nanmax(PAST_comp.data),
                    np.nanmax(PRST_comp.data),
                    np.nanmax(event_cube.data),
                ])
                con_lev = np.round(np.arange(vmin, vmax, 2))
            if var == 't2m':
                vmax = np.nanmax([
                    np.nanmax(PAST_comp.data),
                    np.nanmax(PRST_comp.data),
                    np.nanmax(event_cube.data),
                ])
                con_lev = np.round(np.arange(0, vmax, 2))
            if var == 'tp':
                vmax = np.nanmax([
                    np.nanmax(PAST_comp.data),
                    np.nanmax(PRST_comp.data),
                    np.nanmax(event_cube.data),
                ])
                con_lev = np.arange(0, vmax / 2, 0.2)
            # =====================================================================================

            if con_lev.size < 2:
                vmin = np.nanmin(event_cube.data)
                vmax = np.nanmax(event_cube.data)

                if not np.isfinite(vmin) or not np.isfinite(vmax):
                    Utils.print(f"Skipping {var}: invalid data")
                    continue
                
                if vmin == vmax:
                    vmax = vmin + 1e-3

                con_lev = np.linspace(vmin, vmax, 5)
                     
            # Plotting event
            ax= plt.subplot(len(var_list),4,(i*4)+1,projection=ccrs.PlateCarree())
            c1 = ax.contourf(lons, lats, event_cube.data, levels=con_lev, cmap=CMAP[0], transform=ccrs.PlateCarree(), extend='both')
            cbar = plt.colorbar(c1,fraction=0.046, pad=0.04)
            cbar.ax.tick_params()
            ax.set_ylabel(var)
            Analogues.background(ax)
            # Plotting Past Composite
            ax= plt.subplot(len(var_list),4,(i*4)+2,projection=ccrs.PlateCarree())
            c1 = ax.contourf(lons, lats, PAST_comp.data, levels=con_lev, cmap=CMAP[0], transform=ccrs.PlateCarree(), extend='both')
            cbar = plt.colorbar(c1,fraction=0.046, pad=0.04)
            cbar.ax.tick_params()
            Analogues.background(ax)
            # Plotting Present Composite
            ax= plt.subplot(len(var_list),4,(i*4)+3,projection=ccrs.PlateCarree())
            c1 = ax.contourf(lons, lats, PRST_comp.data, levels=con_lev, cmap=CMAP[0], transform=ccrs.PlateCarree(), extend='both')
            cbar = plt.colorbar(c1,fraction=0.046, pad=0.04)
            cbar.ax.tick_params()
            Analogues.background(ax)
            # Plotting Change            
            ax= plt.subplot(len(var_list),4,(i*4)+4,projection=ccrs.PlateCarree())
            Dmax = np.round(np.nanmax(np.abs([np.nanmin((PRST_comp-PAST_comp).data), np.nanmax((PRST_comp-PAST_comp).data)])))
            diff_lev = np.linspace(-Dmax, Dmax, 41)
            c1 = ax.contourf(lons, lats, (PRST_comp-PAST_comp).data, levels=diff_lev, cmap=CMAP[1], transform=ccrs.PlateCarree(), extend='both')

            if sig_field is not None:
                c2 = ax.contourf(
                    lons, lats, sig_field[i].data,
                    levels=[-2, 0, 2],
                    hatches=['////', None],
                    colors='none',
                    transform=ccrs.PlateCarree()
                )
            
            cbar = plt.colorbar(c1,fraction=0.046, pad=0.04)
            cbar.ax.tick_params()
            Analogues.background(ax)
            #fig.suptitle('Analogue Variable: '+ana_var)

        ax= plt.subplot(4,4,1,projection=ccrs.PlateCarree())    
        ax.set_title('a) '+str(event_date[2])+event_date[1]+str(event_date[0])+'  '+var_list[0], loc='left')
        Analogues.plot_box(ax, region)
        Analogues.plot_region(ax, poly_region)
        ax= plt.subplot(4,4,2,projection=ccrs.PlateCarree())  
        ax.set_title('b) Past Analogues', loc='left')
        Analogues.plot_box(ax, region)
        Analogues.plot_region(ax, poly_region)
        ax= plt.subplot(4,4,3,projection=ccrs.PlateCarree())  
        ax.set_title('c) Present Analogues', loc='left')
        Analogues.plot_box(ax, region)
        Analogues.plot_region(ax, poly_region)
        ax= plt.subplot(4,4,4,projection=ccrs.PlateCarree())  
        ax.set_title('d) Change (Present - Past)', loc='left')
        Analogues.plot_box(ax, region)
        Analogues.plot_region(ax, poly_region)

        if parameter == "Precipitation":
            parameter = "Tmean"

        ax=plt.subplot(4,4,5,projection=ccrs.PlateCarree()) 
        ax.set_title('e) '+var_list[1], loc='left')
        Analogues.plot_box(ax, region)
        Analogues.plot_region(ax, poly_region)
        ax=plt.subplot(4,4,6,projection=ccrs.PlateCarree()) 
        ax.set_title('f) ', loc='left')
        Analogues.plot_box(ax, region)
        Analogues.plot_region(ax, poly_region)
        ax=plt.subplot(4,4,7,projection=ccrs.PlateCarree()) 
        ax.set_title('g) ', loc='left')
        Analogues.plot_box(ax, region)
        Analogues.plot_region(ax, poly_region)
        ax=plt.subplot(4,4,8,projection=ccrs.PlateCarree()) 
        ax.set_title('h) ', loc='left')
        Analogues.plot_box(ax, region)
        Analogues.plot_region(ax, poly_region)

        ax=plt.subplot(4,4,9,projection=ccrs.PlateCarree()) 
        ax.set_title('i) '+var_list[2], loc='left')
        Analogues.plot_box(ax, region)
        Analogues.plot_region(ax, poly_region)
        ax=plt.subplot(4,4,10,projection=ccrs.PlateCarree()) 
        ax.set_title('j) ', loc='left')
        Analogues.plot_box(ax, region)
        Analogues.plot_region(ax, poly_region)
        ax=plt.subplot(4,4,11,projection=ccrs.PlateCarree()) 
        ax.set_title('k) ', loc='left')
        Analogues.plot_box(ax, region)
        Analogues.plot_region(ax, poly_region)
        ax=plt.subplot(4,4,12,projection=ccrs.PlateCarree()) 
        ax.set_title('l) ', loc='left')
        Analogues.plot_box(ax, region)
        Analogues.plot_region(ax, poly_region)

        ax=plt.subplot(4,4,13,projection=ccrs.PlateCarree()) 
        ax.set_title('m) '+parameter, loc='left')
        Analogues.plot_box(ax, region)
        Analogues.plot_region(ax, poly_region)
        ax=plt.subplot(4,4,14,projection=ccrs.PlateCarree()) 
        ax.set_title('n) ', loc='left')
        Analogues.plot_box(ax, region)
        Analogues.plot_region(ax, poly_region)
        ax=plt.subplot(4,4,15,projection=ccrs.PlateCarree()) 
        ax.set_title('o) ', loc='left')
        Analogues.plot_box(ax, region)
        Analogues.plot_region(ax, poly_region)
        ax=plt.subplot(4,4,16,projection=ccrs.PlateCarree()) 
        ax.set_title('p) ', loc='left')
        Analogues.plot_box(ax, region)
        Analogues.plot_region(ax, poly_region)

        return fig, ax


    @staticmethod
    def plot_frequency_timeseries(yr_vals:list, roll_vals:list, Y1:int, Y2:int,
                                  fig_size:tuple[float,float]=(8,4)
                                  ) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes, list, list]:
        
        '''
        Plot annual frequency time series of analogue occurrence together with
        linear trends and 10-year rolling means.

        Three percentile thresholds are shown: upper 5%, 10%, and 20%.
        
        Parameters:
            yr_vals (list):
                List containing three yearly time series
                [upper 5%, upper 10%, upper 20%], each spanning years Y1 to Y2.
            roll_vals (list):
                List containing three 10-year rolling-mean time series corresponding
                to yr_vals [upper 5%, upper 10%, upper 20%].
            Y1 (int):
                Start year for analysis range
            Y2 (int):
                Eng year for analysis range
            fig_size (tuple[float, float]):
                Figure size
        
        Returns:
            fig (matplotlib.figure.Figure):
                Figure
            ax (matplotlib.axes.Axes):
                Plot
            list:
                List of slope values for upper 5%, 10% and 20%
            list:
                List of p values for upper 5%, 10% and 20%

        '''

        # Plot timeseries with linear trends and 10-year rolling means
        fig, ax = plt.subplots(1, 1, figsize = fig_size)

        # linear trends
        trend = np.polyfit(np.arange(Y1, Y2), yr_vals[1] ,1)
        trendpoly = np.poly1d(trend) 
        slope10, intercept, r_value, pval_10, std_err = stats.linregress(np.arange(Y1, Y2), yr_vals[1])
        ax.plot(np.arange(Y1, Y2), trendpoly(np.arange(Y1, Y2)), color='orange', linewidth=.5)
        text10 = 'Upper 10%. trend: '+"%.2f" % round(slope10, 2)+', pval: '+"%.2f" % round(pval_10, 2)

        trend = np.polyfit(np.arange(Y1, Y2), yr_vals[0],1)
        trendpoly = np.poly1d(trend) 
        slope5, intercept, r_value, pval_5, std_err = stats.linregress(np.arange(Y1, Y2), yr_vals[0])
        ax.plot(np.arange(Y1, Y2), trendpoly(np.arange(Y1, Y2)), color='r', linewidth=.5)
        text5 = 'Upper 5%. trend: '+"%.2f" % round(slope5, 2)+', pval: '+"%.2f" % round(pval_5, 2)

        trend = np.polyfit(np.arange(Y1, Y2), yr_vals[2],1)
        trendpoly = np.poly1d(trend) 
        slope20, intercept, r_value, pval_20, std_err = stats.linregress(np.arange(Y1, Y2), yr_vals[2])
        ax.plot(np.arange(Y1, Y2), trendpoly(np.arange(Y1, Y2)), color='b', linewidth=.5)
        text20 = 'Upper 20%. trend: '+"%.2f" % round(slope20, 2)+', pval: '+"%.2f" % round(pval_20, 2)

        ax.plot(np.arange(Y1+5, Y2-5), roll_vals[0], 'r:')
        ax.plot(np.arange(Y1+5, Y2-5), roll_vals[1], color='orange', linestyle=':')
        ax.plot(np.arange(Y1+5, Y2-5), roll_vals[2], 'b:')

        ax.plot(np.arange(Y1, Y2), yr_vals[0], 'r', label=text5)
        ax.plot(np.arange(Y1, Y2), yr_vals[1], color='orange', label=text10)
        ax.plot(np.arange(Y1, Y2), yr_vals[2], 'b', label=text20)

        ax.set_xlabel('Year')
        ax.set_ylabel("Similar days per year")

        ax.set_xlim([Y1, Y2])
        ax.legend(loc=2)

        return fig, ax, [slope5, slope10, slope20], [pval_5, pval_10, pval_20]

    @staticmethod
    def plot_postage_stamps(ana_var:str, haz_var:str, cube_map:dict[str, iris.cube.Cube], event_date:list,
                            region:list, cmap:ListedColormap, dates_plot:list,
                            circ_past:iris.cube.CubeList, haz_past:iris.cube.CubeList,
                            circ_plot:int, n:int, fig_size:tuple[float, float]=(12,12), poly_region:list[float]=None
                            ) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        
        '''
        Plot postage-stamp maps of an event and its analogue days for a circulation
        variable and an associated hazard variable.

        The first panel shows the event day, followed by panels showing analogue
        days. Hazard fields are shown as filled contours, with optional circulation
        contours overlaid.

        Parameters:
            ana_var (str):
                Either 'z500' or 'slp'
            haz_var (str):
                Either 't2m' or 'tp'
            cube_map (dict[str, iris.cube.Cube]):
                Dictionary containing cubes with daily values for 'z500', 'slp', 't2m', and 'tp'
            event_date (list):
                Date of the event
            region (list):
                Region selection for the plot
            cmap (ListedColormap):
                Colormap
            dates_plot (list):
                Dates to plot
            circ_past (iris.cube.CubeList):
                CubeList containing circulation fields for analogue dates
            haz_past (iris.cube.CubeList):
                CubeList containing hazard fields for analogue dates.
            circ_plot (int):
                Flag indicating whether circulation contours are overlaid (1 = plot contours, 0 = no contours).
            n (int):
                Number of analogue dates to plot.
            fig_size (tuple[float, float]):
                Figure size
        
        Returns:
            fig (matplotlib.figure.Figure):
                Figure
            ax (matplotlib.axes.Axes):
                Plot            
        
        '''

        # Past dates plot
        lats=circ_past[0].coord('latitude').points
        lons=circ_past[0].coord('longitude').points
        fig, axs = plt.subplots(nrows=(int(np.ceil((n+1)/5))),
                                ncols=5,
                                subplot_kw={'projection': ccrs.PlateCarree()},
                                figsize=fig_size)

        circ = Analogues.extract_region(Analogues.extract_date_v2(cube_map[ana_var], event_date), region)
        haz = Analogues.extract_region(Analogues.extract_date_v2(cube_map[haz_var], event_date), region)

        if haz_var == 'tp':
            c = axs[0,0].contourf(lons, lats, haz.data,
                                  levels=np.linspace(1, 80, 9),
                                  cmap = cmap,
                                  transform=ccrs.PlateCarree(),
                                  extend='max')
            Analogues.plot_region(axs[0,0], poly_region)
            
            fig.subplots_adjust(right=0.8, hspace=-.2)
            cbar_ax = fig.add_axes([0.81, 0.4, 0.01, 0.2])
            fig.colorbar(c, cax=cbar_ax, ticks=np.arange(0, 100, 10))
            cbar_ax.set_ylabel('Total Precipitation (mm)', labelpad=10, rotation=270, fontsize=10)
            cbar_ax.set_yticklabels(['0', '', '20','','40','','60','','80',''])
        elif haz_var == 't2m':
            c = axs[0,0].contourf(lons, lats, haz.data-273.15,
                                  levels=np.linspace(np.min(haz.data-273.15),np.max(haz.data-273.15), 9),
                                  cmap = plt.cm.get_cmap('RdBu_r'),
                                  transform=ccrs.PlateCarree(),
                                  extend='max')
            Analogues.plot_region(axs[0,0], poly_region)

        if circ_plot == 1:
            c2 = axs[0,0].contour(lons, lats, circ.data/100,
                                  colors='k', transform=ccrs.PlateCarree(),
                                  extend='both')
            axs[0,0].clabel(c2, inline=1, fontsize=12)

        axs[0,0].add_feature(cfeature.BORDERS, color='grey')
        axs[0,0].add_feature(cfeature.COASTLINE, color='grey')
        axs[0,0].set_title('Event: '+str(event_date[2])+event_date[1]+str(event_date[0]), loc='left',
                           fontsize=8)

        for i, ax in enumerate(np.ravel(axs)[1:n+1]):
            if haz_var == 'tp':
                c = ax.contourf(lons, lats, haz_past[i].data,
                                levels=np.linspace(1, 80, 9),
                                cmap = cmap,
                                transform=ccrs.PlateCarree(),
                                extend='max')
                Analogues.plot_region(ax, poly_region)
            elif haz_var == 't2m':
                c = ax.contourf(lons, lats, haz_past[i].data-273.15,
                                levels=np.linspace(np.min(haz_past[i].data-273.15), np.max(haz_past[i].data-273.15), 9),
                                cmap = plt.cm.get_cmap('RdBu_r'),
                                transform=ccrs.PlateCarree(),
                                extend='max')
                Analogues.plot_region(ax, poly_region)

            if circ_plot == 1:
                c2 = ax.contour(lons, lats, circ_past[i].data/100,
                                colors='k', transform=ccrs.PlateCarree(),
                                extend='both')
                ax.clabel(c2, inline=1, fontsize=12)

            ax.add_feature(cfeature.BORDERS, color='grey')
            ax.add_feature(cfeature.COASTLINE, color='grey')
            ax.set_title('Analogue: '+str(dates_plot[i][-2:])+calendar.month_abbr[int(dates_plot[i][4:-2])]+str(dates_plot[i][:4]), loc='left',
                         fontsize=8)

        return fig, axs


    # Util
    ######################################################################################

    # guess bounds
    @staticmethod
    def guess_bounds(cube:iris.cube.Cube, lat:str='latitude', lon:str='longitude') -> iris.cube.Cube:

        '''
        Guess the bounds for an iris cube when no bounds are present

        Parameters:
            cube (iris.cube.Cube):
                Iris cube
            lat (str):
                Latitude column
            lon (str):
                Longitude column

        Returns:
            cube (iris.cube.Cube):
                Iris cube with bounds
        '''

        if not cube.coord(lat).has_bounds():
            cube.coord(lat).guess_bounds()
        if not cube.coord(lon).has_bounds():
            cube.coord(lon).guess_bounds()

        return cube

    @staticmethod
    def remove_bounds(cube:iris.cube.Cube, lat:str='latitude', lon:str='longitude') -> iris.cube.Cube:

        '''
        Remove the bounds for an iris cube when bounds are present

        Parameters:
            cube (iris.cube.Cube):
                Iris cube
            lat (str):
                Latitude column
            lon (str):
                Longitude column

        Returns:
            cube (iris.cube.Cube):
                Iris cube without bounds
        '''

        if cube.coord(lat).has_bounds():
            cube.coord(lat).bounds = None
        if cube.coord(lon).has_bounds():
            cube.coord(lon).bounds = None

        return cube

    # get the ana_van and haz_var values for a range of single dates
    @staticmethod
    def ana_and_haz_date_values(ana_var:str, haz_var:str, cube_map:dict[str, iris.cube.Cube],
                                region:list[float], dates:list
                                ) -> tuple[iris.cube.CubeList,iris.cube.CubeList , int]:

        '''
        Extract circulation and hazard variable fields for a list of single dates.

        For each date in ``dates``, the function extracts the spatial subset
        (defined by ``region``) of the circulation variable ``ana_var`` and
        the hazard variable ``haz_var`` from the provided daily cubes.

        Parameters:
            ana_var (str):
                Either 'z500' or 'slp'
            haz_var (str):
                Either 't2m' or 'tp'
            cube_map (dict[str, iris.cube.Cube]):
                Dictionary containing iris cubes with daily values for 'z500', 'slp', 't2m', and 'tp'
            region (list[float]):
                Region to subset cubes on
            dates (list):
                List of dates
        
        Returns:
            circ_vals (iris.cube.CubeList):
                CubeList containing the circulation variable values for each date, subset to the specified region.
            haz_vals (iris.cube.CubeList):
                CubeList containing the hazard variable values for each date, subset to the specified region.
            n (int):
                Number of dates
        '''

        # Prst date fields
        circ_vals = iris.cube.CubeList([])
        haz_vals = iris.cube.CubeList([])

        n = len(dates)
        for each in range(n):
            year = int(dates[each][:4])
            month = calendar.month_abbr[int(dates[each][4:-2])]
            day = int(dates[each][-2:])

            # Utils.print(ana_var, [year, month, day])
            circ_vals.append(Analogues.extract_region(Analogues.extract_date_v2(cube_map[ana_var], [year, month, day]), region))
            # Utils.print(haz_var, [year, month, day])
            haz_vals.append(Analogues.extract_region(Analogues.extract_date_v2(cube_map[haz_var], [year, month, day]), region))

        return circ_vals, haz_vals, n

    # values for variable choice
    @staticmethod
    def variable_choice_values(cube:iris.cube.Cube, event_date:list,
                               dates_z500:list, dates_slp:list
                               ) -> tuple[iris.cube.Cube, list, list]:
        '''
        Extract impact index values for a single event and for analogue dates 
        corresponding to z500 and slp.

        The function calculates the impact index for the event day and for
        the list of analogue days associated with circulation variables z500
        and slp, returning them as processed values.

        Parameters:
            cube (iris.cube.Cube):
                Iris cube containing daily values, either 't2m' or 'tp'
            event_date (list):
                Date of the event
            dates_z500 (list):
                z500 dates
            dates_slp (list):
                slp dates

        Returns:
            II_event (iris.cube.Cube):
                Impact index value of the event day (spatially averaged cube).
            II_z500 (list):
                List of impact index values for each z500 analogue date.
            II_slp (list):
                List of impact index values for each slp analogue date.
        '''

        II_event = Analogues.impact_index_v2(Analogues.extract_date_v2(cube, event_date))

        II_z500 = []
        daily_analogues = Analogues.analogues_list(cube, dates_z500)
        for each in daily_analogues:
            II_z500.append(Analogues.impact_index_v2(each))

        II_slp = []
        daily_analogues = Analogues.analogues_list(cube, dates_slp)
        for each in daily_analogues:
            II_slp.append(Analogues.impact_index_v2(each))

        return II_event, II_z500, II_slp


    # excess
    ######################################################################################

    # J: this function is never used but was in one of the analogues
    # J: to be removed?
    # @staticmethod
    # def plot_specified_date(axs, fig, ax, date, title, ana_var, haz_var, R1): 
    #     circ = Analogues.extract_region(Analogues.reanalysis_data_single_date(ana_var, date), R1)
    #     #E = E - E.collapsed(['latitude', 'longitude'], iris.analysis.MEAN)
    #     haz = Analogues.extract_region(Analogues.reanalysis_data_single_date(haz_var, date), R1)
    #     lats=haz.coord('latitude').points
    #     lons=haz.coord('longitude').points
    #     if haz_var == 'tp':
    #         c = ax.contourf(lons, lats, haz.data, levels=np.linspace(1, 80, 9), cmap = plt.cm.get_cmap('Blues'), transform=ccrs.PlateCarree(), extend='max')
    #         fig.subplots_adjust(right=0.8)
    #         cbar_ax = fig.add_axes([0.85, 0.3, 0.02, 0.4])
    #         fig.colorbar(c, cax=cbar_ax, ticks=np.arange(0, 100, 10))
    #         cbar_ax.set_ylabel('Total Precipitation (mm)', labelpad=10, rotation=270, fontsize=12)
    #         cbar_ax.set_yticklabels(['0', '', '20','','40','','60','','80',''])
    #     elif haz_var == 't2m':
    #         c = ax.contourf(lons, lats, haz.data-273.15, levels=np.linspace(np.min(haz.data-273.15), np.max(haz.data-273.15), 9), cmap = plt.cm.get_cmap('RdBu_r'), transform=ccrs.PlateCarree(), extend='max')
    #         fig.subplots_adjust(right=0.8)
    #         cbar_ax = fig.add_axes([0.85, 0.3, 0.02, 0.4])
    #         fig.colorbar(c, cax=cbar_ax, ticks=np.arange(np.min(haz.data-273.15), np.max(haz.data-273.15), 5))
    #         cbar_ax.set_ylabel('t2m', labelpad=10, rotation=270, fontsize=12)
    #         #cbar_ax.set_yticklabels(['0', '', '20','','40','','60','','80',''])
    #     lats=circ.coord('latitude').points
    #     lons=circ.coord('longitude').points
    #     c2 = axs.contour(lons, lats, circ.data/100, colors='k', transform=ccrs.PlateCarree(), extend='both')
    #     axs.clabel(c2, inline=1, fontsize=12)
    #     axs.add_feature(cfeature.BORDERS)
    #     axs.add_feature(cfeature.COASTLINE)
    #     axs.set_title(title, loc='left')