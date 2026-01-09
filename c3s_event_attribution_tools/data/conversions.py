import numpy as np
import pandas as pd
from typing import TypeVar

T = TypeVar("T", np.ndarray, pd.Series)

def identity(x: T) -> T:
    return x

class Conversions:
    @staticmethod
    def convert_temperature(array: T, from_unit: str, to_unit: str) -> T:
        """
        Convert temperature values from one unit to another.

        This method performs in-place temperature conversion on the provided array or
        pandas Series and supports Kelvin (k), Celsius (c), and Fahrenheit (f).

        Parameters:
            array (np.ndarray or pd.Series):
                The input array or Series containing temperature values to be converted.
            from_unit (str, optional):
                Source temperature unit ('k', 'c', or 'f'). Default is 'k'.
            to_unit (str, optional):
                Target temperature unit ('k', 'c', or 'f'). Default is 'c'.

        Returns:
            np.ndarray or pd.Series: The array or Series with temperature values converted to the
            target unit.

        Raises:
            ValueError: If from_unit is not one of 'k', 'c', or 'f'.
            TypeError: If the input array is neither a numpy ndarray nor a pandas Series.
            
        Notes:
            - If from_unit and to_unit are identical, no conversion is performed.
            - Conversion modifies the input array or Series in place.
            - Conversion formulas:
                - Kelvin to Celsius: C = K - 273.15
                - Kelvin to Fahrenheit: F = (K - 273.15) × 9/5 + 32
                - Celsius to Kelvin: K = C + 273.15
                - Celsius to Fahrenheit: F = C × 9/5 + 32
                - Fahrenheit to Kelvin: K = (F - 32) × 5/9 + 273.15
                - Fahrenheit to Celsius: C = (F - 32) × 5/9
        """
        if isinstance(array, np.ndarray):
            return Conversions._inner_convert_temperature(array, from_unit, to_unit)
        elif isinstance(array, pd.Series):
            pandas_np_array : np.ndarray = array.values # type: ignore
            return pd.Series(Conversions._inner_convert_temperature(pandas_np_array, from_unit, to_unit))
        else:
            raise TypeError("Unsupported type for temperature conversion: " + str(type(array)))
        
    @staticmethod
    def _inner_convert_temperature(array: np.ndarray, from_unit: str, to_unit: str) -> np.ndarray:
        """
        This static helper method handles the mathematical transformation between 
        Kelvin, Celsius, and Fahrenheit. It ensures consistency by normalizing 
        unit strings to lowercase before processing.

        Parameters:
            array (np.ndarray):
                The numeric array containing temperature values to be converted.
            from_unit (str):
                The current unit of the data. Supported values: "k", "c", "f".
            to_unit (str):
                The target unit for the data. Supported values: "k", "c", "f".

        Returns:
            np.ndarray: A new array containing the converted temperature values.

        Raises:
            ValueError: If an unsupported unit or conversion pair is provided.
        """
        if from_unit == to_unit:
            return array
        # Convert unit lowercase for consistency
        from_unit = from_unit.lower()
        to_unit = to_unit.lower()
        
        if from_unit == "k" and to_unit == "c":
            return array - 273.15
        elif from_unit == "c" and to_unit == "k":
            return array + 273.15
        elif from_unit == "c" and to_unit == "f":
            return (array * 9/5) + 32
        elif from_unit == "f" and to_unit == "c":
            return (array - 32) * 5/9
        elif from_unit == "k" and to_unit == "f":
            return (array - 273.15) * 9/5 + 32
        elif from_unit == "f" and to_unit == "k":
            return (array - 32) * 5/9 + 273.15
        else:
            raise ValueError(f"Unsupported temperature conversion from {from_unit} to {to_unit}")
        
        
    @staticmethod
    def convert_bbox_to_0_360(bbox: tuple[float,float,float,float], eps=1e-9) -> list[tuple[float,float,float,float]]:
        """
        Convert a bounding box from [-180, 180] longitude to [0, 360] longitude.

        If the converted box crosses 0°, return two wrapped bounding boxes.

        Parameters:
            bbox (tuple):
                (min_lon, min_lat, max_lon, max_lat) with longitude in [-180, 180].
            eps (float):
                Small tolerance value.
            as_shapely (bool):
                If True, return shapely boxes (requires shapely).

        Returns:
            list: List of bounding boxes in
            (min_lon_360, min_lat, max_lon_360, max_lat) format or shapely boxes.
        """
        min_lon, min_lat, max_lon, max_lat = bbox

        # Handle antimeridian-style input like (170, ..., -170, ...) meaning it crosses -180/180 already
        if min_lon > max_lon:
            max_lon += 360.0

        # Full-world guard
        if (max_lon - min_lon) >= 360.0 - eps:
            return [(0.0, min_lat, 360.0, max_lat)]
        else:
            to360 = lambda x: x % 360.0
            lo = to360(min_lon)
            hi = to360(max_lon)

            if lo <= hi:  # no wrap in 0–360 space
                out = [(lo, min_lat, hi, max_lat)]
            else:
                # wraps across 0° in 0–360; split into two
                out = [
                    (lo,  min_lat, 360.0, max_lat),
                    (0.0, min_lat, hi,    max_lat),
                ]
            return out
        
    @staticmethod
    def variable_name_translation(variable_name: str) -> str:
        """
        Translate variable names between different conventions.

        Parameters:
            variable_name (str):
                The variable name to translate.

        Returns:
            str: The translated variable name.
        """
        translation_dict = {
            '2m_temperature': 't2m',
            'total_precipitation': 'tp',
        }
        return translation_dict.get(variable_name, variable_name)