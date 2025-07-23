"""Helper classes that allow for a declarative specification of how to map from iris cubes to xarray DataArrays
"""
import xarray as xr

from util import model_level_to_pressure


class DataArrayExtractor:
    """Extract a cube or DataArray from a set of input cubes."""
    def __init__(self, p, z):
        self.p = p
        self.z = z

    @staticmethod
    def extract_cube(map_item, group_cubes, combine_cubes=True):
        """Extract an individual cube from a set of input cubes, possibly combining multiple cubes.

        Uses the constraints in map_item to extract the desired cube."""
        # Extract the cube and combine if necessary.
        if isinstance(map_item, MapItem):
            cube = group_cubes.extract_cube(map_item.iris_constraint)
        elif isinstance(map_item, MultiMapItem):
            constraint_cubes = [group_cubes.extract_cube(map_item.iris_constraint) for map_item in map_item.items]
            cube = constraint_cubes[0]
            if combine_cubes:
                for next_cube, op in zip(constraint_cubes[1:], map_item.ops):
                    cube.data = op(cube.data, next_cube.data)
        else:
            raise Exception(f'unknown type {type(map_item)}')

        return cube

    def extract_da(self, map_item, group_cubes):
        """Extracts a DataArray from a map item and group of cubes, applying optional extra processing if specified.
        """
        cube = self.extract_cube(map_item, group_cubes)
        if map_item.extra_processing is not None:
            if map_item.extra_processing == 'interpolate_model_levels_to_pressure':
                # Not so easy to add this as an extra processing step because it needs p and z.
                cube = model_level_to_pressure(cube, self.p, self.z)
            else:
                # This might be e.g. flipping the sign of some fields.
                cube = map_item.extra_processing(cube)

        # For some cubes (ones with names like m01s30i461 the da gets a name like filled-XXXXXX...
        # Make sure it's got the actual cube name so I can rename it later.
        da = xr.DataArray.from_iris(cube).rename(cube.name())
        da.attrs.update(map_item.extra_attrs)
        return da


class MultiMapItem:
    """Contains several MapItems and the rules for combining these (in ops)."""
    def __init__(self, items, ops, extra_processing=None, extra_attrs=None):
        self.items = items
        self.ops = ops
        self.extra_processing = extra_processing
        self.extra_attrs = extra_attrs if extra_attrs is not None else {}

    def __repr__(self):
        return 'MutliMapItem(' + str(self.items) + ', ' + str(self.ops) + ')'



class MapItem:
    """Contains iris constraints to extract a given cube, along with extra_processing."""
    def __init__(self, iris_constraint, extra_processing=None, extra_attrs=None):
        self.iris_constraint = iris_constraint
        self.extra_processing = extra_processing
        self.extra_attrs = extra_attrs if extra_attrs is not None else {}

    def __repr__(self):
        return f'MapItem({self.iris_constraint})'