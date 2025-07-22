"""Configuration used to control all aspects of regridding to saving files.

creates a processing_config variable that has a key for each experiment."""
import operator
import sys
from pathlib import Path

import iris
import iris.cube
import pandas as pd

from cube_to_da_mapping import MapItem, MultiMapItem

output_vn = 'v5.2'
deploy = 'dev'

# Defaults - can be overridden.
slurm_config = dict(
    account='hrcm',
    ntasks=1,
    cpus_per_task=1,
    partition='standard',
    qos='standard',
    mem=100000,
    nconcurrent_tasks=40,
)

shared_metadata = {
    'Met Office DYAMOND3 simulations': (
        'A group of experiments have been conducted using the Met Office Unified Model (MetUM) with a focus on the '
        'DYAMOND-3 period (Jan 2020-Feb 2021). While this experiments include standalone explicit convection global '
        'simulations we have also developed a cyclic tropical channel and include limited area model simulations to '
        'build our understanding of how resolving smaller-scale processes feeds back on to the large-scale '
        'atmospheric circulation.'),
}


def has_dimensions(*dims):
    """Returns an Iris constraint that filters cubes based on dimensions."""

    def dim_filter(cube):
        cube_dims = tuple([c.name() for c in cube.dim_coords])
        return cube_dims == dims

    return iris.Constraint(cube_func=dim_filter)


def cube_cell_method_is_not_empty(cube):
    return cube.cell_methods != tuple()


def cube_cell_method_is_empty(cube):
    return cube.cell_methods == tuple()


def invert_cube_sign(cube):
    cube.data = -1 * cube.data
    return cube


name_map_2d = {
    ('psl', 'air_pressure_at_mean_sea_level'): MapItem('air_pressure_at_sea_level'),
    ('tas', 'air_temperature'): MapItem('air_temperature'),
    ('psl', 'air_pressure_at_mean_sea_level'): MapItem('air_pressure_at_sea_level'),
    ('tas', 'air_temperature'): MapItem('air_temperature'),
    ('clwvi', 'atmosphere_mass_content_of_cloud_condensed_water'): MapItem('atmosphere_cloud_liquid_water_content'),
    ('clivi', 'atmosphere_mass_content_of_cloud_ice'): MapItem('atmosphere_cloud_ice_content'),
    ('prw', 'atmosphere_mass_content_of_water_vapor'): MapItem('m01s30i461'),
    ('clt', 'cloud_area_fraction'): MapItem('cloud_area_fraction_assuming_maximum_random_overlap'),
    ('uas', 'eastward_wind'): MapItem('x_wind'),
    ('vas', 'northward_wind'): MapItem('y_wind'),
    ('pr', 'precipitation_flux'): MultiMapItem(
        [MapItem(iris.Constraint(name='stratiform_rainfall_flux') & iris.Constraint(
            cube_func=cube_cell_method_is_not_empty)),
         MapItem(iris.Constraint(name='stratiform_snowfall_flux') & iris.Constraint(
             cube_func=cube_cell_method_is_not_empty))],
        ops=[operator.add],
        extra_attrs={
            'notes': 'Combined rain and snow. Hourly mean - time index shifted from half past the hour to the following hour'},
    ),
    ('prs', 'solid_precipitation_flux'): MapItem(
        iris.Constraint(name='stratiform_snowfall_flux') & iris.Constraint(
            cube_func=cube_cell_method_is_not_empty),
        extra_attrs={'notes': 'Hourly mean - time index shifted from half past the hour to the following hour'},
    ),
    ('huss', 'specific_humidity'): MapItem('specific_humidity'),
    ('ps', 'surface_air_pressure'): MapItem('surface_air_pressure'),
    ('hflsd', 'surface_downward_latent_heat_flux'): MapItem('surface_upward_latent_heat_flux',
                                                            extra_processing=invert_cube_sign),
    ('hfssd', 'surface_downward_sensible_heat_flux'): MapItem('surface_upward_sensible_heat_flux',
                                                              extra_processing=invert_cube_sign),
    ('rlds', 'surface_downwelling_longwave_flux_in_air'): MapItem('surface_downwelling_longwave_flux_in_air'),
    ('rldscs', 'surface_downwelling_longwave_flux_in_air_clear_sky'): MapItem(
        'surface_downwelling_longwave_flux_in_air_assuming_clear_sky'),
    ('rsds', 'surface_downwelling_shortwave_flux_in_air'): MapItem('surface_downwelling_shortwave_flux_in_air'),
    ('rsdscs', 'surface_downwelling_shortwave_flux_in_air_clear_sky'): MapItem(
        'surface_downwelling_shortwave_flux_in_air_assuming_clear_sky'),
    ('ts', 'surface_temperature'): MapItem('surface_temperature'),
    ('rsdt', 'toa_incoming_shortwave_flux'): MapItem('toa_incoming_shortwave_flux'),
    ('rlut', 'toa_outgoing_longwave_flux'): MapItem('toa_outgoing_longwave_flux'),
    ('rlutcs', 'toa_outgoing_longwave_flux_clear_sky'): MapItem('toa_outgoing_longwave_flux_assuming_clear_sky'),
    ('rsut', 'toa_outgoing_shortwave_flux'): MapItem(iris.Constraint(
        name='toa_outgoing_shortwave_flux') & iris.AttributeConstraint(
        STASH='m01s01i208')),
    ('rsutcs', 'toa_outgoing_shortwave_flux_clear_sky'): MapItem('toa_outgoing_shortwave_flux_assuming_clear_sky'),
    ('rsus', 'surface_upwelling_shortwave_flux_in_air'): MapItem('m01s01i202', extra_processing=invert_cube_sign()),
    ('rlus', 'surface_upwelling_longwave_flux_in_air'): MapItem('surface_net_downward_longwave_flux', extra_processing=invert_cube_sign()),
}

name_map_3d = {
    ('ua', 'eastward_wind'): MapItem('x_wind'),
    ('zg', 'geopotential height'): MapItem('geopotential_height'),
    ('va', 'northtward_wind'): MapItem('y_wind'),
    ('hur', 'relative_humidity'): MapItem(iris.Constraint(
        name='relative_humidity') & iris.AttributeConstraint(
        STASH='m01s30i206')),
    ('hus', 'specific_humidity'): MapItem('specific_humidity'),
    ('ta', 'temperature'): MapItem('air_temperature'),
    ('wa', 'upward_air_velocity'): MapItem('upward_air_velocity'),
}

name_map_3d_ml = {
    ('cli', 'mass_fraction_of_cloud_ice_in_air'): MapItem('mass_fraction_of_cloud_ice_in_air',
                                                          extra_processing='interpolate_model_levels_to_pressure'),
    ('clw', 'mass_fraction_of_cloud_liquid_water_in_air'): MapItem('mass_fraction_of_cloud_liquid_water_in_air',
                                                                   extra_processing='interpolate_model_levels_to_pressure'),
    ('qg', 'mass_fraction_of_graupel_in_air'): MapItem('mass_fraction_of_graupel_in_air',
                                                       extra_processing='interpolate_model_levels_to_pressure'),
    ('qr', 'mass_fraction_of_rain_in_air'): MapItem('mass_fraction_of_rain_in_air',
                                                    extra_processing='interpolate_model_levels_to_pressure'),
    ('qs', 'mass_fraction_of_snow_water_in_air'): MapItem('mass_fraction_of_cloud_ice_crystals_in_air',
                                                          extra_processing='interpolate_model_levels_to_pressure'),
}

# Aim for chunk sizes between 1-10MB.
chunks2d = {
    # z10 and z9 have to have no chunking over time. (z9 is the highest zoom for N1280.)
    10: (1, 4 ** 10),  # 12 chunks per time.
    9: (1, 4 ** 9),
    # Increase temporal at same rate as reducing spatial.
    8: (4 ** 2, 4 ** 8),
    7: (4 ** 3, 4 ** 7),
    # Transition to 3 chunks per time.
    6: (4 ** 3, 4 ** 7),
    # Transition to 1 chunk per time.
    5: (4 ** 3, 12 * 4 ** 5),
    4: (4 ** 4, 12 * 4 ** 4),
    3: (4 ** 5, 12 * 4 ** 3),
    2: (4 ** 6, 12 * 4 ** 2),
    1: (4 ** 7, 12 * 4 ** 1),
    0: (4 ** 8, 12 * 4 ** 0),
}

# Much better for regional arrays if these have the same spatial chunks as 2d
chunks3d = {
    z: (t, 1, s)
    for z, (t, s) in chunks2d.items()
}

# Aim for chunk sizes between 1-10MB.
chunks2dregional = {
    # z10 has to have no chunking over time.
    10: (1, 4 ** 9),  # 12 x 4 = 48 chunks per time.
    # Increase temporal at same rate as reducing spatial.
    9: (4, 4 ** 9),
    8: (4 ** 2, 4 ** 8),
    7: (4 ** 3, 4 ** 7),
    # Transition to 3 chunks per time.
    6: (4 ** 3, 4 ** 7),
    # Transition to 1 chunk per time.
    5: (4 ** 3, 12 * 4 ** 5),
    4: (4 ** 4, 12 * 4 ** 4),
    3: (4 ** 5, 12 * 4 ** 3),
    2: (4 ** 6, 12 * 4 ** 2),
    1: (4 ** 7, 12 * 4 ** 1),
    0: (4 ** 8, 12 * 4 ** 0),
}

# Much better for regional arrays if these have the same spatial chunks as 2d
chunks3dregional = {
    z: (t, 1, s)
    for z, (t, s) in chunks2dregional.items()
}

# Orig:
# switch to (X, 1, Y), where X and Y are same as 2d.
# chunks3d = {
#     # z10 has to have no chunking over time.
#     10: (1, 5, 4 ** 9),  # 48 chunks per time.
#     9: (4, 5, 4 ** 8),
#     8: (4 ** 2, 5, 4 ** 7),
#     7: (4 ** 3, 5, 4 ** 6),
#     # transition to fewer chunks per time.
#     6: (4 ** 3, 5, 4 ** 6),  # 12 chunks per time
#     5: (4 ** 3, 5, 12 * 4 ** 4),
#     # transition to 1 chunk per time.
#     4: (4 ** 3, 5, 12 * 4 ** 4),
#     3: (4 ** 4, 5, 12 * 4 ** 3),
#     # increase number pressure levels.
#     2: (4 ** 4, 25, 12 * 4 ** 2),
#     1: (4 ** 5, 25, 12 * 4 ** 1),
#     0: (4 ** 6, 25, 12 * 4 ** 0),
# }

# UM vars in the original .pp files/iris cubes that are not needed.
drop_vars = [
    'latitude_0',
    'longitude_0',
    # 'bnds',  # gets dropped automatically and causes problems if you try to drop it.
    'forecast_period',
    'forecast_reference_time',
    'forecast_period_0',
    'height',
    'height_0',
    'forecast_period_1',
    'forecast_period_2',
    'forecast_period_3',
    'latitude_longitude',
    'time_1_bnds',
    'time_0_bnds',
    'forecast_period_1_bnds',
    'surface_altitude',
    'level_height',
    'sigma',
    'altitude',
]

time2d = pd.date_range('2020-01-20', '2021-03-01', freq='h')
time3d = pd.date_range('2020-01-20', '2021-03-01', freq='3h')
# timeP1D = pd.date_range('2020-01-20', '2021-03-01', freq='D')


def check_cube_time_length(cube):
    if cube.shape[0] == 13:
        cube = cube[1:]
    return cube


group2d = {
    'time': time2d,
    'zarr_store': 'PT1H',
    'name_map': name_map_2d,
    'constraint': has_dimensions("time", "latitude", "longitude"),
    # 'extra_constraints': {
    #     'stratiform_rainfall_flux': iris.Constraint(name='stratiform_rainfall_flux') & iris.Constraint(
    #         cube_func=cube_cell_method_is_not_empty),
    #     'stratiform_snowfall_flux': iris.Constraint(name='stratiform_snowfall_flux') & iris.Constraint(
    #         cube_func=cube_cell_method_is_not_empty),
    #     'toa_outgoing_shortwave_flux': iris.Constraint(
    #         name='toa_outgoing_shortwave_flux') & iris.AttributeConstraint(
    #         STASH='m01s01i208'),
    # },
    # 'extra_processing': {
    #     'surface_upward_latent_heat_flux': invert_cube_sign,
    #     'surface_upward_sensible_heat_flux': invert_cube_sign,
    # },
    # 'extra_attrs': {
    #     'stratiform_rainfall_flux': {'notes':
    #                                      'hourly mean - time index shifted from half past the hour to the following hour'},
    #     'stratiform_snowfall_flux': {'notes':
    #                                      'hourly mean - time index shifted from half past the hour to the following hour'},
    # },
    'chunks': chunks2d,
}

# TODO: Not working! Problem with time dimension only having 1 value per day.
# group2d_sm = {
#     'time': timeP1D,
#     'zarr_store': 'P1D',
#     'name_map': {'moisture_content_of_soil_layer': ('soil_liquid_water_content', 'mrso')},
#     'constraint': has_dimensions("depth", "latitude", "longitude"),
#     'extra_constraints': {},
#     'extra_processing': {},
#     'extra_attrs': {},
#     'chunks': chunks2d,
# }

# group2d_strat_conf_pr = {
#     'time': time2d,
#     'zarr_store': 'PT1H',
#     'name_map': {'stratiform_rainfall_flux': ('stratiform_rainfall_flux', 'pr')},
#     'constraint': has_dimensions("time", "latitude", "longitude"),
#     'extra_constraints': {
#         'stratiform_rainfall_flux': iris.Constraint(name='stratiform_rainfall_flux') & iris.Constraint(
#             cube_func=cube_cell_method_is_not_empty),
#     },
#     'extra_processing': {},
#     'extra_attrs': {},
#     'chunks': chunks2d,
# }
#
# CoMA9 use instantaneous values total precip flux: 5, 216.
# group2d_CoMA9_instantaenous = {
#     'time': time2d,
#     'zarr_store': 'PT1H',
#     'name_map': {
#         'precipitation_flux': ('precipitation_flux', 'pr'),
#     },
#     'constraint': has_dimensions("time", "latitude", "longitude"),
#     'extra_constraints': {
#         'precipitation_flux': iris.AttributeConstraint(STASH='m01s05i216') & iris.Constraint(cube_func=cube_cell_method_is_empty)
#     },
#     'extra_processing': {
#         'precipitation_flux': check_cube_time_length,
#     },
#     'extra_attrs': {},
#     'chunks': chunks2d,
# }
#

group3d = {
    'time': time3d,
    'zarr_store': 'PT3H',
    'name_map': name_map_3d,
    'constraint': has_dimensions("time", "pressure", "latitude", "longitude"),
    'chunks': chunks3d,
}

group3d_ml = {
    'time': time3d,
    'zarr_store': 'PT3H',
    'name_map': name_map_3d_ml,
    'constraint': has_dimensions("time", "model_level_number", "latitude", "longitude"),
    'chunks': chunks3d,
}

# Idea is to map to e.g. these filenames.
# Global:
# ./10km-CoMA9/glm/field.pp/apvera.pp/glm.n1280_CoMA9.apvera_20200120T00.pp
# ./5km-RAL3/glm/field.pp/apvera.pp/glm.n2560_RAL3p3.apvera_20200120T00.pp
# ./10km-GAL9-nest/glm/field.pp/apvera.pp/glm.n1280_GAL9_nest.apvera_20200120T00.pp

global_sim_keys = {
    'glm.n1280_CoMA9': '10km-CoMA9',
    'glm.n2560_RAL3p3': '5km-RAL3',
    'glm.n1280_GAL9_nest': '10km-GAL9-nest',
}

global_configs = {
    key: {
        'name': key,
        'regional': False,
        'add_cyclic': True,
        'basedir': Path(f'/gws/nopw/j04/kscale/DYAMOND3_data/{simdir}/glm'),
        'weightsdir': Path('/gws/nopw/j04/hrcm/mmuetz/weights/'),
        'donedir': Path(f'/gws/nopw/j04/hrcm/mmuetz/slurm_done/{deploy}'),
        'donepath_tpl': f'{key}/{output_vn}/{{task}}_{{date}}.done',
        'first_date': pd.Timestamp(2020, 1, 20, 0),
        'max_zoom': 10 if key.startswith('glm.n2560') else 9,
        'zarr_store_url_tpl': f's3://sim-data/{deploy}/{output_vn}/{key}/um.{{freq}}.hp_z{{zoom}}.zarr',
        'drop_vars': drop_vars,
        'groups': {
            '2d': group2d,
            '3d': group3d,
            '3d_ml': group3d_ml,
        },
        'metadata': {
            'simulation': key,
        }
    }
    for key, simdir in global_sim_keys.items()
}

# global_configs['glm.n2560_RAL3p3_SM'] = {
#     'regional': False,
#     'add_cyclic': True,
#     'basedir': Path(f'/gws/nopw/j04/kscale/DYAMOND3_data/5km-RAL3/glm'),
#     'weightsdir': Path('/gws/nopw/j04/hrcm/mmuetz/weights/'),
#     'donedir': Path(f'/gws/nopw/j04/hrcm/mmuetz/slurm_done/{deploy}'),
#     'donepath_tpl': f'glm.n2560_RAL3p3/{output_vn}/{{task}}_{{date}}.sm.done',
#     'first_date': pd.Timestamp(2020, 1, 20, 0),
#     'max_zoom': 10,
#     'zarr_store_url_tpl': f's3://sim-data/{deploy}/{output_vn}/glm.n2560_RAL3p3/um.{{freq}}.hp_z{{zoom}}.sm.zarr',
#     'drop_vars': drop_vars,
#     'groups': {
#         '2d_sm': group2d_sm,
#     },
#     'metadata': {
#         'simulation': 'glm.n2560_RAL3p3',
#     }
# }

# global_configs['glm.n1280_GAL9_nest_strat_conv_pr'] = {
#     'name': 'glm.n1280_GAL9_nest',
#     'regional': False,
#     'add_cyclic': True,
#     'basedir': Path(f'/gws/nopw/j04/kscale/DYAMOND3_data/10km-CoMA9/glm'),
#     'weightsdir': Path('/gws/nopw/j04/hrcm/mmuetz/weights/'),
#     'donedir': Path(f'/gws/nopw/j04/hrcm/mmuetz/slurm_done/{deploy}'),
#     'donepath_tpl': f'glm.n1280_GAL9_nest/{output_vn}/{{task}}_{{date}}.strat_conv_pr2.done',
#     'first_date': pd.Timestamp(2020, 1, 20, 0),
#     'max_zoom': 9,
#     'zarr_store_url_tpl': f's3://sim-data/{deploy}/{output_vn}/glm.n1280_GAL9_nest/um.{{freq}}.hp_z{{zoom}}.zarr',
#     'drop_vars': drop_vars,
#     'groups': {
#         '2d_strat_conv_pr': group2d_strat_conf_pr,
#     },
#     'metadata': {
#         'simulation': 'glm.n1280_GAL9_nest',
#     }
# }

global_configs['glm.n2560_RAL3p3']['metadata'].update({
    'simulation_description': ('The MetUM uses a regular lat-lon grid, for our explicit convection global simulations '
                               'we use the N2560 global grid (~5 km in mid-latitudes) and the latest regional '
                               'atmosphere-land configuration (RAL3p3). As detailed in Bush et al 2025 the RAL3p3 '
                               'includes significant developments over previous configurations including the CASIM '
                               'double-moment cloud microphysics scheme and the bi-modal large-scale cloud scheme. '
                               'Crucially for DYAMOND-3 simulations the parameterisation of convection is not active '
                               'and this science configuration has been developed and evaluated targetting '
                               'high-resolution (regional) simulations.'),
})


# Regional:
# ./10km-GAL9-nest/Africa_km4p4_CoMA9_TBv1/field.pp/apvera.pp/Africa_km4p4_CoMA9_TBv1.n1280_GAL9_nest.apvera_20200120T00.pp
# ./10km-GAL9-nest/Africa_km4p4_RAL3P3/field.pp/apvera.pp/Africa_km4p4_RAL3P3.n1280_GAL9_nest.apvera_20200120T00.pp
# ./10km-GAL9-nest/SAmer_km4p4_CoMA9_TBv1/field.pp/apvera.pp/SAmer_km4p4_CoMA9_TBv1.n1280_GAL9_nest.apvera_20200120T00.pp
# ./10km-GAL9-nest/SAmer_km4p4_RAL3P3/field.pp/apvera.pp/SAmer_km4p4_RAL3P3.n1280_GAL9_nest.apvera_20200120T00.pp
# ./10km-GAL9-nest/SEA_km4p4_CoMA9_TBv1/field.pp/apvera.pp/SEA_km4p4_CoMA9_TBv1.n1280_GAL9_nest.apvera_20200120T00.pp
# ./10km-GAL9-nest/SEA_km4p4_RAL3P3/field.pp/apvera.pp/SEA_km4p4_RAL3P3.n1280_GAL9_nest.apvera_20200120T00.pp
# Cyclic Tropical Channel (CTC):
# ./10km-GAL9-nest/CTC_km4p4_CoMA9_TBv1/field.pp/apvera.pp/CTC_km4p4_CoMA9_TBv1.n1280_GAL9_nest.apvera_20200120T00.pp
# ./10km-GAL9-nest/CTC_km4p4_RAL3P3/field.pp/apvera.pp/CTC_km4p4_RAL3P3.n1280_GAL9_nest.apvera_20200120T00.pp
def map_regional_key_to_path(simdir, regional_key):
    sim_key, _ = regional_key.split('.')
    return Path(f'/gws/nopw/j04/kscale/DYAMOND3_data/{simdir}/{sim_key}')


regional_sim_keys = {
    'SAmer_km4p4_RAL3P3.n1280_GAL9_nest': '10km-GAL9-nest',
    'Africa_km4p4_RAL3P3.n1280_GAL9_nest': '10km-GAL9-nest',
    'SEA_km4p4_RAL3P3.n1280_GAL9_nest': '10km-GAL9-nest',
    'SAmer_km4p4_CoMA9_TBv1.n1280_GAL9_nest': '10km-GAL9-nest',
    'Africa_km4p4_CoMA9_TBv1.n1280_GAL9_nest': '10km-GAL9-nest',
    'SEA_km4p4_CoMA9_TBv1.n1280_GAL9_nest': '10km-GAL9-nest',
    'CTC_km4p4_RAL3P3.n1280_GAL9_nest': '10km-GAL9-nest',
    'CTC_km4p4_CoMA9_TBv1.n1280_GAL9_nest': '10km-GAL9-nest',
}

group2d_regional = group2d.copy()
group3d_regional = group3d.copy()
group3d_ml_regional = group3d_ml.copy()

# For GAL9 simulations, combine stratiform and convective, rainfall and snow.
group2d_GAL9 = {**group2d_regional}
group2d_GAL9[('pr', 'precipitation_flux')] = MultiMapItem(
    [
        MapItem(iris.Constraint(name='convective_rainfall_flux') & iris.Constraint(
            cube_func=cube_cell_method_is_not_empty)),
        MapItem(iris.Constraint(name='convective_snowfall_flux') & iris.Constraint(
            cube_func=cube_cell_method_is_not_empty)),
        MapItem(iris.Constraint(name='stratiform_rainfall_flux') & iris.Constraint(
            cube_func=cube_cell_method_is_not_empty)),
        MapItem(iris.Constraint(name='stratiform_snowfall_flux') & iris.Constraint(
            cube_func=cube_cell_method_is_not_empty)),
    ],
    ops=[operator.add, operator.add, operator.add],
    extra_attrs={
        'notes': 'Combined rain and snow, convective and stratiform. Hourly mean - time index shifted from half past the hour to the following hour'},
)
group2d_regional['chunks'] = chunks2dregional
group3d_regional['chunks'] = chunks3dregional
group3d_ml_regional['chunks'] = chunks3dregional

regional_configs = {
    key: {
        'regional': True,
        # TODO: I think that the CTC simulation has a high enough res that there are no healpix coords outside
        # its domain - check.
        # Orig: I think this should be true for CTC, but it's raising an error: ValueError: The coordinate must be equally spaced.
        # 'add_cyclic': key.startswith('CTC'),  # only difference from regional.
        'add_cyclic': False,
        'basedir': map_regional_key_to_path(simdir, key),
        'weightsdir': Path('/gws/nopw/j04/hrcm/mmuetz/weights/'),
        'donedir': Path(f'/gws/nopw/j04/hrcm/mmuetz/slurm_done/{deploy}'),
        'donepath_tpl': f'{key}/{output_vn}/{{task}}_{{date}}.done',
        'max_zoom': 10,
        'first_date': pd.Timestamp(2020, 1, 20, 0),
        'zarr_store_url_tpl': f's3://sim-data/{deploy}/{output_vn}/{key}/um.{{freq}}.hp_z{{zoom}}.zarr',
        'drop_vars': drop_vars,
        'groups': {
            '2d': group2d_GAL9 if key.endswith('GAL9') else group2d_regional,
            '3d': group3d_regional,
            '3d_ml': group3d_ml_regional,
        },
        'metadata': {
            'simulation': key,
        }
    }
    for key, simdir in regional_sim_keys.items()
}

processing_config = {
    **global_configs,
    **regional_configs,
}


if __name__ == '__main__':
    def stringify_values(d):
        if isinstance(d, dict):
            return {k: stringify_values(v) for k, v in d.items()}
        else:
            return str(d)

    config_key = sys.argv[1]
    print_config = stringify_values(processing_config[config_key])

    import json
    print(json.dumps(print_config, indent=2, sort_keys=False))
