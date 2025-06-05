from collections import defaultdict
from functools import partial
from pathlib import Path

import pandas as pd

static_var_path_map = {
    'Land_mask': '2D/pa/pa_030_Land_mask',
    'orography': '2D/pa/pa_033_orography',
}

var_path_map = {
    'surf_temp_after_timestep': '2D/pa/pa_024_surf_temp_after_timestep',
    'surf_pressure_after_timestep': '2D/pa/pa_0409_surf_pressure_after_timestep',
    'net_down_surf_SW_flux_correct': '2D/pa/pa_1202_net_down_surf_SW_flux_correct',
    # Done: which to use? One's been deleted.
    # 'outgoing_SW_rad_flux_TOA_correct': '2D/pa/pa_1205_outgoing_SW_rad_flux_TOA_correct',
    'outgoing_SW_rad_flux_TOA_corrected': '2D/pa/pa_1205_outgoing_SW_rad_flux_TOA_corrected',
    'incoming_SW_rad_flux_TOA_all_tss': '2D/pa/pa_1207_incoming_SW_rad_flux_TOA_all_tss',
    'tot_downward_surf_SW_flux': '2D/pa/pa_1235_tot_downward_surf_SW_flux',
    'surf_pressure_at_mean_sea_level': '2D/pa/pa_16222_surf_pressure_at_mean_sea_level',
    'net_down_surf_LW_rad_flux': '2D/pa/pa_2201_net_down_surf_LW_rad_flux',
    'outgoing_LW_rad_flux': '2D/pa/pa_2205_outgoing_LW_rad_flux',
    'downward_LW_rad_flux_surf': '2D/pa/pa_2207_downward_LW_rad_flux_surf',
    'surf_sensi_heat_flux': '2D/pa/pa_3217_surf_sensi_heat_flux',
    '10m_wind_u_b_grid': '2D/pa/pa_3225_10m_wind_u_b_grid',
    '10m_wind_v_b_grid': '2D/pa/pa_3226_10m_wind_v_b_grid',
    'surf_latent_heat_flux': '2D/pa/pa_3234_surf_latent_heat_flux',
    'Temp_at_1_5M': '2D/pa/pa_3236_Temp_at_1_5M',
    'specific_humidity_at_1_5M': '2D/pa/pa_3237_specific_humidity_at_1_5M',
    'Tot_precip_rate': '2D/pb/pb_5216_Tot_precip_rate',
    # 'Tot_precip_rate2': '2D/pc/pc_5216_Tot_precip_rate',
    'TotCoud_Random_overlap': '2D/pc/pc_9216_TotCoud_Random_overlap',
    'TotCol_OCL_Rho_Grid': '2D/pe/pe_30405_TotCol_OCL_Rho_Grid',
    'TotCol_OCF_Rho_Grid': '2D/pe/pe_30406_TotCol_OCF_Rho_Grid',
    'TotCol_Q': '2D/pe/pe_30461_TotCol_Q',
    'shum': '3D/pd_shum',
    'T': '3D/pe_T',
    'w': '3D/pe_w',
    'u': '3D/pf_u',
    'Rh': '3D/pg_Rh',
    'v': '3D/pg_v',
    'gpotH': '3D/ph_gpotH',
}


def custom_datestr_fmt(dates, dt):
    archive_error_date = pd.Timestamp('2020-11-16')
    if dt < pd.Timestamp("2021-01-01"):
        hours = (dt - dates[0]).total_seconds() / 3600
        if hours < 1000:
            return f'{int(hours):03}'
        elif dt < archive_error_date:
            return f'{int(hours):04}'
        elif dt == archive_error_date:
            return dt.strftime('%d%m%Y')
        elif dt > archive_error_date:
            return f'{int(hours - 24):04}'
    else:
        return dt.strftime('%d%m%Y')


def gen_date_path_map(dates, formatted_dates):
    date_path_map = defaultdict(list)
    for date, formatted_date in zip(dates, formatted_dates):
        print(date, formatted_date)
        for varname, vardir in var_path_map.items():
            # dim = vardir.split('/')[0]
            if varname == 'TotCoud_Random_overlap':
                # This variable is under dir pc but has pb in its filename
                stream = 'pb'
            else:
                stream = vardir.split('/')[-1].split('_')[0]
            date_path_map[date].append(basedir / vardir / finename_tpl.format(stream=stream, datestr=formatted_date))
    return date_path_map


def check_paths_exist(date_path_map):
    missing_paths = []
    found_paths = []
    for date, paths in date_path_map.items():
        for path in paths:
            if not path.exists():
                missing_paths.append(path)
            else:
                found_paths.append(path)
    if missing_paths:
        raise Exception(f'Number of missing paths: {len(missing_paths)}')
    print(f'Found {len(found_paths)} paths')


if __name__ == '__main__':
    dates = pd.date_range('2020-01-01', '2021-01-31', freq='D')
    formatted_dates = dates.map(partial(custom_datestr_fmt, dates))

    basedir = Path('/gws/nopw/j04/hrcm/hackathon/')

    finename_tpl = '20200101T0000Z_{stream}{datestr}.pp'
    formatted_paths = {}

    date_path_map = gen_date_path_map(dates, formatted_dates)
    check_paths_exist(date_path_map)
