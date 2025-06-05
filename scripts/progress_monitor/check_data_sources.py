import pickle
from collections import defaultdict
from pathlib import Path

import pandas as pd
from loguru import logger

from processing_config import processing_config
from slurm_control import sysrun

TB = 2**40


def scan_input_dirs(sims, dates):
    streams = 'abcd'
    data = defaultdict(list)

    for sim in sims:
        logger.info(f'scan input dirs for {sim}')
        config = processing_config[sim]
        basedir = config['basedir']
        for stream in streams:
            for date in dates:
                # e.g. /gws/nopw/j04/kscale/DYAMOND3_data/5km-RAL3/glm/field.pp/apvera.pp/glm.n2560_RAL3p3.apvera_20200329T00.pp
                datestr = date.strftime('%Y%m%dT%H')
                path = basedir / f'field.pp/apver{stream}.pp/{sim}.apver{stream}_{datestr}.pp'
                data['sim'].append(sim)
                data['stream'].append(stream)
                data['date'].append(date)
                data['path'].append(str(path))
                if path.exists():
                    # exists_paths[sim].append(path)
                    data['exists'].append(True)
                    data['size'].append(path.stat().st_size)
                else:
                    data['exists'].append(False)
                    data['size'].append(0)

    return pd.DataFrame(data)


def get_obj_store_s3urls(vn):
    cmd = f's3cmd ls s3://sim-data/dev/{vn}/'
    ls = sysrun(cmd).stdout
    s3urls = [l.split()[-1] for l in ls.split('\n') if l]
    return s3urls


def check_obj_store_sizes(vn, sims, zooms=(10,)):
    s3urls = [
        f's3://sim-data/dev/{vn}/{sim}/'
        for sim in sims
    ]
    data = defaultdict(list)

    for sim, s3url in zip(sims, s3urls):
        logger.info(f'check object store size for {sim}')
        for zoom in zooms:
            for freq in ['PT1H', 'PT3H']:
                cmd = f's3cmd du {s3url}um.{freq}.hp_z{zoom}.zarr/'
                logger.trace(cmd)
                du = sysrun(cmd).stdout
                logger.trace(du)
                dusize = int(du.strip().split()[0])

                data['sim'].append(sim)
                data['zoom'].append(zoom)
                data['freq'].append(freq)
                data['dusize'].append(int(dusize))
    return pd.DataFrame(data)


def check_donefiles(vn, sims):
    basedir = Path('/gws/nopw/j04/hrcm/mmuetz/slurm_done/dev/')
    data = defaultdict(list)
    for sim in sims:
        logger.info(f'check donefiles for {sim}')
        donefiles = sorted((basedir / sim / vn).glob('regrid*.done'))
        logger.trace(sim, len(donefiles))
        for donefile in donefiles:
            data['sim'].append(sim)
            data['donefile'].append(str(donefile))
    return pd.DataFrame(data)


def main(vn):
    start_date = '2020-01-20'
    end_date = '2021-04-01'
    dates = pd.date_range(start=start_date, end=end_date, freq='12h')

    sims = list(processing_config)
    # sims = [s for s in sims if 'n2560' not in s and 'SEA' in s]
    # sims = [s for s in sims if 'n2560' not in s]
    logger.debug(sims)

    logger.info('scanning input directories...')
    df_inputs = scan_input_dirs(sims, dates)
    logger.info('checking object store sizes...')
    df_obj_store = check_obj_store_sizes(vn, sims)
    logger.info('checking donefiles...')
    df_donefiles = check_donefiles(vn, sims)

    # print(df_inputs)
    # print(df_obj_store)
    # print(df_donefiles)

    return sims, dates, df_inputs, df_obj_store, df_donefiles


class Dashboard:
    base_cachedir = Path('.cache')

    def __init__(self, sims, dates, df_inputs, df_obj_store, df_donefiles):
        self.sims = sims
        self.dates = dates
        self.df_inputs = df_inputs
        self.df_obj_store = df_obj_store
        self.df_donefiles = df_donefiles

    @classmethod
    def from_cache(cls, cachedir):
        cachedir = Path(cachedir)
        sims = pickle.load(open(cachedir / 'cache_sims.pkl', 'rb'))
        dates = pickle.load(open(cachedir / 'cache_dates.pkl', 'rb'))
        df_inputs = pd.read_hdf(cachedir / 'cache.hdf', key='inputs')
        df_donefiles = pd.read_hdf(cachedir / 'cache.hdf', key='donefiles')
        df_obj_store = pd.read_hdf(cachedir / 'cache.hdf', key='obj_store')
        return cls(sims, dates, df_inputs, df_obj_store, df_donefiles)

    def cache(self, datestr):
        simstr = str(len(self.sims))
        cachedir = self.base_cachedir / (datestr + simstr)
        cachedir.mkdir(parents=True, exist_ok=True)

        pickle.dump(self.sims, open(cachedir / 'cache_sims.pkl', 'wb'))
        pickle.dump(self.dates, open(cachedir / 'cache_dates.pkl', 'wb'))
        self.df_inputs.to_hdf(cachedir / f'cache.hdf', key='inputs')
        self.df_donefiles.to_hdf(cachedir / f'cache.hdf', key='donefiles')
        self.df_obj_store.to_hdf(cachedir / f'cache.hdf', key='obj_store')

    def completed_transfer(self):
        df_exists = self.df_inputs[self.df_inputs['exists']]

        curr_size = df_exists["size"].sum() / TB

        print('Number 12-h dates transferred')
        print(df_exists.groupby(['sim'])['exists'].sum() / 4)
        print()
        print('Number days transferred')
        print(df_exists.groupby(['sim'])['exists'].sum() / 4 / 2)
        print()
        print('Percent files transferred')
        print(df_exists.groupby(['sim'])['exists'].sum() / (len(self.dates) * 4) * 100)
        print()
        print('Percent transferred by size')
        print(df_exists.groupby(['sim'])['size'].sum() / (df_exists.groupby(['sim', 'stream'])['size'].mean().groupby('sim').sum() * len(self.dates)) * 100)
        print()
        print('Total estimated sizes for each sim')
        print(df_exists.groupby(['sim', 'stream'])['size'].mean().groupby('sim').sum() * len(self.dates) / TB)
        print()
        print(f'Current size: {curr_size:.3f}T')
        total_est_size = (df_exists.groupby(['sim', 'stream'])['size'].mean() * len(self.dates) / (2**40)).sum()
        print(f'Total estimated size: {total_est_size:.3f}T')
        print(f'Percent done: {curr_size / total_est_size * 100:.2f}%')

    def completed_processing(self):
        df_inputs_complete = self.df_inputs[self.df_inputs.exists].groupby(['sim', 'date']).count()['stream'] == 4
        print('Percent of existing processed')
        print(self.df_donefiles.groupby('sim').count()['donefile'] / df_inputs_complete.groupby('sim').count() * 100)

    def obj_store_sizes(self):
        print('Object store estimated final sizes')
        print('z10')
        est_final_sizes = (self.df_obj_store.groupby('sim').sum()['dusize'] * len(self.dates) /
                           self.df_donefiles.groupby('sim').count()[
                               'donefile'] / TB)
        print(est_final_sizes)
        # df['est_total_size'] = len(self.dates) / df.ndonefiles * df.dusize
        print('z10-0')
        print(est_final_sizes * 1.4)
        print('Object store estimated total final size')
        print(est_final_sizes.sum() * 1.4)

    def print_all(self):
        def title(msg):
            print('=' * len(msg))
            print(msg)
            print('=' * len(msg))
        title('Completed transfer')
        self.completed_transfer()
        title('Completed processing')
        self.completed_processing()
        title('Obj store sizes')
        self.obj_store_sizes()


if __name__ == "__main__":
    pass
    # df_inputs, df_obj_store, df_donefiles = main()
    # df['est_total_size'] = 437 * 2 / df.ndonefiles * df.dusize
    # print(df.est_total_size * 1.4 / TB)
    # print(float(df.est_total_size.sum() * 1.4 / TB))
