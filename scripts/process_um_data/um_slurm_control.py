"""Entry point into submitting SLURM jobs.

* Works by scanning input directories, then builds a list of jobs based on whether each job has already been done yet.
* Uses click to build a nice CLI.
"""
import math
import sys
import json
import subprocess as sp
from collections import defaultdict
from itertools import batched
from pathlib import Path

import click
import pandas as pd
from loguru import logger

from cube_to_da_mapping import DataArrayExtractor
from um_processing_config import slurm_config, processing_config, time2d, time3d
from util import sysrun

# SLURB script template - filled in and written to a file for calling with `sbatch`.
SLURM_SCRIPT_ARRAY = """#!/bin/bash
#SBATCH --job-name="{job_name}"
#SBATCH --time=10:00:00
#SBATCH --mem={mem}
#SBATCH --account={account}
#SBATCH --ntasks={ntasks}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --partition={partition}
#SBATCH --qos={qos}
#SBATCH --array=0-{njobs}%{nconcurrent_tasks}
#SBATCH -o slurm/output/{job_name}_{config_key}_{date_string}_%A_%a.out
#SBATCH -e slurm/output/{job_name}_{config_key}_{date_string}_%A_%a.err
#SBATCH --comment={comment}
{dependency}

# These nodes repeatedly fail to be able to read the kscale GWS.
# Apparently these have been fixed:
# I used to have this #SBATCH --exclude=host1012,host1077,host1087,host1106,host1186,host1080,host1197,host1135,host1238,host1222,host1234

# Quick check to see if it can access the kscale GWS.
if ! ls /gws/nopw/j04/kscale > /dev/null 2>&1; then
    echo "ERROR: kscale GWS not accessible on $(hostname)! Exiting."
    exit 99
fi

ARRAY_INDEX=${{SLURM_ARRAY_TASK_ID}}

python um_process_tasks.py slurm {tasks_path} ${{ARRAY_INDEX}}
"""

def sbatch(slurm_script_path):
    """Submit script using sbatch."""
    try:
        return sysrun(f'sbatch --parsable {slurm_script_path}').stdout.strip()
    except sp.CalledProcessError as e:
        logger.error(f'sbatch failed with exit code {e.returncode}')
        logger.error(e)
        raise


def _parse_date_from_pp_path(path):
    datestr = path.stem.split('.')[-1].split('_')[1]
    return pd.to_datetime(datestr, format="%Y%m%dT%H")


def write_tasks_slurm_job_array(config_key, tasks, job_name, depends_on=None, **kwargs):
    """Write out a script for submission."""
    now = pd.Timestamp.now()
    date_string = now.strftime("%Y%m%d_%H%M%S")

    tasks_path = Path(f'slurm/tasks/tasks_{job_name}_{config_key}_{date_string}.json')
    logger.debug(tasks_path)
    logger.trace(json.dumps(tasks, indent=4))

    if depends_on:
        dependency = f'#SBATCH --dependency=afterok:{depends_on}'
    else:
        dependency = ''

    with tasks_path.open('w') as f:
        json.dump(tasks, f, indent=4)

    comment = f'{config_key},{job_name}'

    slurm_script_path = Path(f'slurm/scripts/script_{job_name}_{config_key}_{date_string}.sh')
    njobs = len(tasks) - 1
    slurm_kwargs = {**slurm_config, **kwargs}
    script_kwargs = dict(
        job_name=job_name,
        config_key=config_key,
        njobs=njobs,
        tasks_path=tasks_path,
        dependency=dependency,
        date_string=date_string,
        comment=comment,
    )

    logger.debug({**slurm_kwargs, **script_kwargs})
    slurm_script_path.write_text(SLURM_SCRIPT_ARRAY.format(**{**slurm_kwargs, **script_kwargs}))
    return slurm_script_path


def find_dyamond3_pp_dates_to_paths(basedir):
    """Search for pp_paths with a specific date (N.B. filename sensitive)."""
    pp_paths = sorted(basedir.glob('field.pp/apve*/**/*.pp'))
    logger.debug(f'found {len(pp_paths)} pp paths')
    pp_paths = [p for p in pp_paths if p.is_file()]
    dates_to_paths = defaultdict(list)
    for path in pp_paths:
        # These appear after about 2020-02-20 - not sure why.
        # Not sure what's in them either.
        if 'apvere' in path.stem:
            continue
        dates_to_paths[_parse_date_from_pp_path(path)].append(path)
    # Only keep completed downloads.
    dates_to_paths = {
        k: v for k, v in dates_to_paths.items()
        if len(v) == 4
    }
    logger.debug(f'found {len(dates_to_paths)} complete dates')
    return dates_to_paths


def write_jobids(jobids):
    now = pd.Timestamp.now()
    date_string = now.strftime("%Y%m%d_%H%M%S")
    jobids_path = Path(f'slurm/jobids/jobids_{date_string}.json')
    with jobids_path.open('w') as f:
        json.dump(jobids, f, indent=4)
    logger.info(f'written jobids to: {jobids_path}')


@click.group()
@click.option('--dry-run', '-n', is_flag=True)
@click.option('--debug', '-D', is_flag=True)
@click.option('--trace', '-T', is_flag=True)
@click.option('--nconcurrent-tasks', '-N', default=40, type=int)
@click.pass_context
def cli(ctx, dry_run, debug, trace, nconcurrent_tasks):
    ctx.ensure_object(dict)
    ctx.obj['dry_run'] = dry_run
    ctx.obj['nconcurrent_tasks'] = nconcurrent_tasks
    logger.remove()
    if trace:
        logger.add(sys.stderr, level="TRACE")
    elif debug:
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.add(sys.stderr, level="INFO")

    if dry_run:
        logger.warning("Dry run: not launching any jobs")

    for path in ['slurm/tasks', 'slurm/scripts', 'slurm/output', 'slurm/jobids/']:
        path = Path(path)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)


@cli.result_callback()
@click.pass_context
def cli_exit(ctx, result, **kwargs):
    # This runs after any subcommand completes
    if ctx.obj['dry_run']:
        logger.warning("Dry run: not launching any jobs")


@cli.command()
@click.argument('config_key')
@click.pass_context
def process(ctx, config_key):
    nconcurrent_tasks = ctx.obj['nconcurrent_tasks']

    logger.debug(f'using {nconcurrent_tasks} concurrent tasks')
    logger.info(f'Running for {config_key}')
    config = processing_config[config_key]
    logger.trace(config)
    basedir = config['basedir']
    donedir = config['donedir']
    donepath_tpl = config['donepath_tpl']
    logger.debug(f'basedir: {basedir}')
    logger.debug(f'donedir: {donedir}')
    logger.debug(f'donepath_tpl: {donepath_tpl}')

    dates_to_paths = find_dyamond3_pp_dates_to_paths(basedir)

    create_jobid = None
    jobids = []

    # Build a list of tasks for all donepaths that don't exist.
    tasks = []
    for date in dates_to_paths:
        # TODO:!!
        if date > pd.Timestamp('2020-02-01 00:00'):
            break
        if date == config['first_date']:
            create_donepath = donedir / donepath_tpl.format(task='create_empty_zarr_store', date=date)
            logger.debug(create_donepath)
            if not create_donepath.exists():
                # Create a task to create empty zarr stores on first date and if not already completed.
                logger.info('Creating zarr store')
                create_task = {
                    'task_type': 'create_empty_zarr_stores',
                    'config_key': config_key,
                    'date': str(date),
                    'inpaths': [str(p) for p in dates_to_paths[date]],
                    'donepath': str(create_donepath),
                }
                slurm_script_path = write_tasks_slurm_job_array(config_key, [create_task], f'createzarr',
                                                                nconcurrent_tasks=nconcurrent_tasks)
                logger.debug(slurm_script_path)
                create_donepath.parent.mkdir(parents=True, exist_ok=True)
                if not ctx.obj['dry_run']:
                    create_jobid = sbatch(slurm_script_path)
                    logger.info(f'create empty zarr stores jobid: {create_jobid}')
                    jobids.append(create_jobid)

        donepath = (donedir / donepath_tpl.format(task='regrid', date=date))
        donepath.parent.mkdir(parents=True, exist_ok=True)
        if donepath.exists():
            logger.debug(f'{date}: already processed')
        else:
            # Create regrid task for a given date.
            logger.info(f'{date}: processing')
            tasks.append(
                {
                    'task_type': 'regrid',
                    'config_key': config_key,
                    'date': str(date),
                    'inpaths': [str(p) for p in dates_to_paths[date]],
                    'donepath': str(donepath),
                }
            )

    regrid_jobid = None
    if len(tasks):
        # Run tasks.
        logger.info(f'Running {len(tasks)} tasks')
        slurm_script_path = write_tasks_slurm_job_array(config_key, tasks, 'regrid',
                                                        nconcurrent_tasks=nconcurrent_tasks,
                                                        depends_on=create_jobid)
        logger.debug(slurm_script_path)
        if not ctx.obj['dry_run']:
            regrid_jobid = sbatch(slurm_script_path)
        logger.debug(f'regrid jobid: {regrid_jobid}')
        jobids.append(regrid_jobid)
    else:
        logger.info('No tasks to run')

    if not ctx.obj['dry_run']:
        write_jobids(jobids)


@cli.command()
@click.option('--nbatch', '-B', default=5)
@click.option('--endtime', '-E', default='2021-03-01 00:00')
@click.argument('config_key')
@click.pass_context
def coarsen(ctx, nbatch, endtime, config_key):
    nconcurrent_tasks = ctx.obj['nconcurrent_tasks']
    config = processing_config[config_key]
    jobids = []
    dummy_donepath_tpl = config['donepath_tpl']
    dummy_donepath = dummy_donepath_tpl.format(task='dummy', date='dummy')
    donereldir = Path(dummy_donepath).parent
    donepath_tpl = str(config['donedir'] / donereldir / 'coarsen/{dim}/z{zoom}/{job_id}.done')
    max_zoom = config['max_zoom']

    for dim in ['2d', '3d']:
        prev_zoom_job_id = None
        if dim == '2d':
            time_idx = time2d
        elif dim == '3d':
            time_idx = time3d
        else:
            raise Exception(f'unknown dim: {dim}')
        if endtime is not None:
            time_idx = time_idx[time_idx <= endtime]

        chunks = config['groups'][dim]['chunks']

        for zoom in range(max_zoom - 1, -1, -1):
            logger.info(f'calc jobs for zoom {zoom}')
            tasks = []
            timechunk = chunks[zoom][0]
            logger.debug(f'timechunk: {timechunk}')
            njobs = int(math.ceil(len(time_idx) / timechunk))
            job_idx = [
                i for i in range(njobs)
                if not Path(donepath_tpl.format(dim=dim, zoom=zoom, job_id=i)).exists()
            ]
            tgt_time_calcs = [
                {
                    'start_idx': i * timechunk,
                    'end_idx': (i + 1) * timechunk,
                    'donepath': donepath_tpl.format(dim=dim, zoom=zoom, job_id=i),
                }
                for i in job_idx
            ]
            for tgt_times in batched(tgt_time_calcs, nbatch):
                tasks.append(
                    {
                        'task_type': 'coarsen',
                        'config_key': config_key,
                        'tgt_zoom': zoom,
                        'dim': dim,
                        'tgt_times': tgt_times,
                    }
                )
            if len(tasks):
                logger.info(f'Running {len(tasks)} tasks')
                if dim == '3d':
                    mem = 256000
                else:
                    mem = 100000
                # The heart of this method is a ds.coarsen(cell=4).mean() call.
                # This benefits massively from a dask speed up.
                # Request lots of cores per task.
                slurm_script_path = write_tasks_slurm_job_array(
                    config_key, tasks, f'coarsen_{dim}_{zoom}',
                    depends_on=prev_zoom_job_id,
                    partition='standard',
                    nconcurrent_tasks=nconcurrent_tasks,
                    mem=mem,
                    qos='high',
                    # cpus_per_task=48,  # maxes out at 6 tasks/288 cpus because of max cpus.
                    cpus_per_task=12,  # maxes out at 24 tasks/288 cpus because of max cpus.
                )

                logger.debug(slurm_script_path)

                if not ctx.obj['dry_run']:
                    prev_zoom_job_id = sbatch(slurm_script_path)
                jobids.append(prev_zoom_job_id)

    if not ctx.obj['dry_run']:
        write_jobids(jobids)


@cli.command()
@click.pass_context
def ls(ctx):
    for key in processing_config:
        print(key)


@cli.command()
@click.argument('config_key')
@click.option('--date', '-d', default=None)
@click.option('--output-file', '-o', default=None)
@click.pass_context
def check_output_mapping(ctx, config_key, date, output_file):
    import iris
    import operator

    operator_symbol_map = {
        operator.add: '+',
        operator.sub: '-',
        operator.mul: '*',
        operator.truediv: '/',
    }
    if config_key == 'all':
        config_keys = list(processing_config)
    else:
        config_keys = [config_key]

    cols = ['expt', 'store', 'short_name', 'long_name', 'present', 'cube_name', 'stash_code']
    data = []
    for config_key in config_keys:
        # TODO: can't load data for Africa or SEA CTC??
        if 'Africa' in config_key or 'SEA' in config_key or 'CTC_km4p4_CoMA9' in config_key:
            continue
        logger.info(f'processing {config_key}')
        config = processing_config[config_key]
        if date is None:
            date = config['first_date']

        basedir = config['basedir']
        dates_to_paths = find_dyamond3_pp_dates_to_paths(basedir)
        inpaths = dates_to_paths[date]
        cubes = iris.load(inpaths)

        extractor = DataArrayExtractor(None, None)
        for group_name, group in config['groups'].items():
            logger.info(group_name)

            group_constraint = group['constraint']
            name_map = group['name_map']
            store = group['zarr_store']

            group_cubes = cubes.extract(group_constraint)
            for key, map_item in name_map.items():
                logger.debug(f'  {key}: {map_item}')
                short_name, long_name = key
                try:
                    item_cubes = extractor.extract_cubes(map_item, group_cubes)
                    if len(item_cubes) == 1:
                        cube = item_cubes[0]
                        cubestr = cube.name()
                        stashstr = cube.attributes['STASH']
                    else:
                        cube = item_cubes[0]
                        ops = map_item.ops
                        cube_list = [cube.name()]
                        stash_list = [str(cube.attributes['STASH'])]
                        for op, next_cube in zip(ops, item_cubes[1:]):
                            cube_list.extend([operator_symbol_map[op], next_cube.name()])
                            stash_list.extend([operator_symbol_map[op], str(next_cube.attributes['STASH'])])
                        cubestr = ' '.join(cube_list)
                        stashstr = ' '.join(stash_list)
                    data.append(
                        str(v) for v in [config_key, store, short_name, long_name, True, cubestr, stashstr, map_item.extra_attrs]
                    )
                except iris.exceptions.ConstraintMismatchError as cme:
                    data.append(
                        str(v) for v in [config_key, store, short_name, long_name, False, None, None, map_item.extra_attrs]
                    )

    df = pd.DataFrame(data, columns=cols)
    if output_file is not None:
        df.to_csv(output_file, index=False)
    else:
        print(df)


if __name__ == '__main__':
    cli(obj={})
