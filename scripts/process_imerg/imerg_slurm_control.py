import json
import subprocess as sp
import sys
from pathlib import Path

import click
import pandas as pd
from loguru import logger

from imerg_config import vn

SLURM_SCRIPT_ARRAY = """#!/bin/bash
#SBATCH --job-name="{job_name}"
#SBATCH --time=10:00:00
#SBATCH --mem={mem}
#SBATCH --account=hrcm
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --array=0-{njobs}%{nconcurrent_tasks}
#SBATCH -o slurm/output/{job_name}_imerg_{date_string}_%A_%a.out
#SBATCH -e slurm/output/{job_name}_imerg_{date_string}_%A_%a.err
#SBATCH --comment={comment}
# These nodes repeatedly fail to be able to read the kscale GWS.
#SBATCH --exclude=host1012,host1077,host1087,host1106
{dependency}

# Quick check to see if it can access the kscale GWS.
if ! ls /gws/nopw/j04/hrcm > /dev/null 2>&1; then
    echo "ERROR: kscale GWS not accessible on $(hostname)! Exiting."
    exit 99
fi

ARRAY_INDEX=${{SLURM_ARRAY_TASK_ID}}

python imerg_process_tasks.py slurm {tasks_path} ${{ARRAY_INDEX}}
"""


def sysrun(cmd):
    return sp.run(cmd, check=True, shell=True, stdout=sp.PIPE, stderr=sp.PIPE, encoding='utf8')


def sbatch(slurm_script_path):
    try:
        return sysrun(f'sbatch --parsable {slurm_script_path}').stdout.strip()
    except sp.CalledProcessError as e:
        logger.error(f'sbatch failed with exit code {e.returncode}')
        logger.error(e)
        raise


def parse_date_from_imerg_path(path):
    datestr = path.stem.split('_')[1]
    return pd.to_datetime(datestr, format="%Y%m%d%H")


def write_tasks_slurm_job_array(tasks, job_name, nconcurrent_tasks=30, depends_on=None, mem=100000):
    now = pd.Timestamp.now()
    date_string = now.strftime("%Y%m%d_%H%M%S")

    tasks_path = Path(f'slurm/tasks/tasks_{job_name}_imerg_{date_string}.json')
    logger.debug(tasks_path)
    logger.trace(json.dumps(tasks, indent=4))

    if depends_on:
        dependency = f'#SBATCH --dependency=afterok:{depends_on}'
    else:
        dependency = ''

    with tasks_path.open('w') as f:
        json.dump(tasks, f, indent=4)

    comment = f'imerg,{job_name}'

    slurm_script_path = Path(f'slurm/scripts/script_{job_name}_imerg_{date_string}.sh')
    njobs = len(tasks) - 1
    slurm_script_path.write_text(
        SLURM_SCRIPT_ARRAY.format(job_name=job_name, njobs=njobs,
                                  mem=mem,
                                  nconcurrent_tasks=nconcurrent_tasks,
                                  tasks_path=tasks_path,
                                  dependency=dependency,
                                  date_string=date_string,
                                  comment=comment))
    return slurm_script_path


def find_imerg_paths(basedir):
    # Search for pp_paths with a specific date (N.B. filename sensitive).
    nc_paths = sorted(basedir.glob('*/*.nc'))
    logger.debug(f'found {len(nc_paths)} nc paths')
    dates_to_paths = {parse_date_from_imerg_path(p): p for p in nc_paths}
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


@cli.command()
@click.pass_context
def process(ctx):
    nconcurrent_tasks = ctx.obj['nconcurrent_tasks']

    logger.debug(f'using {nconcurrent_tasks} concurrent tasks')
    basedir = Path('/gws/nopw/j04/hrcm/mmuetz/obs/IR_IMERG_Combined_V07B')
    logger.debug(f'basedir: {basedir}')
    donedir = Path(f'/gws/nopw/j04/hrcm/mmuetz/slurm_done/obs/dev/ir_imerg_combined/{vn}')
    logger.debug(f'donedir: {donedir}')
    donepath_tpl = 'imerg.{task}.{date}.done'
    logger.debug(f'donepath_tpl: {donepath_tpl}')

    dates_to_paths = find_imerg_paths(basedir)
    dates = list(dates_to_paths)
    first_date = dates[0]
    last_date = dates[-1]

    create_jobid = None
    jobids = []

    create_donepath = donedir / donepath_tpl.format(task='create_empty_zarr_store', date=first_date)
    if not create_donepath.exists():
        logger.info('Creating zarr store')
        create_task = {
            'task_type': 'create_empty_zarr_stores',
            'first_date': str(first_date),
            'last_date': str(last_date),
            'inpath': str(dates_to_paths[first_date]),
            'donepath': str(create_donepath),
        }
        slurm_script_path = write_tasks_slurm_job_array([create_task], f'createzarr',
                                                        nconcurrent_tasks=1)
        logger.debug(slurm_script_path)
        create_donepath.parent.mkdir(parents=True, exist_ok=True)
        if not ctx.obj['dry_run']:
            create_jobid = sbatch(slurm_script_path)
            logger.info(f'create empty zarr stores jobid: {create_jobid}')
            jobids.append(create_jobid)

    # Build a list of tasks for all donepaths that don't exist.
    tasks = []
    for first_of_month in pd.date_range(f'{first_date.year}-{first_date.month}', f'{last_date.year}-{last_date.month}', freq="MS"):

        donepath = (donedir / donepath_tpl.format(task='regrid', date=first_of_month))
        donepath.parent.mkdir(parents=True, exist_ok=True)
        if donepath.exists():
            logger.debug(f'{first_of_month}: already processed')
        else:
            logger.info(f'{first_of_month}: processing')
            dates = pd.date_range(first_of_month, first_of_month + pd.DateOffset(months=1), freq='h')
            tasks.append(
                {
                    'task_type': 'regrid',
                    'inpaths': [str(dates_to_paths[date])
                                for date in dates
                                if date in dates_to_paths],
                    'donepath': str(donepath),
                }
            )

    regrid_jobid = None
    if len(tasks):
        logger.info(f'Running {len(tasks)} tasks')
        slurm_script_path = write_tasks_slurm_job_array(tasks, 'regrid',
                                                        nconcurrent_tasks=nconcurrent_tasks,
                                                        depends_on=create_jobid,
                                                        mem=200000)
        logger.debug(slurm_script_path)
        if not ctx.obj['dry_run']:
            regrid_jobid = sbatch(slurm_script_path)
        logger.debug(f'regrid jobid: {regrid_jobid}')
        jobids.append(regrid_jobid)
    else:
        logger.info('No tasks to run')

    if not ctx.obj['dry_run']:
        write_jobids(jobids)


if __name__ == '__main__':
    cli(obj={})
