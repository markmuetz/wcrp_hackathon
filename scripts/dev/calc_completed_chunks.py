import asyncio
import sys
import math
import random
from itertools import groupby
from pathlib import Path
from time import sleep

from loguru import logger
import numpy as np
import s3fs
import xarray as xr

from processing_config import processing_config

s3cfg = dict([l.split(' = ') for l in Path('/home/users/mmuetz/.s3cfg').read_text().split('\n') if l])

def get_initialized_chunk_ids(url, variable):
    """Find all initalized chunks for variable.

    does an `ls` on the path, and perses the output."""
    # I think I might need to recreate this obj every time, or I get a:
    # PermissionError: Invalid token, possibly expired
    jasmin_s3 = s3fs.S3FileSystem(
        anon=False, secret=s3cfg['secret_key'],
        key=s3cfg['access_key'],
        client_kwargs={'endpoint_url': 'http://hackathon-o.s3.jc.rl.ac.uk'}
    )

    lsout = jasmin_s3.ls(f'{url}/{variable}/')
    fnames = [p.split('/')[-1] for p in lsout]
    chunk_ids_str = [p for p in fnames if not p.startswith('.z')]
    chunk_ids = sorted([tuple([int(v) for v in c.split('.')]) for c in chunk_ids_str])
    # logger.trace(f'chunk_ids: {chunk_ids}')
    return chunk_ids

def count_first_chunk_id(chunk_ids):
    """Count number of first chunk ids"""
    return [(g, len(list(v))) for g, v in groupby(chunk_ids, lambda x: x[0])]

def completed_chunks(chunk_ids, complete_size, nchunks):
    """For each time chunk, see if there are the correct number of spatial."""
    first_chunks = count_first_chunk_id(chunk_ids)
    completed_idx = np.array([fc[0] for fc in first_chunks if fc[1] == complete_size])
    completed = np.zeros(nchunks, dtype=bool)
    if len(completed_idx):
        completed[completed_idx] = True
    return completed

def find_completed_time_chunks(src_ds, chunksize, complete_size, zoom, urls, variable):
    """Find all initialized chunks and check whether they are complete for each time"""
    rel_url = urls[zoom]
    # ds = xr.open_zarr('http://hackathon-o.s3.jc.rl.ac.uk/' + rel_url)
    ntimechunks = math.ceil(len(src_ds.time) / chunksize[0])
    init_chunk_ids = get_initialized_chunk_ids(rel_url, variable)
    return completed_chunks(init_chunk_ids, complete_size, ntimechunks)

def find_tgt_can_complete(factor, src_completed, tgt_completed):
    """Based on factor, see which tgt chunks can be completed from completed src chunks."""
    tgt_can_complete = np.zeros_like(tgt_completed, dtype=bool)
    # Length of complete chunks. Final possibly shorter chunk handled below.
    src_n = len(src_completed) - len(src_completed) % factor
    tgt_n = src_n // factor
    tgt_can_complete[:tgt_n] = src_completed[:src_n].reshape(-1, factor).all(axis=-1)
    # Handle shorter chunk.
    tgt_can_complete[-1] = len(src_completed[src_n:]) != 0 and src_completed[src_n:].all()
    return tgt_can_complete


async def async_retry_open_zarr(url, max_retries=20):
    retries = 0
    while retries < max_retries:
        try:
            ds = xr.open_zarr(url)
            logger.debug(f'Successfully opened {url}')
            return ds
        except Exception as e:
            # This has started (30/4/2025) raising exceptions
            # It has previously been fine.
            logger.warning(f'Failed to open {url}')
            logger.warning(e)
            retries += 1
            # Sleep 10s, then 20s... with 5s jitter.
            timeout = 1 * retries + random.uniform(-0.5, 0.5)
            logger.warning(f'sleeping for {timeout} s')
            await asyncio.sleep(timeout)
    raise Exception(f'failed to open {url} after {retries} retries')


def find_tgt_calcs(urls, chunks, variable='ta', max_zoom=10, dim='2d', end_time='2021-03-01 00:00'):
    """Loop over zoom, starting at highest, working out which chunks can be completed and need calculating

    src: higher zoom level
    tgt: lower zoom level"""
    tgt_calcs = {}
    tgt_ds = None
    for src_zoom in range(max_zoom, 0, -1):
        tgt_zoom = src_zoom - 1
        logger.info(f'Calculating jobs to do for z{tgt_zoom} {urls[tgt_zoom]}')
        if src_zoom == max_zoom:
            src_ds = asyncio.run(async_retry_open_zarr('http://hackathon-o.s3.jc.rl.ac.uk/' + urls[src_zoom]))
        else:
            src_ds = tgt_ds
        tgt_ds = asyncio.run(async_retry_open_zarr('http://hackathon-o.s3.jc.rl.ac.uk/' + urls[tgt_zoom]))
        if end_time:
            src_ds = src_ds.sel(time=slice(end_time))
            tgt_ds = tgt_ds.sel(time=slice(end_time))
        src_ncell = len(src_ds.cell)
        tgt_ncell = len(tgt_ds.cell)

        src_chunksize = chunks[src_zoom]
        tgt_chunksize = chunks[tgt_zoom]
        if dim == '2d':
            src_complete_size = src_ncell / src_chunksize[1]
            tgt_complete_size = tgt_ncell / tgt_chunksize[1]
        elif dim == '3d':
            npressure = 25
            src_complete_size = npressure / src_chunksize[1] * src_ncell / src_chunksize[2]
            tgt_complete_size = npressure / tgt_chunksize[1] * tgt_ncell / tgt_chunksize[2]
        else:
            raise Exception(f'unknown dim {dim}')

        assert tgt_chunksize[0] % src_chunksize [0] == 0, 'target chunksize 0 does not exactly divide source'
        factor = tgt_chunksize[0] // src_chunksize[0]

        if src_zoom == max_zoom:
            src_completed = find_completed_time_chunks(src_ds, src_chunksize, src_complete_size, src_zoom, urls, variable)
        else:
            src_completed = tgt_will_be_complete

        tgt_completed = find_completed_time_chunks(tgt_ds, tgt_chunksize, tgt_complete_size, tgt_zoom, urls, variable)
        logger.debug(f'already completed: {tgt_completed.sum()}, remaining: {(~tgt_completed).sum()}')
        tgt_can_complete = find_tgt_can_complete(factor, src_completed, tgt_completed)
        logger.debug(f'can complete by coarsening: {tgt_can_complete.sum()}, cannot complete: {(~tgt_can_complete).sum()}')
        logger.trace(f'cannot complete idx: {np.where(~tgt_can_complete)}')

        tgt_calc = tgt_can_complete & ~tgt_completed
        logger.debug(f'adding: {tgt_calc.sum()} new calcs')
        tgt_will_be_complete = tgt_completed | tgt_can_complete
        logger.debug(f'will be complete: {tgt_will_be_complete.sum()}, remaining: {(~tgt_will_be_complete).sum()}')
        if not tgt_will_be_complete.any():
            logger.debug(f'No target chunks are completable at {tgt_zoom}')
            break
        tgt_calcs[tgt_zoom] = tgt_calc
    return tgt_calcs


def find_tgt_time_calcs(tgt_calcs, chunks):
    tgt_time_calcs = {}
    for zoom, tgt_calc in tgt_calcs.items():
        calcs = []
        logger.info(f'Calc time indices for {zoom}')
        time_chunk = chunks[zoom][0]
        tgt_chunk_idx = np.arange(len(tgt_calc))
        for can_calc, chunk_idx in zip(tgt_calc, tgt_chunk_idx):
            if can_calc:
                calcs.append({'start_idx': int(time_chunk * chunk_idx), 'end_idx': int(time_chunk * (chunk_idx + 1))})
        logger.info(f'Adding {len(calcs)} calcs')
        tgt_time_calcs[zoom] = calcs
    return tgt_time_calcs


if __name__ == '__main__':
    config_key = sys.argv[1]
    dim = sys.argv[2]
    freqs = {
        '2d': 'PT1H',
        '3d': 'PT3H',
    }
    # Last variables for each.
    variables = {
        '2d': 'rsutcs',
        '3d': 'qs',
    }
    config = processing_config[config_key]
    rel_url_tpl = config['zarr_store_url_tpl'][5:]  # chop off 's3://'
    freq = freqs[dim]
    urls = {
        z: rel_url_tpl.format(freq=freq, zoom=z)
        for z in range(11)
    }

    chunks = config['groups'][dim]['chunks']
    max_zoom = config['max_zoom']
    variable = variables[dim]

    tgt_calcs = find_tgt_calcs(urls, chunks=chunks, variable=variable, max_zoom=max_zoom, dim=dim)
    tgt_time_calcs = find_tgt_time_calcs(tgt_calcs, chunks=chunks)

