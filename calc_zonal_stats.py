# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 14:00:03 2020

@author: disbr007
"""
import argparse
import json
import logging
import os
import matplotlib.pyplot as plt
import numpy as np
import sys

import pandas as pd
import geopandas as gpd
# import fiona
# import rasterio
from rasterstats import zonal_stats
from skimage.feature import greycomatrix, greycoprops

from lib import detect_ogr_driver, read_vec, write_gdf


logger = logging.getLogger(__name__)
logger.info('logging from zs')

# custom_stat_fxn = {
#     'glcm': calc_glcm
# }

# def calc_glcm(patch, distance = [5], angles=[0], levels=)

def load_stats_dict(stats_json):
    if isinstance(stats_json, str):
        if os.path.exists(stats_json):
            with open(stats_json) as jf:
                data = json.load(jf)
        else:
            logger.error('Zonal stats file not found: {}'.format(stats_json))
    elif isinstance(stats_json, dict):
        data = stats_json

    names = []
    rasters = []
    stats = []
    bands = []
    for n, d in data.items():
        names.append(n)
        rasters.append(d['path'])
        stats.append(d['stats'])
        if 'bands' in d.keys():
            bands.append(d['bands'])
        else:
            bands.append(None)

    return rasters, names, stats, bands


def calc_compactness(geometry):
    # Polsby - Popper Score - - 1 = circle
    compactness = (np.pi * 4 * geometry.area) / (geometry.boundary.length)**2
    return compactness


def apply_compactness(gdf, out_field='compactness'):
    gdf[out_field] = gdf.geometry.apply(lambda x: calc_compactness(x))
    return gdf


def calc_roundness(geometry):
    # Circularity = Perimeter^2 / (4 * pi * Area)
    roundess = (geometry.length**2 / (4 * np.pi * geometry.area))
    return roundess


def apply_roundness(gdf, out_field='roundness'):
    gdf[out_field] = gdf.geometry.apply(lambda x: calc_roundness(x))
    return gdf


def compute_stats(gdf, raster, name=None,
                  stats=None,
                  custom_stats=None, band=None,
                  renamer=None):
    """
    Computes statistics for each polygon in geodataframe
    based on raster. Statistics to be computed are the keys
    in the stats_dict, and the renamed columns are the values.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame of polygons to compute statistics over.
    raster : os.path.abspath | rasterio.raster
        Raster to compute statistics from.
    stats_dict : dict
        Dictionary of stat:renamed_col pairs.
        Stats must be one of: min, max, median, sum, std,
                              unique, range, percentile_<q>
    Returns
    -------
    The geodataframe with added columns.

    """
    if stats is None:
        stats = ['mean', 'min', 'max', 'std']
    logger.info('Computing {} on raster:\n\t{}'.format(' '.join(stats), raster))
    if renamer is None:
        renamer = {x: '{}_{}'.format(name, x) for x in stats}

    if band:
        logger.info('Band: {}'.format(band))
        gdf = gdf.join(pd.DataFrame(zonal_stats(gdf['geometry'], raster,
                                                stats=stats,
                                                add_stats=custom_stats,
                                                band=band))
                       .rename(columns=renamer),
                       how='left')
    else:
        stats_df = pd.DataFrame(zonal_stats(gdf['geometry'], raster,
                                                stats=stats,
                                                add_stats=custom_stats))
        # logger.info('Stats DF Cols: {}'.format(stats_df.columns))
        # logger.info('GDF cols: {}'.format(gdf.columns))
        gdf = gdf.join(stats_df.rename(columns=renamer), how='left')

    return gdf


def calc_zonal_stats(shp, rasters,
                     names=None,
                     stats=['min', 'max', 'mean', 'count', 'median'],
                     area=True,
                     compactness=False,
                     roundness=False,
                     out_path=None):
    """
    Calculate zonal statistics on the given vector file
    for each raster provided.

    Parameters
    ----------
    shp : os.path.abspath
        Vector file to compute zonal statistics for the features in.
    out_path : os.path.abspath
        Path to write vector file with computed stats. Default is to
        add '_stats' suffix before file extension.
    rasters : list or os.path.abspath
        List of rasters to compute zonal statistics for.
        Or path to .txt file of raster paths (one per line)
        or path to .json file of
            name: {path: /path/to/raster.tif, stats: ['mean']}.
        or dict of same format as json
    names : list
        List of names to use as prefixes for created stats. Order
        is order of rasters.
    stats : list, optional
        List of statistics to calculate. The default is None.
    area : bool
        True to also compute area of each feature in units of
        projection.
    compactness : bool
        True to also compute compactness of each object
    roundness : bool
        True to also compute roundess of each object

    Returns
    -------
    out_path.

    """
    # Load data
    if isinstance(shp, gpd.GeoDataFrame):
        seg = shp
    else:
        logger.info('Reading in segments from: {}...'.format(shp))
        seg = read_vec(shp)
    logger.info('Segments found: {:,}'.format(len(seg)))

    # Determine rasters input type
    # TODO: Fix logic here, what if a bad path is passed?
    if isinstance(rasters[0], dict):
            rasters, names, stats, bands = load_stats_dict(rasters[0])
    elif len(rasters) == 1:
        print(type(rasters))
        if os.path.exists(rasters[0]):
            logger.info('Reading raster file...')
            ext = os.path.splitext(rasters[0])[1]
            if ext == '.txt':
                # Assume text file of raster paths, read into list
                logger.info('Reading rasters from text file: '
                            '{}'.format(rasters[0]))
                with open(rasters[0], 'r') as src:
                    content = src.readlines()
                    rasters = [c.strip() for c in content]
                    rasters, names = zip(*(r.split("~") for r in rasters))
                    logger.info('Located rasters:'.format('\n'.join(rasters)))
                    for r, n in zip(rasters, names):
                        logger.info('{}: {}'.format(n, r))
                # Create list of lists of stats passed, one for each raster
                stats = [stats for i in range(len(rasters))]
            elif ext == '.json':
                logger.info('Reading rasters from json file:'
                            ' {}'.format(rasters[0]))
                rasters, names, stats, bands = load_stats_dict(rasters[0])
            else:
                # Raster paths directly passed
                stats = [stats for i in range(len(rasters))]

    # Confirm all rasters exist before starting
    for r in rasters:
        if not os.path.exists(r):
            logger.error('Raster does not exist: {}'.format(r))
            logger.error('FileNotFoundError')

    # Iterate rasters and compute stats for each
    for r, n, s, bs in zip(rasters, names, stats, bands):
        if bs is None:
            # Split custom stat functions from built-in options
            accepted_stats = ['min', 'max', 'median', 'sum', 'std', 'mean',
                              'unique', 'range', 'majority']
            stats_acc = [k for k in s if k in accepted_stats
                         or k.startswith('percentile_')]
            # Assume any key not in accepted_stats is a name:custom_fxn
            custom_stats = [k for k in stats if k not in accepted_stats]
            custom_stats_dict = {}
            # for cs in custom_stats:
            #     custom_stats[cs] = custom_stat_fxn(cs)

            seg = compute_stats(gdf=seg, raster=r, name=n,
                                stats=stats_acc)
        else:
            # Compute stats for each band
            for b in bs:
                stats_dict = {x: '{}b{}_{}'.format(n, b, x) for x in s}
                seg = compute_stats(gdf=seg, raster=r,
                                    stats=stats_dict,
                                    renamer=stats_dict,
                                    band=b)

    # Area recording
    if area:
        seg['area_zs'] = seg.geometry.area

    # Compactness: Polsby-Popper Score -- 1 = circle
    if compactness:
        seg = apply_compactness(seg)
        
    if roundness:
        seg = apply_roundness(seg)

    # Write segments with stats to new shapefile
    if not out_path:
        out_path = os.path.join(os.path.dirname(shp),
                       '{}_stats.shp'.format(os.path.basename(shp).split('.')[0]))
    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))
    logger.info('Writing segments with statistics to: {}'.format(out_path))
    # driver = auto_detect_ogr_driver(out_path, name_only=True)
    # seg.to_file(out_path, driver=driver)
    write_gdf(seg, out_path)

    return seg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_shp',
                        type=os.path.abspath,
                        help='Vector file to compute zonal statistics for the features in.')
    parser.add_argument('-o', '--out_path',
                        type=os.path.abspath,
                        help="""Path to write vector file with computed stats. Default is to
                                add '_stats' suffix before file extension.""")
    parser.add_argument('-r', '--rasters',
                        nargs='+',
                        type=os.path.abspath,
                        help="""List of rasters to compute zonal statistics 
                                for, or path to .txt file of raster paths 
                                (one per line) or path to .json file in format:
                                {"name": 
                                    {"path": "C:\\raster", 
                                     "stats": ["mean", "min"]}
                                }""")
    parser.add_argument('-n', '--names',
                        type=str,
                        nargs='+',
                        help="""List of names to use as prefixes for created stats fields.
                                Length must match number of rasters supplied. Order is
                                the order of the rasters to apply prefix names for. E.g.:
                                'ndvi' -> 'ndvi_mean', 'ndvi_min', etc.""")
    parser.add_argument('-s', '--stats',
                        type=str,
                        nargs='+',
                        default=['min', 'max', 'mean', 'count', 'median'],
                        help='List of statistics to compute.')
    parser.add_argument('-a', '--area',
                        action='store_true',
                        help='Use to compute an area field.')
    parser.add_argument('-c', '--compactness',
                        action='store_true',
                        help='Use to compute a compactness field.')
    parser.add_argument('-rd', '--roundness',
                        action='store_true',
                        help='Use to compute a roundness field.')

    args = parser.parse_args()

    calc_zonal_stats(shp=args.input_shp,
                     rasters=args.rasters,
                     names=args.names,
                     stats=args.stats,
                     area=args.area,
                     compactness=args.compactness,
                     roundness=args.roundness,
                     out_path=args.out_path)
    logger.info('Done.')