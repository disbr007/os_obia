import copy
import logging
import os
import pathlib
from pathlib import Path, PurePath
import subprocess
from subprocess import PIPE
import typing
from typing import Union

from osgeo import gdal, ogr
import geopandas as gpd
import rasterio as rio
from rasterio.features import shapes
from rasterio.fill import fillnodata
import rasterio.mask

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


def run_subprocess(command):
    proc = subprocess.Popen(command, stdout=PIPE, stderr=PIPE, shell=True)
    for line in iter(proc.stdout.readline, b''):  # replace '' with b'' for Python 3
        logger.info(line.decode())
    output, error = proc.communicate()
    logger.debug('Output: {}'.format(output.decode()))
    logger.debug('Err: {}'.format(error.decode()))


def clean4cmdline(command):
    # Remove whitespace, newlines
    command = command.replace('\n', ' ')
    command = ' '.join(command.split())
    return command


def create_grm_outname(img=None, out_seg=None, out_dir=None,
                       criterion='bs', threshold=None, niter=0,
                       speed=0, spectral=0.5, spatial=0.5,
                       out_format='vector', name_only=False):
    # Create output names as needed
    if out_seg is None:
        if out_dir is None:
            out_dir = os.path.dirname(img)
        out_name = os.path.basename(img).split('.')[0]
        out_name = '{}_{}t{}ni{}s{}spec{}spat{}.tif'.format(out_name, criterion,
                                                            str(threshold).replace('.', 'x'),
                                                            niter, speed,
                                                            str(spectral).replace('.', 'x'),
                                                            str(spatial).replace('.', 'x'))
        out_seg = os.path.join(out_dir, out_name)
    if name_only and out_format == 'vector':
        out_seg = out_seg.replace('tif', 'shp')

    return out_seg


def detect_ogr_driver(ogr_ds: str, name_only: bool = False) -> typing.Tuple[gdal.Driver, str]:
    """
    Autodetect the appropriate driver for an OGR datasource.

    Parameters
    ----------
    ogr_ds : OGR datasource
        Path to OGR datasource.
    name_only : bool
        True to return the name of the driver, else the ogr.Driver object will
        be return
    Returns
    -------
    OGR driver OR
    OGR driver, layer name

    """
    # Driver names
    FileGDB = 'FileGDB'
    OpenFileGDB = 'OpenFileGDB'
    # Suffixes
    GPKG = '.gpkg'
    SHP = '.shp'
    GEOJSON = '.geojson'
    GDB = '.gdb'

    supported_drivers = [gdal.GetDriver(i).GetDescription()
                         for i in range(gdal.GetDriverCount())]
    if FileGDB in supported_drivers:
        gdb_driver = FileGDB
    else:
        gdb_driver = OpenFileGDB

    # OGR driver lookup table
    driver_lut = {
        GEOJSON: 'GeoJSON',
        SHP: 'ESRI Shapefile',
        GPKG: 'GPKG',
        GDB: gdb_driver
                  }
    layer = None

    # Check if in-memory datasource
    if isinstance(ogr_ds, PurePath):
        ogr_ds = str(ogr_ds)
    if isinstance(ogr_ds, ogr.DataSource):
        driver = 'Memory'
    elif 'vsimem' in ogr_ds:
        driver = 'ESRI Shapefile'
    else:
        # Check if extension in look up table
        if GPKG in ogr_ds:
            drv_sfx = GPKG
            layer = Path(ogr_ds).stem
        elif GDB in ogr_ds:
            drv_sfx = GDB
            layer = Path(ogr_ds).stem
        else:
            drv_sfx = Path(ogr_ds).suffix

        if drv_sfx in driver_lut.keys():
            driver = driver_lut[drv_sfx]
        else:
            logger.warning("""Unsupported driver extension {}
                            Defaulting to 'ESRI Shapefile'""".format(drv_sfx))
            driver = driver_lut[SHP]

    logger.debug('Driver autodetected: {}'.format(driver))

    if not name_only:
        try:
            driver = ogr.GetDriverByName(driver)
        except ValueError as e:
           logger.error('ValueError with driver_name: {}'.format(driver))
           logger.error('OGR DS: {}'.format(ogr_ds))
           raise e

    return driver, layer


def read_vec(vec_path: str, **kwargs) -> gpd.GeoDataFrame:
    """
    Read any valid vector format into a GeoDataFrame
    """
    driver, layer = detect_ogr_driver(vec_path, name_only=True)
    if layer is not None:
        gdf = gpd.read_file(Path(vec_path).parent, layer=layer, driver=driver, **kwargs)
    else:
        gdf = gpd.read_file(vec_path, driver=driver, **kwargs)

    return gdf


def write_gdf(src_gdf, out_footprint, to_str_cols=None,
              out_format=None,
              nan_to=None,
              precision=None,
              overwrite=True,
              **kwargs):
    """
    Handles common issues with writing GeoDataFrames to a variety of formats,
    including removing datetimes, converting list/dict columns to strings,
    handling NaNs.
    date_format : str
        Use to convert datetime fields to string fields, using format provided
    TODO: Add different handling for different formats, e.g. does gpkg allow datetime/NaN?
    """
    # Drivers
    ESRI_SHAPEFILE = 'ESRI Shapefile'
    GEOJSON = 'GeoJSON'
    GPKG = 'GPKG'
    OPEN_FILE_GDB = 'OpenFileGDB'
    FILE_GBD = 'FileGDB'

    gdf = copy.deepcopy(src_gdf)

    if not isinstance(out_footprint, pathlib.PurePath):
        out_footprint = Path(out_footprint)

    # Format agnostic functions
    # Remove if exists and overwrite
    if out_footprint.exists():
        if overwrite:
            logger.warning('Overwriting existing file: '
                           '{}'.format(out_footprint))
            os.remove(out_footprint)
        else:
            logger.warning('Out file exists and overwrite not specified, '
                           'skipping writing.')
            return None

    # Round if precision
    if precision:
        gdf = gdf.round(decimals=precision)
    logger.debug('Writing to file: {}'.format(out_footprint))

    # Get driver and layer name. Layer will be none for non database formats
    driver, layer = detect_ogr_driver(out_footprint, name_only=True)
    if driver == ESRI_SHAPEFILE:
        # convert NaNs to empty string
        if nan_to:
            gdf = gdf.replace(np.nan, nan_to, regex=True)

    # Convert columns that store lists to strings
    if to_str_cols:
        for col in to_str_cols:
            logger.debug('Converting to string field: {}'.format(col))
            gdf[col] = [','.join(map(str, l)) if isinstance(l, (dict, list))
                        and len(l) > 0 else '' for l in gdf[col]]

    # Write out in format specified
    if driver in [ESRI_SHAPEFILE, GEOJSON]:
        if driver == GEOJSON:
            if gdf.crs != 4326:
                logger.warning('Attempting to write GeoDataFrame with non-WGS84 '
                               'CRS to GeoJSON. Reprojecting to WGS84.')
                gdf = gdf.to_crs('epsg:4326')
        gdf.to_file(out_footprint, driver=driver, **kwargs)
    elif driver in [GPKG, OPEN_FILE_GDB, FILE_GBD]:
        gdf.to_file(str(out_footprint.parent), layer=layer, driver=driver, **kwargs)
    else:
        logger.error('Unsupported driver: {}'.format(driver))


def rio_polygonize(img: str, out_vec: str = None, band: int = 1, mask_value=None):
    logger.info('Polygonizing: {}'.format(img))
    with rio.Env():
        with rio.open(img) as src:
            arr = src.read(band)
            src_crs = src.crs
            if mask_value is not None:
                mask = arr == mask_value
            else:
                mask = None
            results = ({'properties': {'raster_val': v},
                        'geometry': s}
                       for i, (s, v) in
                       enumerate(shapes(arr,
                                        mask=mask,
                                        transform=src.transform)))
    geoms = list(results)
    gdf = gpd.GeoDataFrame.from_features(geoms, crs=src_crs)
    if out_vec:
        logger.info('Writing polygons to: {}'.format(out_vec))
        write_gdf(gdf, out_vec)

    return gdf