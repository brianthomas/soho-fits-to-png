
from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits

from dask import delayed
from dask.distributed import Client

from PIL import Image

import logging
import matplotlib.pyplot as plt
import numpy as np

import os
from pathlib import Path

DEF_NUM_THREADS = 4

LOG = logging.getLogger('soho-fits-to-png')
logging.basicConfig(level=logging.INFO)

# returns numpy data array
def get_numpy_data(file_name:str):
    try :

        image_file = get_pkg_data_filename(file_name)
        LOG.debug(fits.info(image_file))

        image_data = fits.getdata(image_file, ext=0,)

        # deal with zeros and negative values in data
        zero_threshold_indices = image_data <= 0.
        image_data[zero_threshold_indices] = 1.e-20

        LOG.debug("Image Shape : ", image_data.shape)

    except :

        LOG.debug("Error opening file {}".format(file_name))
        image_data = None

    return image_data

def create_png(image_data:np.array, outdir:str="/tmp", fileout:str="test.png", overwrite:bool=False)->bool:
    """ Create png. Returns True if it thinks its successful.
    """

    LOG.debug("writing file {} to {}".format(fileout, outdir))
    outfile = os.path.join(outdir, fileout)

    if(image_data is not None) :

        # make the output directory if its not already there
        if not os.path.exists(outdir):
            # it is always OK to create an empty target directory if none exists
            os.makedirs(outdir, exist_ok=True)
                
        if os.path.exists(outfile) and not overwrite:
            LOG.debug(f"Can't write {outfile} -- file already exists.") 
            return False

        #plt.figure()
        plt.imsave(outfile, image_data, cmap='magma')

    else :

        LOG.fatal(f"Can't write {outfile}. Empty array passed, bad pipeline configuration?") 
        return False

    return True


def check_directory(dirname: str) -> bool:
    if not os.path.isdir(dirname):
        LOG.fatal(f" {dirname} directory does not exist (or is not a directory), please fix")
        return False

    # is location writable?
    if not os.access(dirname, os.W_OK):
        LOG.fatal(f" {dirname} directory is not writable")
        return False

    return True


def create_output_filename(file_in:str):
    p = Path(file_in)
    basename = os.path.basename(p).replace('.fits', '')
    file_out = basename.split('\.')[0] + ".png"
    return file_out


def find_files(dirname: str, base_output_dir:os.PathLike, extension:str="fits") -> dict:
    '''
    Locate files in subdirectories and bundle into a dictionary of lists.
    '''
    files_to_process = {}

    for root, dirs, files in os.walk(dirname):
        loc = Path(os.path.join(base_output_dir, root))
        file_list = []
        for f in files:
            if f.endswith(extension):
                file_list.append(os.path.join(root, f))

        if len(file_list) > 0:
            files_to_process[loc] = file_list

    return files_to_process



def process_file(output_path:str, file:str, overwrite:bool=False):
    """
    This function will be run in parallel ...
    """
    img_data = get_numpy_data(file)
    out_filename = create_output_filename(file)

    success = create_png(img_data, outdir=output_path, fileout=out_filename, overwrite=overwrite)

    return 1 if success else 0


def create_jobs (files_to_process:dict, num_of_threads:int=DEF_NUM_THREADS, overwrite:bool=False, file_limit:int=None)->list:
    """
    Create a list of jobs to process fits files to png.
    """

    cnt = 0
    dask_processing_list = []
    for output_path, flist in files_to_process.items():
        for f in flist:
            if file_limit != None and cnt >= file_limit:
                break

            LOG.debug(f" File to process:{f} out:{output_path}")
            dask_processing_list.append(delayed(process_file)(output_path, f, overwrite))

        cnt += 1

    return dask_processing_list


if __name__ == '__main__':

    import time
    import argparse

    ap = argparse.ArgumentParser(description='Pipeline to transform SOHO/LASCO FITS difference images to PNG.')
    ap.add_argument('-d', '--debug', default=False, action='store_true', help='Turn on debugging messages')
    ap.add_argument('-t', '--num_threads', type=int, help=f'Number of threads to use. Default:{DEF_NUM_THREADS}',
                    default=DEF_NUM_THREADS)
    ap.add_argument('-ow', '--overwrite', default=False, action='store_true',
                    help='Overwrite existing data locally with new files.')
    ap.add_argument('-i', '--input_dir', help='Directory/path to input data from')
    ap.add_argument('-o', '--output_dir', help='Directory/path to output processed data to')

    args = ap.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
        LOG.setLevel(logging.DEBUG)

    # validation checks
    # check the input/output directories
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not check_directory(input_dir) or not check_directory(output_dir):
        exit()

    start_time = time.time()

    # find files to process in input dir
    found_files_to_process = find_files(input_dir, output_dir)
    LOG.debug(f"files to process: {found_files_to_process}")

    #client = Client(threads_per_worker=num_of_threads, n_workers=1) 
    #print(client)

    # assemble the dask jobs
    process_list = create_jobs (found_files_to_process, args.num_threads, args.overwrite)
    if len(process_list) > 0:
        process_list_sum = sum(process_list)
        process_list_sum.compute()

    LOG.info(f"\nTotal Process time: %8.2f sec" % (time.time() - start_time))

