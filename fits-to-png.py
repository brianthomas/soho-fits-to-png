import matplotlib.pyplot as plt
import numpy as np
#from PIL import Image

# Astropy library - pip install astropy worked for me with and Open-CE build
from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits

import logging
import os
from pathlib import Path

DEF_NUM_THREADS = 8

LOG = logging.getLogger('soho-fits-to-png')
logging.basicConfig(level=logging.INFO)

# returns numpy data array
def get_numpy_data(file_name:str):
    try :
        image_file = get_pkg_data_filename(file_name)
        LOG.debug(fits.info(image_file))
        image_data = fits.getdata(image_file, ext=0,)
        LOG.debug("Image Shape : ", image_data.shape)
    except :
        LOG.debug("Error opening file {}".format(file_name))
        image_data = None

    return image_data

def create_png(image_data:np.array, outdir:str="/tmp", fileout:str="test.png") :

    if(image_data is not None) :

        LOG.debug("writing file {} to {}".format(fileout, outdir))
        outfile = os.path.join(outdir, fileout)

        plt.figure()
        plt.imsave(outfile, image_data, cmap='magma')

    else :

        msg = "Can't write PNG. Empty array passed, bad pipeline configuration?"
        LOG.fatal(msg)
        raise Exception(msg)


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



from dask import delayed
def process_file(output_path, file):
    """
    This function will be run in parallel ...
    """
    rv = 0
    img_data = get_numpy_data(file)
    if (img_data is not None):
        out_filename = create_output_filename(file)
        create_png(img_data, outdir=output_path, fileout=out_filename)
        rv = 1

    return rv

def process_files(files_to_process:list, output_path:str, num_of_threads:int=DEF_NUM_THREADS, file_limit:int=None):
    """
    Process a list of files.
    """
    cnt = 0
    dask_processing_list = []
    for f in files_to_process:
        dask_processing_list.append(delayed(process_file)(output_path, f))

        if file_limit != None and cnt >= file_limit:
            break

        cnt += 1

    # trigger the jobs
    delayed_processing_sum = sum(dask_processing_list)
    delayed_processing_sum.compute()


if __name__ == '__main__':

    import time
    import argparse

    ap = argparse.ArgumentParser(description='Pipeline to transform SOHO/LASCO FITS difference images to PNG.')
    ap.add_argument('-d', '--debug', default=False, action='store_true', help='Turn on debugging messages')
    ap.add_argument('-t', '--num_threads', type=int, help=f'Number of threads to use. Default:{DEF_NUM_THREADS}',
                    default=DEF_NUM_THREADS)
    ap.add_argument('-o', '--overwrite', default=False, action='store_true',
                    help='Overwrite existing data locally with new files.')
    ap.add_argument('-input_dir', help='Directory/path to input data from')
    ap.add_argument('-output_dir', help='Directory/path to output processed data to')

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
    print (found_files_to_process)
    for output_path in found_files_to_process.keys():
        process_files(found_files_to_process, output_path, args.num_threads)

    LOG.info(f"\nTotal Process time: %8.2f sec" % (time.time() - start_time))

