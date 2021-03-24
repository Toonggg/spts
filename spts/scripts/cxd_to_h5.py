#!/usr/local/bin/python3
import argparse
import os,sys
import olefile
import numpy as np

from scipy.ndimage import percentile_filter 

import sys
print(sys.path)

import h5writer
import spts 
import spts.camera
from spts.camera import CXDReader

def cxd_to_h5(filename_cxd, filename_bg_cxd, filename_cxi, Nbg_max, filt_percent, filt_frames, cropping, minx, maxx, miny, maxy):

    # Initialise reader(s)
    # Data 
    print("Opening %s" % filename_cxd) 
    R = CXDReader(filename_cxd)     

    # Background
    Rbg = None 
    if filename_bg_cxd is not None: 
        if filename_bg_cxd == filename_cxd: 
            Rbg = R 
        else: 
            print("Opening %s" % filename_bg_cxd) 
            Rbg = CXDReader(filename_bg_cxd) 
        # Collect background stack 
        print("Collecting background frames...") 
        Nbg = min([Nbg_max, Rbg.get_number_of_frames()]) 
        for i in range(Nbg): 
            frame = Rbg.get_frame(i) # dtype: uint16 
            if i == 0: 
                if cropping:  
                    shape = (Nbg, (maxy-miny), (maxx-minx))
                    bg_stack = np.zeros(shape, dtype=frame.dtype) # cropped background stack 
                    print("Cropping raw images...") 
                else: 
                    shape = (Nbg, frame.shape[0], frame.shape[1])
                    bg_stack = np.zeros(shape, dtype=frame.dtype) # uncropped background stack
                    print("Not cropping raw images...") 
            if cropping: 
                bg_stack[i,:,:] = frame[miny:maxy, minx:maxx]   
            else: 
                bg_stack[i,:,:] = frame 
            print('(%d/%d) filling buffer for background estimation ...' % (i+1, Nbg)) 

        # Calculate median from background stack => background image 
        print("Calculating background estimate by median of buffer... ") 
        bg = percentile_filter(bg_stack, filt_percent, size = (filt_frames, 1, 1))  
    else: 
        bg = None 

    # Initialise output CXI file 
    W = h5writer.H5Writer(filename_cxi) 

    # Initialise integration variables
    integrated_raw = None 
    integrated_image = None 
    integratedsq_raw = None 
    integratedsq_image = None 

    # Write frames
    N  = R.get_number_of_frames() 
    for i in range(N):

        print('(%d/%d) Writing frame ...' % (i+1, N)) 

        frame = R.get_frame(i) 
        if cropping: 
            image_raw = frame[miny:maxy, minx:maxx] 
        else: 
            image_raw = frame 

        out = {}
        out["entry_1"] = {}

        # Raw data 
        out["entry_1"]["data_1"] = {"data": image_raw}

        # Background-subtracted image 
        if bg is not None:
            image_bgcor = np.ndarray.astype(frame[miny:maxy, minx:maxx], dtype = 'float32') - np.ndarray.astype(bg[i, :, :], dtype = 'float32') 
            out["entry_1"]["image_1"] = {"data": image_bgcor} 

        # Write to disc
        W.write_slice(out)
            
        if integrated_raw is None:
            if cropping: 
                integrated_raw = np.zeros(shape=((maxy-miny), (maxx-minx)), dtype='float32')
            else:
                integrated_raw = np.zeros(shape=frame.shape, dtype='float32')
        if integratedsq_raw is None: 
            if cropping: 
                integratedsq_raw = np.zeros(shape=((maxy-miny), (maxx-minx)), dtype='float32')
            else:
                integratedsq_raw = np.zeros(shape=frame.shape, dtype='float32')
        integrated_raw += np.asarray(image_raw, dtype='float32') 
        integratedsq_raw += np.asarray(image_raw, dtype='float32')**2 

        if bg is not None: 
            if integrated_image is None: 
                if cropping:
                    integrated_image = np.zeros(shape=((maxy-miny), (maxx-minx)), dtype='float32') 
                else:     
                    integrated_image = np.zeros(shape=frame.shape, dtype='float32') 
            if integratedsq_image is None:
                if cropping:
                    integratedsq_image = np.zeros(shape=((maxy-miny), (maxx-minx)), dtype='float32') 
                else:
                    integratedsq_image = np.zeros(shape=frame.shape, dtype='float32') 
            integrated_image += image_bgcor 
            integratedsq_image += np.asarray(image_bgcor, dtype='f')**2 
        
    # Write integrated images 
    print('Writing integrated images...') 
    out = {"entry_1": {"data_1": {}, "image_1": {}}}
    if integrated_raw is not None: 
        out["entry_1"]["data_1"]["data_mean"] = integrated_raw / float(N)
    if integrated_image is not None:
        out["entry_1"]["image_1"]["data_mean"] = integrated_image / float(N)
    if integratedsq_raw is not None:
        out["entry_1"]["data_1"]["datasq_mean"] = integratedsq_raw / float(N)
    if integratedsq_image is not None:
        out["entry_1"]["image_1"]["datasq_mean"] = integratedsq_image / float(N)
    W.write_solo(out)

    print("Closing files...") 
    # Close CXI file 
    W.close()
    # Close readers 
    R.close()
    if Rbg is not None and not Rbg.is_closed():
        Rbg.close()
    print("Clean exit.")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Conversion of CXD (Hamamatsu file format) to HDF5') 
    parser.add_argument('filename', type=str, help='CXD filename') 
    parser.add_argument('-b','--background-filename', type=str, help='CXD filename with photon background data.') 

    parser.add_argument('-p', '--percentile-number', type = int, help='Percentile value for percentile filter.', default = 50) 
    parser.add_argument('-pf','--percentile-frames', type=int, help='Number of frames in kernel for percentile filter.', default = 4) 
    parser.add_argument('-n','--bg-frames-max', type=int, help='Maximum number of frames used for background calculation.', default = 500) 

    parser.add_argument('-crop', '--crop-raw', action = 'store_true', help = 'Enable cropping of raw images.') 
    parser.add_argument('-minx','--min-x', type=int, help='Minimum x-coordinate of cropped raw data.', default = 960) 
    parser.add_argument('-maxx','--max-x', type=int, help='Maximum x-coordinate of cropped raw data.', default = 1300) 
    parser.add_argument('-miny','--min-y', type=int, help='Minimum y-coordinate of cropped raw data.', default = 400) 
    parser.add_argument('-maxy','--max-y', type=int, help='Maximum y-coordinate of cropped raw data.', default = 900) 
    parser.add_argument('-o','--out-filename', type=str, help='destination file') 
    
    args = parser.parse_args()

    f = args.filename

    BackgroundFileExists = bool(args.background_filename)
    if BackgroundFileExists == False: 
        f_bg = None 
    elif BackgroundFileExists == True:
        f_bg = args.background_filename
    else: 
        f_bg = f[:-4] + "_bg.cxd"
        if not os.path.isfile(f_bg):
            print("WARNING: File with background frames not found in location %s.\n\tFalling back to determining background from the real data frames." % f_bg)
            f_bg = f

    if args.out_filename:
        f_out = args.out_filename
    else:
        f_out = f[:-4] + ".cxi"

    if not args.filename.endswith(".cxd"):
        print("ERROR: Given filename %s does not end with \".cxd\". Wrong format!" % args.filename)
    else:
        cxd_to_h5(f, f_bg, f_out, args.bg_frames_max, args.percentile_number, args.percentile_frames, args.crop_raw, args.min_x, args.max_x, args.min_y, args.max_y)
