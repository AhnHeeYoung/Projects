"""run_infer.py

Usage:
  run_infer.py [options] [--help] <command> [<args>...]
  run_infer.py --version
  run_infer.py (-h | --help)

Options:
  -h --help                   Show this string.
  --version                   Show version.

  --gpu=<id>                  GPU list. [default: 0]
  --nr_types=<n>              Number of nuclei types to predict. [default: 0]
  --type_info_path=<path>     Path to a json define mapping between type id, type name, 
                              and expected overlaid color. [default: type_info.json]

  --model_path=<path>         Path to saved checkpoint.
  --model_mode=<mode>         Original HoVer-Net or the reduced version used PanNuke and MoNuSAC, 
                              'original' or 'fast'. [default: fast]
  --nr_inference_workers=<n>  Number of workers during inference. [default: 8]
  --nr_post_proc_workers=<n>  Number of workers during post-processing. [default: 16]
  --batch_size=<n>            Batch size per 1 GPU. [default: 32]

Two command mode are `tile` and `wsi` to enter corresponding inference mode
    tile  run the inference on tile
    wsi   run the inference on wsi

Use `run_infer.py <command> --help` to show their options and usage.
"""

tile_cli = """
Arguments for processing tiles.

usage:
    tile (--input_dir=<path>) (--output_dir=<path>) \
         [--draw_dot] [--save_qupath] [--save_raw_map] [--mem_usage=<n>]
    
options:
   --input_dir=<path>     Path to input data directory. Assumes the files are not nested within directory.
   --output_dir=<path>    Path to output directory..

   --mem_usage=<n>        Declare how much memory (physical + swap) should be used for caching. 
                          By default it will load as many tiles as possible till reaching the 
                          declared limit. [default: 0.2]
   --draw_dot             To draw nuclei centroid on overlay. [default: True]
   --save_qupath          To optionally output QuPath v0.2.3 compatible format. [default: True]
   --save_raw_map         To save raw prediction or not. [default: True]
"""

wsi_cli = """
Arguments for processing wsi

usage:
    wsi (--input_dir=<path>) (--output_dir=<path>) [--proc_mag=<n>]\
        [--cache_path=<path>] [--input_mask_dir=<path>] \
        [--ambiguous_size=<n>] [--chunk_shape=<n>] [--tile_shape=<n>] \
        [--save_thumb] [--save_mask]
    
options:
    --input_dir=<path>      Path to input data directory. Assumes the files are not nested within directory.
    --output_dir=<path>     Path to output directory.
    --cache_path=<path>     Path for cache. Should be placed on SSD with at least 100GB. [default: cache]
    --mask_dir=<path>       Path to directory containing tissue masks. 
                            Should have the same name as corresponding WSIs. [default: '']

    --proc_mag=<n>          Magnification level (objective power) used for WSI processing. [default: 40]
    --ambiguous_size=<int>  Define ambiguous region along tiling grid to perform re-post processing. [default: 128]
    --chunk_shape=<n>       Shape of chunk for processing. [default: 3000]
    --tile_shape=<n>        Shape of tiles for processing. [default: 270]
    --save_thumb            To save thumb. [default: True]
    --save_mask             To save mask. [default: True]
"""

import torch
import logging
import os
import copy
from misc.utils import log_info
from docopt import docopt
import glob
import cv2
import json
import numpy as np
import scipy.io as sio
import openslide

from misc.wsi_handler import get_file_handler
from misc.viz_utils import visualize_instances_dict


#-------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    sub_cli_dict = {'tile' : tile_cli, 'wsi' : wsi_cli}
    args = docopt(__doc__, help=False, options_first=True, 
                    version='HoVer-Net Pytorch Inference v1.0')
    sub_cmd = args.pop('<command>')
    sub_cmd_args = args.pop('<args>')

    # ! TODO: where to save logging
    logging.basicConfig(
        level=logging.INFO,
        format='|%(asctime)s.%(msecs)03d| [%(levelname)s] %(message)s',datefmt='%Y-%m-%d|%H:%M:%S',
        handlers=[
            logging.FileHandler("debug.log"),
            logging.StreamHandler()
        ]
    )

    if args['--help'] and sub_cmd is not None:
        if sub_cmd in sub_cli_dict: 
            print(sub_cli_dict[sub_cmd])
        else:
            print(__doc__)
        exit()
    if args['--help'] or sub_cmd is None:
        print(__doc__)
        exit()

    sub_args = docopt(sub_cli_dict[sub_cmd], argv=sub_cmd_args, help=True)
    
    args.pop('--version')
    gpu_list = args.pop('--gpu')
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list

    nr_gpus = torch.cuda.device_count()
    log_info('Detect #GPUS: %d' % nr_gpus)

    args = {k.replace('--', '') : v for k, v in args.items()}
    sub_args = {k.replace('--', '') : v for k, v in sub_args.items()}
    if args['model_path'] == None:
        raise Exception('A model path must be supplied as an argument with --model_path.')

    nr_types = int(args['nr_types']) if int(args['nr_types']) > 0 else None
    method_args = {
        'method' : {
            'model_args' : {
                'nr_types'   : nr_types,
                'mode'       : args['model_mode'],
            },
            'model_path' : args['model_path'],
        },
        'type_info_path'  : None if args['type_info_path'] == '' \
                            else args['type_info_path'],
    }

    # ***
    run_args = {
        'batch_size' : int(args['batch_size']) * nr_gpus,

        'nr_inference_workers' : int(args['nr_inference_workers']),
        'nr_post_proc_workers' : int(args['nr_post_proc_workers']),
    }

    if args['model_mode'] == 'fast':
        run_args['patch_input_shape'] = 256
        run_args['patch_output_shape'] = 164
    else:
        run_args['patch_input_shape'] = 270
        run_args['patch_output_shape'] = 80

    if sub_cmd == 'tile':
        run_args.update({
            'input_dir'      : sub_args['input_dir'],
            'output_dir'     : sub_args['output_dir'],

            'mem_usage'   : float(sub_args['mem_usage']),
            'draw_dot'    : True,
            'save_qupath' : True,
            'save_raw_map': True,
        })

    if sub_cmd == 'wsi':
        run_args.update({
            'input_dir'      : sub_args['input_dir'],
            'output_dir'     : sub_args['output_dir'],
            'input_mask_dir' : sub_args['input_mask_dir'],
            'cache_path'     : sub_args['cache_path'],

            'proc_mag'       : int(sub_args['proc_mag']),
            'ambiguous_size' : int(sub_args['ambiguous_size']),
            'chunk_shape'    : int(sub_args['chunk_shape']),
            'tile_shape'     : int(sub_args['tile_shape']),
            'save_thumb'     : True,
            'save_mask'      : True,
        })
    # ***
    
    if sub_cmd == 'tile':
        from infer.tile import InferManager
        infer = InferManager(**method_args)
        infer.process_file_list(run_args)
    else:
        from infer.wsi import InferManager
        infer = InferManager(**method_args)
        infer.process_wsi_list(run_args)
        
        
        
        
        
        
    #################################################
    ###############        Tile          ############
    #################################################
        
    if sub_cmd == 'tile':
          
        log_info(" Result Save .... ")
        
        tile_path = sub_args['input_dir'] + '/'
        tile_json_path = sub_args['output_dir'] + '/json/'
        tile_mat_path = sub_args['output_dir'] + '/mat/'
        tile_overlay_path = sub_args['output_dir'] + '/overlay/'

        

        image_list = glob.glob(tile_path + '*')
        image_list.sort()
        
        patient_name = [i.split('/')[-1].split('.')[0] for i in image_list]

        # get a random image 
        rand_nr = np.random.randint(0,len(image_list))
        rand_nr = 0
        
        for i in range(len(image_list)):

            image_file = image_list[i]

            basename = os.path.basename(image_file)
            image_ext = basename.split('.')[-1]
            basename = basename[:-(len(image_ext)+1)]



            image = cv2.imread(image_file)
            # convert from BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


            # get the corresponding `.mat` file 
            result_mat = sio.loadmat(tile_mat_path + basename + '.mat')

            # get the overlay
            overlay = cv2.imread(tile_overlay_path + basename + '.png')
            overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            inst_map = result_mat['inst_map']


            json_path = tile_json_path + basename + '.json'

            bbox_list = []
            centroid_list = []
            contour_list = [] 
            type_list = []

            with open(json_path) as json_file:
                data = json.load(json_file)
                mag_info = data['mag']
                nuc_info = data['nuc']
                for inst in nuc_info:
                    inst_info = nuc_info[inst]
                    inst_centroid = inst_info['centroid']
                    centroid_list.append(inst_centroid)
                    inst_contour = inst_info['contour']
                    contour_list.append(inst_contour)
                    inst_bbox = inst_info['bbox']
                    bbox_list.append(inst_bbox)
                    inst_type = inst_info['type']
                    type_list.append(inst_type)





            if int(args['nr_types']) != 0:

                x_tile = 0
                y_tile = 0
                w_tile = image.shape[1]
                h_tile = image.shape[0]
                # load the wsi object and read region
                #wsi_obj = get_file_handler(wsi_file, wsi_ext)
                #wsi_obj.prepare_reading(read_mag=mag_info)
                #wsi_tile = wsi_obj.read_region((x_tile,y_tile), (w_tile,h_tile))

                # only consider results that are within the tile

                coords_xmin = x_tile
                coords_xmax = x_tile + w_tile
                coords_ymin = y_tile
                coords_ymax = y_tile + h_tile





                tile_info_dict = {}
                count = 0
                for idx, cnt in enumerate(contour_list):
                    cnt_tmp = np.array(cnt)
                    cnt_tmp = cnt_tmp[(cnt_tmp[:,0] >= coords_xmin) & (cnt_tmp[:,0] <= coords_xmax) & (cnt_tmp[:,1] >= coords_ymin) & (cnt_tmp[:,1] <= coords_ymax)] 
                    label = str(type_list[idx])
                    if cnt_tmp.shape[0] > 0:
                        cnt_adj = np.round(cnt_tmp - np.array([x_tile,y_tile])).astype('int')
                        tile_info_dict[idx] = {'contour': cnt_adj, 'type':label}
                        count += 1


                # plot the overlay

                # the below dictionary is specific to PanNuke checkpoint - will need to modify depeending on categories used
                type_info = {
                    "0" : ["nolabe", [0  ,   0,   0]], 
                    "1" : ["neopla", [255,   0,   0]], 
                    "2" : ["inflam", [0  , 255,   0]], 
                    "3" : ["connec", [0  ,   0, 255]], 
                    "4" : ["necros", [255, 255,   0]], 
                    "5" : ["no-neo", [255, 165,   0]] 
                }

                overlaid_output = visualize_instances_dict(image, tile_info_dict, type_colour=type_info)

                cv2.imwrite(sub_args['output_dir'] + '/result_overlay_classification_map_{}_NucleiNumber{}.jpg'.format(patient_name[i], len(centroid_list)), cv2.cvtColor(overlaid_output, cv2.COLOR_BGR2RGB))




            cv2.imwrite(sub_args['output_dir'] + '/result_inst_map_{}_NucleiNumber{}.jpg'.format(patient_name[i], len(centroid_list)), inst_map)
            cv2.imwrite(sub_args['output_dir'] + '/result_overlay_map_{}_NucleiNumber{}.jpg'.format(patient_name[i], len(centroid_list)), cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))

            log_info("Instance map  Save Complete")
            log_info("Overlay map  Save Complete")

            log_info("{} Number of Nuclei : {} ".format(patient_name[i], len(centroid_list)))
        
        
        
        
        
        
        
    #################################################
    ###############        WSI          ############
    #################################################        
        
    if sub_cmd == 'wsi':

        wsi_path = sub_args['input_dir'] + '/'
        wsi_json_path = sub_args['output_dir'] + '/'

        # get the list of all wsis
        wsi_list = glob.glob(wsi_path + '*')

        # get a random wsi from the list
        rand_wsi = np.random.randint(0,len(wsi_list))
        wsi_file = wsi_list[rand_wsi]
        wsi_ext = '.svs'

        wsi_basename = os.path.basename(wsi_file)
        wsi_basename = wsi_basename[:-(len(wsi_ext))]
    
    
        temp = openslide.OpenSlide(wsi_list[0])

        image = temp.read_region((0, 0), 0, (temp.dimensions[0], temp.dimensions[1]))
        image = np.array(image)[:, :, :3]
    
    
        patient_name = [i.split('/')[-1].split('.')[0] for i in wsi_list]
    
        # load the json file (may take ~20 secs)
        json_path_wsi = wsi_json_path + 'json/' + wsi_basename + '.json'

        bbox_list_wsi = []
        centroid_list_wsi = []
        contour_list_wsi = [] 
        type_list_wsi = []

        # add results to individual lists
        with open(json_path_wsi) as json_file:
            data = json.load(json_file)
            mag_info = data['mag']
            nuc_info = data['nuc']
            for inst in nuc_info:
                inst_info = nuc_info[inst]
                inst_centroid = inst_info['centroid']
                centroid_list_wsi.append(inst_centroid)
                inst_contour = inst_info['contour']
                contour_list_wsi.append(inst_contour)
                inst_bbox = inst_info['bbox']
                bbox_list_wsi.append(inst_bbox)
                inst_type = inst_info['type']
                type_list_wsi.append(inst_type)
    
    
        overlay = image.copy()
        for i in range(len(centroid_list_wsi)):

            contour_coord = contour_list_wsi[i]
            overlay = cv2.drawContours(overlay.astype('uint8'), [np.array(contour_coord)], -1, (0, 0, 0), 2)
    
        
        
        if int(args['nr_types']) != 0:
        
            # let's generate a tile from the WSI

            # define the region to select
            x_tile = 0
            y_tile = 0
            w_tile = temp.dimensions[0]
            h_tile = temp.dimensions[1]
            # load the wsi object and read region
            wsi_obj = get_file_handler(wsi_file, wsi_ext)
            wsi_obj.prepare_reading(read_mag=mag_info)
            wsi_tile = wsi_obj.read_region((x_tile,y_tile), (w_tile,h_tile))

            # only consider results that are within the tile

            coords_xmin = x_tile
            coords_xmax = x_tile + w_tile
            coords_ymin = y_tile
            coords_ymax = y_tile + h_tile

            tile_info_dict = {}
            count = 0
            for idx, cnt in enumerate(contour_list_wsi):
                cnt_tmp = np.array(cnt)
                cnt_tmp = cnt_tmp[(cnt_tmp[:,0] >= coords_xmin) & (cnt_tmp[:,0] <= coords_xmax) & (cnt_tmp[:,1] >= coords_ymin) & (cnt_tmp[:,1] <= coords_ymax)] 
                label = str(type_list_wsi[idx])
                if cnt_tmp.shape[0] > 0:
                    cnt_adj = np.round(cnt_tmp - np.array([x_tile,y_tile])).astype('int')
                    tile_info_dict[idx] = {'contour': cnt_adj, 'type':label}
                    count += 1


            # plot the overlay

            # the below dictionary is specific to PanNuke checkpoint - will need to modify depeending on categories used
            type_info = {
                "0" : ["nolabe", [0  ,   0,   0]], 
                "1" : ["neopla", [255,   0,   0]], 
                "2" : ["inflam", [0  , 255,   0]], 
                "3" : ["connec", [0  ,   0, 255]], 
                "4" : ["necros", [255, 255,   0]], 
                "5" : ["no-neo", [255, 165,   0]] 
            }

            overlaid_output = visualize_instances_dict(wsi_tile, tile_info_dict, type_colour=type_info)
            cv2.imwrite(sub_args['output_dir'] + '/result_overlay_classification_map.jpg', cv2.cvtColor(overlaid_output, cv2.COLOR_BGR2RGB))
    
        
        cv2.imwrite(sub_args['output_dir'] + '/result_overlay_map.jpg', cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    
        log_info('Number of Nuclei : {}'.format(len(contour_list_wsi)))
    