#!/usr/bin/env python
# coding: utf-8

# In[1]:


from xml.etree.ElementTree import parse
import numpy as np
#from Class_ID_Name import *
import os
import openslide

from PIL import Image

import glob

import cv2 as cv2

import torch
import torchvision
from torchvision import transforms

from torch.utils.data import Dataset, DataLoader

from shapely.geometry import Polygon
from skimage import draw

import matplotlib.pyplot as plt

import pickle

import argparse

import warnings

import logging

warnings.filterwarnings('ignore')




parser = argparse.ArgumentParser(description="Deep Multi-Patch Hierarchical Network")
parser.add_argument("-n","--num",type = int, default = 1)
parser.add_argument("-p","--patch_size",type = int, default = 1024)
args = parser.parse_args()

num = args.num
patch_size = args.patch_size



class Annotation():
    def __init__(self):
        self.Class=[]
        self.Name = []
        self.Type = []
        self.Color = []
        self.Group = []
        self.x_s=[]
        self.y_s=[]
        self.points=[]
        self.min_x=[]
        self.max_x=[]
        self.min_y=[]
        self.max_y=[]


def xml_aperio_name_class(annot_path, annot_c):

    tree = parse(annot_path)
    root = tree.getroot()
    Annotations = root.findall('Annotation')

    if os.path.basename(annot_path)[0:2] == 'No':
        Class_Name = 'Metastasis_No'
    else:
        Class_Name = 'Metastasis_Yes'

    for Annotation in Annotations:
        regionlist = Annotation.find('Regions').findall('Region')

        # Cancer_type = Annotation.attrib['Name']
        # Cancer_type = Annotation.find('Attributes').find('Attribute').attrib['Name']
        # assert Cancer_type in CLASS_ID_NAME_Samsung.keys()
        for region in regionlist:
            vertices = region.find('Vertices').findall('Vertex')
            region_X = []
            region_Y = []
            region_Points = []
            if len(vertices) > 0:
                for vertex in vertices:
                    x = int(round(float(vertex.attrib['X']) ))
                    y = int(round(float(vertex.attrib['Y']) ))
                    region_X.append(x)
                    region_Y.append(y)
                    region_Points.append([x, y])

                annot_c.x_s.append(region_X)
                annot_c.y_s.append(region_Y)

                annot_c.min_x.append(min(region_X))
                annot_c.max_x.append(max(region_X))
                annot_c.min_y.append(min(region_Y))
                annot_c.max_y.append(max(region_Y))

                annot_c.points.append(region_Points)
                annot_c.Class.append(Class_Name)

    return annot_c




def xml_aperio_multi_class(annot_path):

    tree = parse(annot_path)
    root = tree.getroot()
    Annotations = root.findall('Annotation')
    Annotation_X = []
    Annotation_Y = []
    class_ids=[]
    temp_vertex_count = 0
    for Annotation in Annotations:
        regionlist = Annotation.find('Regions').findall('Region')
        Cancer_type = Annotation.attrib['Name']
        # Cancer_type = Annotation.find('Attributes').find('Attribute').attrib['Name']
        assert Cancer_type in CLASS_ID_NAME_Samsung.keys()
        for region in regionlist:
            vertices = region.find('Vertices').findall('Vertex')
            region_X = []
            region_Y = []
            if len(vertices) > 0:
                for vertex in vertices:
                    x = int(round(float(vertex.attrib['X']) ))
                    y = int(round(float(vertex.attrib['Y']) ))
                    region_X.append(x)
                    region_Y.append(y)
                Annotation_X.append(region_X)
                Annotation_Y.append(region_Y)
                class_ids.append(CLASS_ID_NAME_Samsung[Cancer_type])

    return Annotation_X, Annotation_Y, class_ids



def xml_aperio_single_class(annot_path,seg_name):

    assert seg_name in CLASS_ID_NAME_CNN3.keys()
    tree = parse(annot_path)
    root = tree.getroot()
    Annotations = root.findall('Annotation')
    Annotation_X = []
    Annotation_Y = []
    class_ids=[]
    temp_vertex_count = 0
    for Annotation in Annotations:
        regionlist = Annotation.find('Regions').findall('Region')

        for region in regionlist:
            vertices = region.find('Vertices').findall('Vertex')
            region_X = []
            region_Y = []
            if len(vertices) > 0:
                for vertex in vertices:
                    x = int(round(float(vertex.attrib['X']) ))
                    y = int(round(float(vertex.attrib['Y']) ))
                    region_X.append(x)
                    region_Y.append(y)
                Annotation_X.append(region_X)
                Annotation_Y.append(region_Y)
                class_ids.append(CLASS_ID_NAME_CNN3[seg_name])

    return Annotation_X, Annotation_Y, class_ids


def Parse_Qupath_Annotation(annotation_fn,annot_c):

    def Points_Split(temp_points_str):
        temp_points_str = temp_points_str.replace('[','')
        temp_points_str = temp_points_str.replace(']','')
        temp_points_str = temp_points_str.split('Point:')[1:]
        return temp_points_str

    lines = [line.rstrip('\n') for line in open(annotation_fn)]
    img_path = lines[0]
    for i in range(1,len(lines),2):
        annot_c.Class.append(lines[i])
        temp_points_str = Points_Split(lines[i+1])
        temp_x_s = []
        temp_y_s = []
        temp_points = []
        for ii in range(0,len(temp_points_str)):
            temp_x = round(float(temp_points_str[ii].split(',')[0]))
            temp_y = round(float(temp_points_str[ii].split(',')[1]))
            temp_x_s.append(temp_x)
            temp_y_s.append(temp_y)
            temp_points.append([temp_x,temp_y])

        annot_c.x_s.append(temp_x_s)
        annot_c.y_s.append(temp_y_s)
        annot_c.min_x.append(min(temp_x_s))
        annot_c.max_x.append(max(temp_x_s))
        annot_c.min_y.append(min(temp_y_s))
        annot_c.max_y.append(max(temp_y_s))
        annot_c.points.append(temp_points)

    return annot_c, img_path



    return annot_c

def ACDC_S1_XML_Parsing(annot_path, annot_c):

    tree = parse(annot_path)
    root = tree.getroot()
    Annotations = root.find('Annotations').findall('Annotation')

    temp_vertex_count = 0
    for Annotation in Annotations:
        Cancer_type = Annotation.attrib['Name']
        if Cancer_type == 'Annot_ROI':
            continue
        region_X = []
        region_Y = []
        region_Points = []
        Coords = Annotation.find('Coordinates').findall('Coordinate')
        for Coord in Coords:

            x = int(round(float(Coord.attrib['X'])))
            y = int(round(float(Coord.attrib['Y'])))
            region_X.append(x)
            region_Y.append(y)
            region_Points.append([x, y])
        
        
        annot_c.x_s.append(region_X)
        annot_c.y_s.append(region_Y)
        annot_c.min_x.append(min(region_X))
        annot_c.max_x.append(max(region_X))
        annot_c.min_y.append(min(region_Y))
        annot_c.max_y.append(max(region_Y))
        annot_c.points.append(region_Points)

        annot_c.Class.append('Cancer') ##CLASS_ID_NAME_ACDC_S1

    return annot_c


def ACDC_S1_XML_Parsing_Tiling(annot_path, annot_c):

    tree = parse(annot_path)
    root = tree.getroot()
    Annotations = root.find('Annotations').findall('Annotation')

    temp_vertex_count = 0
    for Annotation in Annotations:
        Name = Annotation.attrib['Name']
        Type = Annotation.attrib['Type']
        Color = Annotation.attrib['Color']
        Group = Annotation.attrib['PartOfGroup']

        region_X = []
        region_Y = []
        region_Points = []
        Coords = Annotation.find('Coordinates').findall('Coordinate')
        for Coord in Coords:

            x = int(round(float(Coord.attrib['X'])))
            y = int(round(float(Coord.attrib['Y'])))
            region_X.append(x)
            region_Y.append(y)
            region_Points.append([x, y])
        
        
        annot_c.Name.append(Name)
        annot_c.Type.append(Type)
        annot_c.Color.append(Color)
        annot_c.Group.append(Group)
        
        annot_c.x_s.append(region_X)
        annot_c.y_s.append(region_Y)
        annot_c.min_x.append(min(region_X))
        annot_c.max_x.append(max(region_X))
        annot_c.min_y.append(min(region_Y))
        annot_c.max_y.append(max(region_Y))
        annot_c.points.append(region_Points)

        if Name == 'Annot_ROI':
            annot_c.Class.append(CLASS_ID_NAME_ACDC_S1_Tiling['Annot_ROI'])
        else:
            annot_c.Class.append(CLASS_ID_NAME_ACDC_S1_Tiling['Cancer'])
            
    return annot_c



#patch_size = 4096




run_info = 'makedataset_{}'.format(patch_size)

if not os.path.exists('checkpoints/{}'.format(run_info)):
    os.mkdir('checkpoints/{}'.format(run_info))

log = logging.getLogger('staining_log')
log.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fileHandler = logging.FileHandler('checkpoints/{}/log.txt'.format(run_info))
streamHandler = logging.StreamHandler()
fileHandler.setFormatter(formatter)
streamHandler.setFormatter(formatter)
#
log.addHandler(fileHandler)
log.addHandler(streamHandler)




#list_ = ['Page-1', 'Page-2', 'Page-3', 'Page-4', 'Page-5', 'Page-6', 'Page-7', 'Page-8', 'Page-9', 'Page10']
list_ = ['Page-{}'.format(num)]

for idx in list_:
    
    log.info(" Start : {}".format(idx))

    image_path = sorted(glob.glob('../../../../Projects/Pathology/datasets/HE_IHC/Stomach/{}/*.svs'.format(idx)))
    xml_path = sorted(glob.glob('../../../../Projects/Pathology/datasets/HE_IHC/Stomach/{}/*.xml'.format(idx)))
    
    patient_name = [i.split('/')[-1].split('.')[0] for i in image_path]
    
    
    for num_ in range(len(patient_name)):
        
        patient = patient_name[num_]
        
        patient_svs_path = image_path[num_]
        patient_xml_path = xml_path[num_]
        
        
        temp = openslide.OpenSlide(patient_svs_path)


        x_s = []
        y_s = []
        min_x = []
        max_x = []
        min_y = []
        max_y = []

        points = []


        tree = parse(patient_xml_path)
        root = tree.getroot()  # 최상단 태그
        Annotations = root.findall('Annotation')

        for Annotation in Annotations:

            regions = Annotation.find("Regions").findall("Region")

            for region in regions:

                vertices = region.find("Vertices").findall("Vertex")
                region_X = []
                region_Y = []
                region_Points = []

                for vertex in vertices:
                    x = int(round(float(vertex.attrib['X'])))
                    y = int(round(float(vertex.attrib['Y'])))

                    region_X.append(x)
                    region_Y.append(y)
                    region_Points.append([x, y])

                x_s.append(region_X)
                y_s.append(region_Y)

                min_x.append(min(region_X))
                max_x.append(max(region_X))
                min_y.append(min(region_Y))
                max_y.append(max(region_Y))

                points.append(region_Points)


        downscale = 32
        binary_mask = np.zeros(shape=(int(temp.dimensions[1]/downscale), int(temp.dimensions[0]/downscale)))

        for i in range(len(points)):
            #region1_poly = Polygon(np.array(points[0]))
            #region2_poly = Polygon(np.array(points[1]))

            vertex_row_coords = np.array(points[i])[:,0]
            vertex_col_coords = np.array(points[i])[:,1]

            fill_row_coords, fill_col_coords = draw.polygon(np.array([int(i/downscale) for i in vertex_col_coords]),
                                                        np.array([int(i/downscale) for i in vertex_row_coords]),
                                                        binary_mask.shape)

            binary_mask[fill_row_coords, fill_col_coords] = 255



        ################### All Coords #################

        width_ = int(temp.dimensions[0] / patch_size)
        height_ = int(temp.dimensions[1] / patch_size)

        all_coords = np.array([i[0] for i in np.ndenumerate(np.ones(shape=(width_, height_)))]) * patch_size






        ################### Hotspot Coords #################
        image_size = int(patch_size / downscale)
        height = int(binary_mask.shape[0] / image_size)
        width = int(binary_mask.shape[1] / image_size)


        hotspot_ = []
        for a in range(height):
            for b in range(width):

                patch = binary_mask[a*image_size : (a+1)*image_size, b*image_size : (b+1)*image_size]

                
                Rate = 0.7
                if (patch == 255).sum() > int((image_size*image_size) * Rate):

                    y = [a*image_size]
                    x = [b*image_size]

                    hotspot_.append(x + y)
        hotspot_coords = np.array([i*downscale for i in np.array(hotspot_)])

        
        
        if len(hotspot_coords) == 0:
            log.info("{} - There is No Hotspot !!".format(patient))
            
            continue
        ################### Hotspot Coords #################
        
        
        
        
        
        
        
        ################### Back Coords #################
        idx_ = np.isin([str(i) for i in all_coords], [str(i) for i in hotspot_coords]) == False
        all_except_hotspot_coords =  all_coords[idx_]
        
        back_coords = []
        for i in range(len(all_except_hotspot_coords)):

            patch = np.array(temp.read_region((all_except_hotspot_coords[i][0], all_except_hotspot_coords[i][1]), 0, (patch_size, patch_size)))[:, :, :3]

            if patch.mean() > 220:
                back_coords.append([all_except_hotspot_coords[i][0], all_except_hotspot_coords[i][1]])

        back_coords = np.array(back_coords)
        ################### Back Coords #################



        
        ################### No Hotspot Coords #################
        concat = np.vstack([hotspot_coords, back_coords])

        no_hotspot_idx = np.isin(np.array([str(i) for i in all_coords]), np.array([str(i) for i in concat])) == False
        no_hotspot = all_coords[no_hotspot_idx]

        np.random.shuffle(no_hotspot)
        no_hotspot = no_hotspot.tolist()
        ################### No Hotspot Coords #################


        
        
        

        Total_dict= {'{}'.format(patient): {}}

        nohotspot_label = np.zeros_like(no_hotspot)[:, 0].tolist()
        hotspot_label = np.ones_like(hotspot_coords)[:, 0].tolist()



        Total_dict['{}'.format(patient)]['slides_path'] = patient_svs_path        

        number = len(hotspot_label)

        Total_dict['{}'.format(patient)]['coords'] = no_hotspot[:number] + hotspot_coords.tolist()[:number]
        Total_dict['{}'.format(patient)]['label'] =  nohotspot_label[:number] + hotspot_label[:number]





        result_path = 'Dict{}_new'.format(patch_size)
        if not os.path.exists('{}'.format(result_path)):
            os.mkdir('{}'.format(result_path))

        if not os.path.exists('{}/{}'.format(result_path, idx)):
            os.mkdir('{}/{}'.format(result_path, idx))

        with open('{}/{}/{}.pickle'.format(result_path, idx, patient), 'wb') as ok:
            pickle.dump(Total_dict, ok)

            
            
        log.info(" All : {}  Hotspot : {}  No Hotspot : {}  Back : {}".format(len(all_coords), len(hotspot_coords), len(no_hotspot),len(back_coords)))
        log.info('No Hotspot : {} Hotspot : {}'.format(len(nohotspot_label[:number]), len(hotspot_label[:number])))
        log.info('{} - {}   Complete'.format(idx, patient))
        log.info('\n')
            
            
            
            
            