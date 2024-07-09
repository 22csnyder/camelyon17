import multiresolutionimageinterface as mir
import glob2
import os
from os.path import join, isfile

import time

import argparse

'''

This is code to generate image masks for all annotation files that have
matching images in the directory provided.



##Hi Ken
#you gotta have the image files unzipped
#then you just run make_masks( '/home/ken/Camelyon') assuming
/home/ken/Camelyon' is where you keep your data

If you have trouble let me know

'''

def get_arguments():
    d='produce mask files in data_dir/lesion_masks corresponding images with lesion'
    parser= argparse.ArgumentParser(description=d)

    parser.add_argument('data_dir',type=str,help='a directory with tif files\
                        (anywhere) that match the names of xml files in a \
                        lesion_annotations folder.')
    return parser



#TODO: should be modified to accept cmd line args for folders

def get_node_name(full_filename):
    fname=os.path.basename(full_filename)
    ptname=fname.split('.')[0]
    return ptname


def find_node_image(nodename,data_dir):
    possible_matches=sorted(glob2.glob(data_dir+'/**/'+nodename+'.tif'),
                            key=lambda f: filename_parse(f))
    if len(possible_matches)>=2:
        raise ValueError('should only be 1 or 0 matches for:',nodename,'\n',
                         'matches:',possible_matches)
    elif len(possible_matches)==0:
        print 'warning, no match for ',nodename
        return None
    elif len(possible_matches)==1:
        return possible_matches[0]

def get_image_details(img_file):
    reader = mir.MultiResolutionImageReader()
    mr_image = reader.open(img_file)
    dim,spacing=mr_image.getDimensions(), mr_image.getSpacing()
    return dim,spacing



def make_masks(data_dir):
    fname='lesion_masks'
    masks_folder=join(data_dir,fname)

    #make destination file
    if not fname in os.listdir(data_dir):
        os.makedirs( masks_folder)

    #get list of filename
    annotations_folder= join(data_dir, 'lesion_annotations')
    annos=os.listdir(annotations_folder)
    annotation_filenames=[join(annotations_folder,f) for f in annos]

    #make list of potential output files
    mask_suffix='_mask.tif'
    mas=[f.split('.')[0]+mask_suffix for f in annos]
    mask_filenames=[join(masks_folder,f) for f in mas]

    timestart=time.time()
    i=0
    for anno_source,output_path in zip(annotation_filenames,mask_filenames):
        if os.path.exists(output_path):
            print 'skipping..',output_path,'..already exists\n'
            continue#annotation mask already exists

        node_name=get_node_name(anno_source)
        img_file=find_node_image(node_name,data_dir)
        if img_file is None:
            print 'skipping ..',anno_source, '.. no corresponding node image\n'
            continue #no corresp mask img

        print 'starting node:',node_name,'..'

        i+=1

        dims,spacing=get_image_details(img_file)
        print 'starting mask ',i,' elapsed time:',time.time()-timestart ,'\n'
        make(anno_source,output_path,dims,spacing)

    print 'finished. time per mask:', (time.time()-timestart)/i,'\n'



def make(anno_source,output_path,dims,spacing):
    print 'making mask:',output_path,'..'
    annotation_list = mir.AnnotationList()
    xml_repository = mir.XmlRepository(annotation_list)

    xml_repository.setSource(anno_source)
    xml_repository.load()
    annotation_mask = mir.AnnotationToMask()

    label_map = {'metastases': 1, 'normal': 2}
    annotation_mask.convert(annotation_list, output_path,dims,spacing, label_map)


if __name__ == '__main__':
    parser=get_arguments()
    args=parser.parse_args()

    print 'making masks'
    make_masks(data_dir= args.data_dir)

