import os,sys,inspect,glob,io,subprocess,re
def import_or_install(package,pipname):
    try:
        __import__(package) 
    except ImportError:
        #pip.main(['install', pipname])
        subprocess.check_call([sys.executable, "-m", "pip", "install", pipname])

import_or_install("matplotlib","matplotlib")
import_or_install("visdom","visdom")
import_or_install("numpy","numpy")
import_or_install("matplotlib","matplotlib")

import torch
from torch import nn
import matplotlib as plt
import visdom as vis
import numpy as np

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
    # User's modules from another directory
sys.path.append(os.path.join(parentdir , "bases"))
sys.path.append(os.path.join(parentdir , "models"))
sys.path.append(os.path.join(parentdir , "data_loaders"))
sys.path.append(os.path.join(parentdir , "density_map_generators"))

from datasets import *
from params import *
from dm_generator import *
from knn_gaussian_kernal import *
from loaders import *
from mcnn import *



def prepare_datasets(baseRootPath,datasets_list:list,dm_generator,resetFlag=False):
    print("####### Preparing Data...")
    paths_list=[]
    for dataset_name in datasets_list:
        if 'ShanghaiTech_partA'==dataset_name:
            paths_list.append(prepare_ShanghaiTech_dataset(baseRootPath,'A',dm_generator,resetFlag))
        elif 'ShanghaiTech_partB'==dataset_name:
            paths_list.append(prepare_ShanghaiTech_dataset(baseRootPath,'B',dm_generator,resetFlag))   
        else:
            print("nop")

    
    return paths_list         
    

def prepare_ShanghaiTech_dataset(root,part,dm_generator,resetFlag=False):
    root=os.path.join(root,"ShanghaiTech")
    paths_dict=dict()
        # generate the ShanghaiA's ground truth
    if not part=="A" and not part=="B": raise Exception("Invalide parts passed for shanghai ")

    train_path=os.path.join(root,'part_'+part,'train_data')
    test_path=os.path.join(root,'part_'+part,'test_data')
    # part_A_train = os.path.join(root,'part_A\\train_data','images')
    # part_A_test = os.path.join(root,'part_A\\test_data','images')
    # part_A_train = os.path.join(root,'part_A\\train_data','images')
    # part_A_test = os.path.join(root,'part_A\\test_data','images')
    # part_B_train = os.path.join(root,'part_B_final/train_data','images')
    # part_B_test = os.path.join(root,'part_B_final/test_data','images')
    # path_sets = [part_A_train,part_A_test]
    

        # save both train and test paths
    paths_dict["images"]=os.path.join(train_path,'images')
    paths_dict["ground-truth"]=os.path.join(train_path,'ground-truth')

    path_sets = [paths_dict["images"],paths_dict["ground-truth"]]
    
    img_paths = []
        # Grab all .jpg images paths
    for path in path_sets:
        for img_path in glob.glob(os.path.join(path, '*.jpg')):
            img_paths.append(img_path)

            # Generate density map for each image
    for img_path in img_paths:
        if os.path.exists(img_path.replace('.jpg','.npy').replace('images','ground-truth')) and not resetFlag:
            #print("\t Already exists.")
            continue
        print('Generating Density map for : ',img_path[list(re.finditer("[\\\/]",img_path))[-1].start(0):]," :")

            # load matrix containing ground truth infos
        mat = io.loadmat(img_path.replace('.jpg','.mat').replace('images','ground-truth').replace('IMG_','GT_IMG_'))
        img= plt.imread(img_path)#768行*1024列
        density_map = np.zeros((img.shape[0],img.shape[1]))
        points = mat["image_info"][0,0][0,0][0] #1546person*2(col,row)

            # Generate the density map
        density_map = dm_generator.generate_densitymap(img,points)
        # plt.imshow(k,cmap=CM.jet)

            # save density_map on disk
        np.save(img_path.replace('.jpg','.npy').replace('images','ground-truth'), density_map)

    return paths_dict        


def getloader(loader_type,img_gtdm_paths):
    print("####### Getting DataLoader...")
    if loader_type=="Generic_Loader":
        return GenericLoader(img_gtdm_paths)



def getModel(model_type,weightsFlag=False):
    print("####### Getting Model : ",model_type,"...")
    if model_type=="MCNN":
        return MCNN(weightsFlag)

if __name__=="__main__":
    if len(sys.argv)>1:
        root = os.path.join(sys.argv[1],'ShanghaiTech')
    else :
        root = 'C:\\Users\\PC\\Desktop\\PFE related\\existing works\\Zhang_Single-Image_Crowd_Counting_CVPR_2016_paper code sample\\MCNN-pytorch-master\\MCNN-pytorch-master\\ShanghaiTech'
    dm_generator_type="knn_gaussian_kernal"
    dataset_names=["ShanghaiTech_partA","ShanghaiTech_partB"]
    dm_generator=None
    loader_type="Generic_Loader"
    model_type="MCNN"
    model=None
   # device=torch.device("cuda")
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params={"lr":1e-6,
            "momentum":0.95,
            "maxEpochs":1000,
            "criterionMethode":'MSELoss',
            "optimizationMethod":'SGD'
            }

    if dm_generator_type=="knn_gaussian_kernal":
        dm_generator=KNN_Gaussian_Kernal_DMGenerator()

    datasets_paths=prepare_datasets(root,dataset_names,dm_generator)
    img_gtdm_paths=[(el["images"],el["ground-truth"]) for el in datasets_paths]


    data_loader=getloader(loader_type,img_gtdm_paths)

    dataloaders=data_loader.load()

    model=getModel(model_type)

        # This loop is basically used in experimentations
    # for train_loader,test_loader in dataloaders:
    #     model.train_model(train_loader,test_loader,train_params)

    merged_train_dataset,merged_test_dataset=data_loader.merge_datasets(dataloaders)
    train_dataloader=torch.utils.data.DataLoader(merged_train_dataset)
    test_dataloader=torch.utils.data.DataLoader(merged_test_dataset)
    
    train_params=TrainParams(device,model,params["lr"],params["momentum"],params["maxEpochs"],params["criterionMethode"],params["optimizationMethod"])
    model.train_model(merged_train_dataset,merged_test_dataset,train_params,resume=True)
    
    
    
