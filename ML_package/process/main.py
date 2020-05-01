import os,sys,inspect,glob,io,subprocess,re,gc
def import_or_install(package,pipname):
    try:
        __import__(package) 
    except ImportError:
        #pip.main(['install', pipname])
        subprocess.check_call([sys.executable, "-m", "pip", "install", pipname])

import_or_install("matplotlib","matplotlib")
import_or_install("visdom","visdom")
import_or_install("numpy","numpy")
import_or_install("pydrive","Pydrive")
import_or_install("github","PyGithub")

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
from CSRNet import *
from SANet import *
import utils
import plots



def prepare_datasets(baseRootPath,datasets_list:list,dm_generator,resetFlag=False):
    '''
        Prepars DataSets for training by generating ground-truth density map for every image of every Dataset.
    '''
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

    
def check_previous_loaders(loader_type,img_gtdm_paths,params:dict=None):
    '''
        Checks for previous versions of the loader, in order to optimize creating and generating Dataloaders.
    '''
    print("\t Checking for previous loader ...")
    if params is None:
        test_size=20
        batch_size=1
    else:
        test_size=params['test_size']
        batch_size=params['batch_size']
    restore_path=os.path.join(utils.BASE_PATH,'obj','loaders',loader_type)    
    if not os.path.exists(restore_path) :
        return None
    if len( glob.glob(restore_path) )==0:
        return None

    saved_infos=utils.load_obj(os.path.join(restore_path,'saved.pkl'))
    if saved_infos['paths']!=img_gtdm_paths: return None
    if saved_infos['test_size']!=test_size: return None
    if saved_infos['batch_size']!=batch_size: return None

    return saved_infos['samplers']





def getloader(loader_type,img_gtdm_paths,restore_flag=True):
    '''
        Returns a new loader according to the passed type.
    '''
    print("####### Getting DataLoader...")
    if loader_type=="GenericLoader":
        return GenericLoader(img_gtdm_paths)


    

def getModel(model_type,load_saved=False,weightsFlag=False):
    '''
        Loads a models according to type, if a previous version was found, load it directely.
    '''
    print("####### Getting Model : ",model_type,"...")
    if load_saved and  os.path.exists(os.path.join(utils.BASE_PATH,'obj','models',model_type)):
        return torch.load(os.path.join(utils.BASE_PATH,'obj','models',model_type))
    if model_type=="MCNN":
        return MCNN(weightsFlag)
    elif model_type=="CSRNet":
        return CSRNet(weightsFlag)
    elif model_type=="SANet":
        return SANet()            

def get_best_model(min_epoch,className):
    '''
        Loads the best model resulting from train.
    '''
    
    if not os.path.exists(os.path.join(utils.BASE_PATH , 'checkpoints2',className)):
        raise Exception("Cannot load model. Checkpoint directory not found!")
    if not os.path.exists(os.path.join(utils.BASE_PATH , 'checkpoints2',className,'epoch_'+str(min_epoch)+'.pth')):
        raise Exception("Cannot load model.Best epoch checkpoint does not exists!")

    return torch.load(os.path.join(utils.BASE_PATH , 'checkpoints2',className,'epoch_'+str(min_epoch)+'.pth'))

if __name__=="__main__":
    if len(sys.argv)>1:
        root = os.path.join(sys.argv[1],'ShanghaiTech')
    else :
        root = 'C:\\Users\\PC\\Desktop\\PFE related\\existing works\\Zhang_Single-Image_Crowd_Counting_CVPR_2016_paper code sample\\MCNN-pytorch-master\\MCNN-pytorch-master\\ShanghaiTech'
    dm_generator_type="knn_gaussian_kernal"
    dataset_names=["ShanghaiTech_partA","ShanghaiTech_partB"]
    dm_generator=None
    loader_type="GenericLoader"
    model_type=sys.argv[2] if len(sys.argv)>2 else "CSRNet"
    model=None
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
    samplers=check_previous_loaders(loader_type,img_gtdm_paths)
    if samplers is None:
        dataloaders=data_loader.load(save=True)
        gc.collect()
    else:
        print('\t A previous version of the loader was found! Restoring samplers ...')
        dataloaders=data_loader.load_from_samplers(samplers)    
        

    

        # This loop is basically used in experimentations
    # for train_loader,test_loader in dataloaders:
    #     model.train_model(train_loader,test_loader,train_params)

    merged_train_dataset,merged_test_dataset=data_loader.merge_datasets(dataloaders)
    train_dataloader=torch.utils.data.DataLoader(merged_train_dataset)
    test_dataloader=torch.utils.data.DataLoader(merged_test_dataset)
    
    model=getModel(model_type,load_saved=True)
    train_params=TrainParams(device,model,params["lr"],params["momentum"],params["maxEpochs"],params["criterionMethode"],params["optimizationMethod"])
    epochs_list,train_loss_list,test_error_list,min_epoch,min_MAE,train_time=model.train_model(merged_train_dataset,merged_test_dataset,train_params,resume=True)
    print(epochs_list,train_loss_list,test_error_list,min_epoch,min_MAE,train_time)
    _,model.min_MAE,model.min_epoch=model.load_chekpoint(os.path.join(utils.BASE_PATH , 'checkpoints2',model.__class__.__name__,'epoch_'+str(min_epoch)+'.pth'))
    model.save()

    print(model.eval_model(test_dataloader))
    
    
    
