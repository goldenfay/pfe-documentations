import torch
from torch.utils.data import Dataset
from PIL import Image
import os,cv2
import matplotlib.pyplot as plt
import numpy as np

class BasicDataSet(Dataset):
    def __init__(self,instances):
        self.instances=instances
        self.n_instances=len(instances)

    def __len__(self):
        return self.n_instances

    def __getitem__(self,index):
        assert index <= len(self), 'index range error'
        return self.instances[index]
            

class CrowdDataset(Dataset):
    
    '''
    crowdDataset
    '''
    def __init__(self,img_rootPath,gt_dmap_rootPath,gt_downsample=1):
        '''
        img_rootPath: the root path of img.
        gt_dmap_rootPath: the root path of ground-truth density-map.
        gt_downsample: default is 0, denote that the output of deep-model is the same size as input image.
        '''
        self.gt_downsample=gt_downsample
        self.img_rootPath=img_rootPath
        self.gt_dmap_rootPath=gt_dmap_rootPath
        
        self.img_names=[filename for filename in os.listdir(img_rootPath) if os.path.isfile(os.path.join(img_rootPath,filename))]
        self.n_samples=len(self.img_names)

    def __len__(self):
        return self.n_samples

    def __getitem__(self,index):
        assert index <= len(self), 'index range error'
        img_name=self.img_names[index]
        # img=plt.imread(os.path.join(self.img_rootPath,img_name))
        # if len(img.shape)==2: # expand grayscale image to three channel.
        #     img=img[:,:,np.newaxis]
        #     img=np.concatenate((img,img,img),2)
        img=Image.open(os.path.join(self.img_rootPath,img_name))
        if img.mode == 'L':
            img = img.convert('RGB')
            
        img=np.asarray(img)    
        gt_dmap=np.load(os.path.join(self.gt_dmap_rootPath,img_name.replace('.jpg','.npy')))
        # temp=Image.fromarray(gt_dmap)
        # if temp.mode == 'L':
        #     gt_dmap = np.asarray(temp.convert('RGB'))
            

        if self.gt_downsample>1: # to downsample image and density-map to match deep-model.
            ds_rows=int(img.shape[0]//self.gt_downsample)
            ds_cols=int(img.shape[1]//self.gt_downsample)
            img = cv2.resize(img,(ds_cols*self.gt_downsample,ds_rows*self.gt_downsample))
            img=img.transpose((2,0,1)) # convert to order (channel,rows,cols)
            gt_dmap=cv2.resize(gt_dmap,(ds_cols,ds_rows))
        gt_dmap=gt_dmap[np.newaxis,:,:]
    
        img_tensor=torch.tensor(img,dtype=torch.float).permute((2,0,1))/255
        # img_tensor=torch.from_numpy(img).permute((2,0,1))
        gt_dmap_tensor=torch.tensor(gt_dmap,dtype=torch.float)/255
        
        return img_tensor,gt_dmap_tensor
        # return img,gt_dmap


# test code
if __name__=="__main__":
    import torchvision
    img_rootPath="C:\\Users\\PC\\Desktop\\PFE related\\existing works\\Zhang_Single-Image_Crowd_Counting_CVPR_2016_paper code sample\\MCNN-pytorch-master\\MCNN-pytorch-master\\ShanghaiTech\\part_A\\train_data\\images"
    gt_dmap_rootPath="C:\\Users\\PC\\Desktop\\PFE related\\existing works\\Zhang_Single-Image_Crowd_Counting_CVPR_2016_paper code sample\\MCNN-pytorch-master\\MCNN-pytorch-master\\ShanghaiTech\\part_A\\train_data\\ground-truth"
    dataset=CrowdDataset(img_rootPath,gt_dmap_rootPath)
    for i,(img,gt_dmap) in enumerate(dataset):
        print(img[:,1:5,1:3])
        plt.imshow(np.asanyarray(img.permute(1,2,0),dtype=np.int) )
        plt.show()
        plt.imshow(gt_dmap,cmap='jet')
        plt.show()
        if i>5:
            break
        print(img.shape,gt_dmap.shape)