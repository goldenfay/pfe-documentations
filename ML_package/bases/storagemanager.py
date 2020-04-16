
import os,sys,glob
import torch
from pydrive.drive import GoogleDrive
from pydrive.auth import GoogleAuth

import utils
from gitmanager import GitManager

def save_file(path,file_to_save,env,saver_module='torch',alternative=None):
    if env!='drive':
        if saver_module=='torch':
            torch.save(file_to_save,path)
        else:
            utils.save_obj(file_to_save,path)

        return

        # If platform is Google drive, then do checks 
    from google.colab import auth
    from oauth2client.client import GoogleCredentials

    auth.authenticate_user()
    gauth = GoogleAuth()
    gauth.credentials = GoogleCredentials.get_application_default()
    #gauth.LocalWebserverAuth()
    
    drive = GoogleDrive(gauth)
    infos=drive.GetAbout()
    if int(infos['quotaBytesUsed'])+sys.getsizeof(file_to_save)>=int(infos['quotaBytesTotal'])-(200*1024*1024):
            print('\t [Alert] Maximum storage reached !','\n\t',' Migration of all checkpoints to github ...')
                # Authentification to github
            # git_manager=GitManager('5598c0e73e05423e7538fd19cb2d510379e9e588')
            git_manager=GitManager(user='ihasel2020@gmail.com',pwd='pfemaster2020')
            git_manager.authentification()
            target_repo=git_manager.get_repo('checkpoints')
                # Fetch checkpoints from the directory in order to push them all to github
            files_to_push=[os.path.abspath(el) for el in glob.glob(os.path.join(os.path.split(path)[0],'*.pth'))]
            res=git_manager.push_files(target_repo,files_to_push,'checkpoints migration')
                # If all files were pushed without problem, delete them
            if isinstance(res,int)and res==len(files_to_push):
                print('\t Successfully transfered checkpoints to github')
                for f in glob.glob(os.path.join(os.path.split(path)[0],'*.pth')):
                    os.remove(f)
                    # Now save the file
                torch.save(file_to_save,path)
                assert os.path.exists(path), 'Error ! File to save couldn\'t be saved !'
            else: raise RuntimeError('Couldn\'t push all files')






    else: 
        print('there is a space')
        torch.save(file_to_save,path)
          



