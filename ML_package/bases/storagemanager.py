
import os,sys,glob
import torch
from pydrive.drive import GoogleDrive
from pydrive.auth import GoogleAuth
from oauth2client.client import GoogleCredentials
import utils
from gitmanager import GitManager


def authentificate_Grive():
    
    # from google.colab import auth
    

    gauth = GoogleAuth()

    # gauth.LoadCredentialsFile(os.path.dirname(os.path.join(utils.BASE_PATH,'mycredentials.txt')))
    if gauth.credentials is None:
 
        gauth.GetFlow()
        gauth.flow.params.update({'access_type': 'offline'})
        gauth.flow.params.update({'approval_prompt': 'force'})

        gauth.LocalWebserverAuth()
        # gauth.credentials = GoogleCredentials.get_application_default()

    elif gauth.access_token_expired:


        gauth.Refresh()
    else:

        gauth.Authorize()
        # auth.authenticate_user()
    # gauth.SaveCredentialsFile(os.path.dirname(os.path.join(utils.BASE_PATH,'mycredentials.txt')))

    return gauth

def save_file(path,file_to_save,env,min_epoch,saver_module='torch',alternative=None):
    
    if env!='drive':
        if saver_module=='torch':
            torch.save(file_to_save,path)
        else:
            utils.save_obj(file_to_save,path)

        return 1

        # If platform is Google drive, then do checks 
    
    gauth=authentificate_Grive
    
    drive = GoogleDrive(gauth)
    infos=drive.GetAbout()
    if int(infos['quotaBytesUsed'])+sys.getsizeof(file_to_save)>=int(infos['quotaBytesTotal']):
            print('\t [Alert] Maximum storage reached on Drive!','\n\t',' Migration of all checkpoints to github ...')
                # Authentification to github
            # git_manager=GitManager('5598c0e73e05423e7538fd19cb2d510379e9e588')
            # git_manager=GitManager(user='ihasel2020@gmail.com',pwd='pfemaster2020')
            # git_manager.authentification()
            # target_repo=git_manager.get_repo('checkpoints')
            #     # Fetch checkpoints from the directory in order to push them all to github
            # files_to_push=[os.path.abspath(el) for el in glob.glob(os.path.join(os.path.split(path)[0],'*.pth'))]
            # res=git_manager.push_files(target_repo,files_to_push,'checkpoints migration')
            #     # If all files were pushed without problem, delete them
            # if isinstance(res,int)and res==len(files_to_push):
            #     print('\t Successfully transfered checkpoints to github')
            #     for f in glob.glob(os.path.join(os.path.split(path)[0],'*.pth')):
            #         os.remove(f)
            #         # Now save the file
            #     torch.save(file_to_save,path)
            #     assert os.path.exists(path), 'Error ! File to save couldn\'t be saved !'
            # else: raise RuntimeError('Couldn\'t push all files')
            return 0
            






    else: 
        torch.save(file_to_save,path)
        return 1
          



