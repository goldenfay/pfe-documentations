import os,traceback,time,datetime
from github import InputGitTreeElement
from github import Github
from github.Repository import Repository
import base64
class GitManager:

    def __init__(self,access_token):
       self.token=access_token

    def authentification(self):
    
        x='707a725cc9ca67b0ee3c9662a65f02ab88e1b597'
        self.gth = Github(self.token)

        return self.gth

    
    def get_repo(self,repo_name):
        
        for repo in self.gth.get_user().get_repos():
            if repo.name==repo_name:
                return repo
            
        return None
    @classmethod    
    def get_repo_files(cls,repo:Repository):
        return [el.path for el in repo.get_contents('')]

    def push_files(self,repo:Repository,files_list,push_msg,branch='master'):
 
        master_ref = repo.get_git_ref('heads/'+branch)
        master_sha = master_ref.object.sha
        base_tree = repo.get_git_tree(master_sha)
        element_list = list()
        for entry in files_list:
            with open(entry, 'rb') as input_file:
                data = input_file.read()
            if entry.endswith('.pth'):
                data = base64.b64encode(data)
            blob = repo.create_git_blob(data.decode("utf-8"), "base64")    
            element = InputGitTreeElement(os.path.basename(entry), '100644', 'blob', sha=blob.sha)
            element_list.append(element)    
        tree = repo.create_git_tree(element_list, base_tree)
        parent = repo.get_git_commit(master_sha)
        commit = repo.create_git_commit(push_msg, tree, [parent])
        
        master_ref.edit(commit.sha)
        self.log_commit('commit.txt',files_list)
        print('\t Done')
        """ An egregious hack to change the PNG contents after the commit """
        # for entry in files_list:
        #     with open(entry, 'rb') as input_file:
        #         data = input_file.read()
        #     if entry.endswith('.pth'):
        #         old_file = repo.get_contents(entry)
        #         commit = repo.update_file('/' + entry, 'Update content', data, old_file.sha)
        return len(element_list)    
    @classmethod
    def log_commit(cls,logfile_path,files_list):
        with open(logfile_path,'a') as f:
            f.write('Commit :'+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")+'\nList :'+','.join(files_list)+'\n')