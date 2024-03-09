def extract_tarfile(tar_file_path, extract_folder):
    # Check if the extract folder exists, if not, create it
    if not os.path.exists(extract_folder):
        os.makedirs(extract_folder)

    # Open the tar file
    with tarfile.open(tar_file_path, 'r') as tar:
        # Extract all contents to the specified folder
        tar.extractall(extract_folder)

    print(f"Successfully extracted {tar_file_path} to {extract_folder}")
    
    
def pathListIntoIds(dirList):
    #ids container
    x = []
    #extract all ids from dirctory path
    for i in range(0,len(dirList)):
        x.append(dirList[i][dirList[i].rfind('/')+1:])
    return x

def GetListIds(DATASET_PATH):
    # lists of directories with
    directories  = [file.path for file in os.scandir(DATASET_PATH) if file.is_dir()]
    return pathListIntoIds(directories); 

def check_update_ids(DATASET_PATH,all_ids):
    print('N of directories befor:',len(all_ids))
    update = []
    # verify if all the samples hase the 4 cat info t1,t1ce,t2,flair and seg
    for i in all_ids:
        if os.path.exists(os.path.join(DATASET_PATH,i,f'{i}_t2.nii.gz')) and  os.path.exists(os.path.join(DATASET_PATH,i,f'{i}_t1ce.nii.gz')) and os.path.exists(os.path.join(DATASET_PATH,i,f'{i}_t1.nii.gz')) and os.path.exists(os.path.join(DATASET_PATH,i,f'{i}_flair.nii.gz')) and os.path.exists(os.path.join(DATASET_PATH,i,f'{i}_seg.nii.gz')):
            update.append(i)

    print('N of directories after:',len(update))
    return update

def main(work_path,tar_file_path,extract_folder):

    #extract the dataset 
    extract_tarfile(tar_file_path, extract_folder)
    #path to extracted DATASET
    DATASET_PATH = os.path.join(work_path,extract_folder)
    #Get all ids
    all_ids = GetListIds(DATASET_PATH)
    #remove the ids that dosn't contain the 4 modalities
    all_ids = check_update_ids(DATASET_PATH,all_ids)
    
    return DATASET_PATH,all_ids