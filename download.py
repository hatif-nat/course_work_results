import gdown
import zipfile

url = 'https://drive.google.com/uc?export=download&id=1xTLaO1awX3NmjVwKs0Od_eEKc933bVty'
output = 'train.csv'
gdown.download(url, output, quiet=False)

url = 'https://drive.google.com/uc?export=download&id=1ra_Bv4wo9_sg4_QyzDm7t_NGWCb_nRLU'
output = 'extraWords.tsv'
gdown.download(url, output, quiet=False)

url = 'https://drive.google.com/uc?export=download&id=1QvLqPg0I2SkxW55w4K8PlPMOclnZyxjC'
output = 'SenseOfTrainWordsBTS.csv'
gdown.download(url, output, quiet=False)


url = 'https://drive.google.com/u/0/uc?export=download&confirm=baTG&id=1IcQx6t5qtv4bdcGjjVCwXnRkpr67eisJ'
output = 'a.zip'
gdown.download(url, output, quiet=False)


with zipfile.ZipFile("a.zip","r") as zip_ref:
    zip_ref.extractall("")