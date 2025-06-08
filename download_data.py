import gdown
import os

ManySig_LINK = "https://drive.google.com/uc?id=1szuns8MhcYocdbipK9t9TM9MLgEMklxk"

LoRa_Indor_LINK = "https://research.engr.oregonstate.edu/hamdaoui/RFFP-dataset/LoRa-Dataset/Diff_Days_Indoor_Setup/"

LoRa_Outdor_LINK = "https://research.engr.oregonstate.edu/hamdaoui/RFFP-dataset/LoRa-Dataset/Diff_Days_Outdoor_Setup/"

# Download WiSig dataset
gdown.download(ManySig_LINK, 'many_sig', quiet=False)
os.system('unzip many_sig')

# Download LoRa datasets
os.system(f"wget -r -np -nH --cut-dirs=2 -R index.html {LoRa_Indor_LINK}")
os.system(f"wget -r -np -nH --cut-dirs=2 -R index.html {LoRa_Outdor_LINK}")