from PIL import Image
import numpy as np

def PSNRLoss(img1, img2):
    # print(img1.type)
    # img1 = np.array(img1).astype(np.float64)
    # img2 = np.array(img2).astype(np.float64)
    img1 = img1.cpu().numpy()
    # print(img1.type)
    img2 = img2.cpu().detach().numpy()
    # print(img2)
    mse = np.mean((img1-img2)**2)
    if mse == 0:
        return float('inf')
    else:
        return 20*np.log10(1/np.sqrt(mse))

