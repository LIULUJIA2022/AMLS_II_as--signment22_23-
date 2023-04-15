# DataPreparation

import os
import shutil
import pandas as pd



# The original data has been disrupted, and the first 3000 samples are divided as the validation set
class Data():
    def __init__(self,
                 ###
                 file_path="AMLS_LeafDiseaseClassification",
                 is_train=True,
                 label_file="train.csv",
                 new_train_path="new_train",
                 new_val_path="new_val",
                 image_path="train_images"):

        self.file_path = file_path

        # Create a training set directory
        if not os.path.exists(os.path.join(file_path, new_train_path)):
            os.mkdir(os.path.join(file_path, new_train_path))

        # Create a validation set directory
        if not os.path.exists(os.path.join(file_path, new_val_path)):
            os.mkdir(os.path.join(file_path, new_val_path))

        datas = pd.read_csv(os.path.join(file_path, label_file))  # Read images and labels
        print("samples number:",
              datas.shape)  # (Number of samples, 2 columns), the first column is the image path, and the second column is the image category
        print("samples classification：corresponding to samples number")
        print(datas.label.value_counts())  # Number of statistical categories

        '''
        Divide the training set and validation set. And store the data in the root/class_a/1.ext format, and use the DatasetFolder method to read the data       
        '''
        for index, row in datas.iterrows():
            # Divide the first 3000 images as a validation set
            if index < 3000:
                # Create folders for each class
                if not os.path.exists(os.path.join(file_path, new_val_path, str(row[-1]))):
                    os.mkdir(os.path.join(file_path, new_val_path, str(row[-1])))
                # Copy the data under the original path to the new path
                shutil.move(os.path.join(file_path, image_path, str(row[0])),
                            os.path.join(file_path, new_val_path, str(row[-1]), str(row[0])))
            else:
                # Create folders for each class
                if not os.path.exists(os.path.join(file_path, new_train_path, str(row[-1]))):
                    os.mkdir(os.path.join(file_path, new_train_path, str(row[-1])))
                # Copy the data under the original path to the new path
                shutil.move(os.path.join(file_path, image_path, str(row[0])),
                            os.path.join(file_path, new_train_path, str(row[-1]), str(row[0])))

    '''
    # look image
    def show_img(self):
        to_bgr = cv2.imread(os.path.join(self.file_path, "test_images/2216849948.jpg"))
        print("image information：", to_bgr.shape)
        to_rgb = to_bgr[:, :, ::-1]  # The reading is read in bgr format and converted to rgb format
        plt.imshow(to_rgb)
        plt.show()
    '''