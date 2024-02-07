import os
import numpy as np
from PIL import Image, ImageEnhance
import shutil
import argparse

parser = argparse.ArgumentParser(description="Prepare Dataset")
parser.add_argument('-i', '--input-dir', metavar="input_dir", type=str, default="./data/cards", help="File path for the directory with the cards")
parser.add_argument('-ao', '--augmentation-output-dir', metavar="augmentation_output_dir", type=str, default="./data/augmented/", help="File path of the output directory for augmented images")
parser.add_argument('-so', '--split-output-dir', metavar="split_output_dir", type=str, default="./data/dataset/", help="File path of the output directory for split directory")
parser.add_argument('-a', '--augmentation', metavar="augmentation", type=bool, default=True, help="Option for the augmentation")
parser.add_argument('-v', '--validation', metavar="validation split ratio", type=float, default=0.1, help="Float for the validation split size")
parser.add_argument('-t', '--test', metavar="test split ratio", type=float, default=0.1, help="Float for the test split size")

def augment_dataset(card_path:str, output_path:str) -> None:
    """
    Augments card images to make multiple images with variations to the Brightness, Sharpness, Colour, Temperature and Orientation.

    Parameters
    ----------
    card_path : str
        A file path to the folder which contains the images to augment.
        This folder should contain all of the cards labelled like '[Value][Suit]' e.g. 2C, AD, KH, 6S
    output_path : str 
        A file path to the folder which the augmented images will save in
        This folder will structure the cards 

    return None
    """

    # Check Paths
    if not os.path.exists(card_path):
        raise Exception("The specified card path does not exist")

    if not os.path.exists(output_path):
        try:  
            os.mkdir(output_path)
        except OSError as error:  
            raise OSError(error)
    
    total_progress = 2
    total_progress = total_progress + total_progress * 5
    total_progress = total_progress + total_progress * 5
    total_progress = total_progress + total_progress * 7
    total_progress = total_progress * len(os.listdir((card_path)))

    counter = 1
    print(f"Augmentation Progress: {counter}/{total_progress}", end="\r")

    # Create Folder Structure, copy original and create flipped image
    for _, _, files in os.walk(card_path):
        for file in files:
            try:  
                os.mkdir(f"{output_path}\\{file.split('.')[0]}")
            except FileExistsError as error:  
                pass

            img = Image.open(f"{card_path}\\{file}")
            img.save(f"{output_path}\\{file.split('.')[0]}\\original.jpg")

            # Flip
            flipped = img.transpose(Image.FLIP_TOP_BOTTOM)
            flip = flipped.transpose(Image.FLIP_LEFT_RIGHT)
            flip.save(f"{output_path}\\{file.split('.')[0]}\\flipped.jpg")

            counter += 1
            print(f"Augmentation Progress: {counter}/{total_progress}", end="\r")

    # Brightness Augmentation for all images
    for _, dirs, files in os.walk(output_path):
        for folder in dirs:
            for file in os.listdir(f"{output_path}\\{folder}"):
                img = Image.open(f"{output_path}\\{folder}\\{file}")
                for val in np.arange(0.6, 1.6, 0.2):
                    enhancer = ImageEnhance.Brightness(img)
                    enhanced_img = enhancer.enhance(val)
                    enhanced_img.save(f"{output_path}\\{folder}\\bright-{str(round((val), 1))}-{file}")

                    counter += 1
                    print(f"Augmentation Progress: {counter}/{total_progress}", end="\r")

    # Sharpness Augmentation for all images
    for _, dirs, files in os.walk(output_path):
        for folder in dirs:    
            for file in os.listdir(f"{output_path}\\{folder}"):
                img = Image.open(f"{output_path}\\{folder}\\{file}")
                for val in np.arange(0.2, 1.7, 0.3):
                    enhancer = ImageEnhance.Sharpness(img)
                    enhanced_img = enhancer.enhance(val)
                    enhanced_img.save(f"{output_path}\\{folder}\\sharp-{str(round((val), 1))}-{file}")

                    counter += 1
                    print(f"Augmentation Progress: {counter}/{total_progress}", end="\r")

    # Colour temp
    # https://stackoverflow.com/questions/11884544/setting-color-temperature-for-a-given-image-like-in-photoshop - Colour temperature converter
    kelvin_table = {
        4000: (255,209,163),
        5000: (255,228,206),
        6000: (255,243,239),
        7000: (245,243,255),
        8000: (227,233,255),
        9000: (214,225,255),
        10000: (204,219,255)
    }

    def convert_temp(image, temp):
        r, g, b = kelvin_table[temp]
        matrix = ( r / 255.0, 0.0, 0.0, 0.0,
                0.0, g / 255.0, 0.0, 0.0,
                0.0, 0.0, b / 255.0, 0.0 )
        return image.convert('RGB', matrix)

    # Temperature Augmentation for all images
    for _, dirs, files in os.walk(output_path):
        for folder in dirs:    
            for file in os.listdir(f"{output_path}\\{folder}"):
                img = Image.open(f"{output_path}\\{folder}\\{file}")
                for temp in kelvin_table:
                    temp_img = convert_temp(img, temp)
                    temp_img.save(f"{output_path}\\{folder}\\temp-{str(temp)}-{file}")
                    
                    counter += 1
                    print(f"Augmentation Progress: {counter}/{total_progress}", end="\r")

    print("\nAugmentation Complete")


def split_dataset(data_dir, output_dir, val_ratio, test_ratio):
    """ Split the dataset into relevant folders for training, validation and testing

    Parameters
    ----------
    data_dir : str
        Filepath with the structured dataset path
    val_ratio : float
        Ratio for the percentage of the validation split
    test_ratio : float
        Ratio for the percentage of the test split

    return None
    """

    suits = ["C", "S", "H", "D"]
    card_values = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
    classes_dir = ["/Back"]

    num_files = sum([len(files) for _, _, files in os.walk(data_dir)])
    counter = 0

    print('Splitting...')
    print('Total images: ', num_files)
    print(' Training: ', num_files * (1-val_ratio-test_ratio))
    print(' Validation: ', num_files * val_ratio)
    print(' Testing: ', num_files * test_ratio)


    # Add each card combination to cards list
    for suit in suits:
        for value in card_values:
            classes_dir.append("/" + value + suit)

    # https://stackoverflow.com/questions/53074712/how-to-split-folder-of-images-into-test-training-validation-sets-with-stratified
    for cls in classes_dir:
        os.makedirs(output_dir +'/train' + cls)
        os.makedirs(output_dir +'/val' + cls)
        os.makedirs(output_dir +'/test' + cls)

        # Creating partitions of the data after shuffling
        src = data_dir + cls # Folder to copy images from

        all_file_names = os.listdir(src)
        np.random.shuffle(all_file_names)
        train_file_names, val_file_names, test_file_names = np.split(np.array(all_file_names),
                                                                [int(len(all_file_names)* (1 - (val_ratio + test_ratio))),
                                                                int(len(all_file_names)* (1 - test_ratio))])

        train_file_names = [src+'/'+ name for name in train_file_names.tolist()]
        val_file_names = [src+'/' + name for name in val_file_names.tolist()]
        test_file_names = [src+'/' + name for name in test_file_names.tolist()]

        # Copy-pasting images
        for name in train_file_names:
            shutil.copy(name, output_dir +'/train' + cls)
            counter += 1
            print(f"Splitting Progress: {counter}/{num_files}", end="\r")

        for name in val_file_names:
            shutil.copy(name, output_dir +'/val' + cls)
            counter += 1
            print(f"Splitting Progress: {counter}/{num_files}", end="\r")

        for name in test_file_names:
            shutil.copy(name, output_dir +'/test' + cls)
            counter += 1
            print(f"Splitting Progress: {counter}/{num_files}", end="\r")

    print("\nSplitting Complete")

if __name__ == "__main__":
    args = parser.parse_args()

    augment_dataset(args.input_dir, args.augmentation_output_dir)
    split_dataset(args.augmentation_output_dir, args.split_output_dir, args.validation, args.test)
