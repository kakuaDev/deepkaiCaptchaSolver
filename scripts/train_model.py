import glob
from deepkaiCaptchaSolver.core import CreateModel

if __name__ == "__main__":
    # Training image data path
    train_img_path_list = glob.glob("./train/*.png")

    # Training image data size
    img_width = 200
    img_height = 80

    # Creating an instance that creates a model
    CM = CreateModel(train_img_path_list, img_width, img_height)

    # Performing model training
    model = CM.train_model(epochs=8000)

    # Saving the weights learned by the model to a file
    model.save_weights("weights.h5")
