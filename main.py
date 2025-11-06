import models.mnist_cnn as cnn
import models.mnist_transformers as trans
from models.DaViT import DaViT_MNIST
from train import train_model

def main():
    model = cnn.MNIST_Refer()
    train_model(model, epochs=20, save_path="./params/MNIST_ResNet.pth")

if __name__ == "__main__":
    main()
