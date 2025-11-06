from models.mnist_cnn import MNIST_ResNet
from train import train_model

def main():
    model = MNIST_ResNet()
    train_model(model, epochs=20, save_path="./params/MNIST_ResNet.pth")

if __name__ == "__main__":
    main()
