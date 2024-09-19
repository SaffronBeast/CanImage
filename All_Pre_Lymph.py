{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyN8FuEI3gKjSe+IpSXOn/mV"
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "vDjyh0-mvg2G"
      },
      "outputs": [],
      "source": [
        "import os\n",
        "import torch\n",
        "from torch import nn, optim\n",
        "import torchvision.transforms as img_transforms\n",
        "from torch.utils.data import DataLoader\n",
        "from torchvision.datasets import ImageFolder\n",
        "from torchvision.utils import make_grid, save_image\n",
        "import matplotlib.pyplot as plt\n",
        "from tqdm.notebook import tqdm\n",
        "import cv2\n",
        "import zipfile\n",
        "from google.colab import files\n",
        "from IPython.display import Image"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Configurations and Constants\n",
        "DIRECTORY_PATH = '/content/drive/MyDrive/multi cancer image/MultiCancer/ALL3'\n",
        "IMG_DIMENSION = 64\n",
        "BATCH_SIZE = 128\n",
        "NORM_VALUES = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))\n",
        "RANDOM_SEED = 42\n",
        "LATENT_VECTOR_SIZE = 128"
      ],
      "metadata": {
        "id": "VUQShPw8vmc9"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Setting random seed for reproducibility\n",
        "torch.manual_seed(RANDOM_SEED)"
      ],
      "metadata": {
        "id": "YNAKgmvBvqg5"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Custom function to build the data loader\n",
        "def create_data_loader(path, img_size, batch, norm_values):\n",
        "    preprocess = img_transforms.Compose([\n",
        "        img_transforms.Resize(img_size),\n",
        "        img_transforms.CenterCrop(img_size),\n",
        "        img_transforms.ToTensor(),\n",
        "        img_transforms.Normalize(*norm_values)\n",
        "    ])\n",
        "    dataset = ImageFolder(root=path, transform=preprocess)\n",
        "    return DataLoader(dataset, batch_size=batch, shuffle=True, num_workers=4, pin_memory=True)\n"
      ],
      "metadata": {
        "id": "V2MPCM51vrSK"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Preparing the training data loader\n",
        "training_loader = create_data_loader(DIRECTORY_PATH, IMG_DIMENSION, BATCH_SIZE, NORM_VALUES)\n"
      ],
      "metadata": {
        "id": "-2t-AAcJvunC"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Function to denormalize image tensors\n",
        "def undo_normalization(tensor_image):\n",
        "    return tensor_image * NORM_VALUES[1][0] + NORM_VALUES[0][0]\n"
      ],
      "metadata": {
        "id": "GqhlKA70vwzb"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Displaying images in a grid format\n",
        "def visualize_images(imgs, max_images=64):\n",
        "    fig, ax = plt.subplots(figsize=(8, 8))\n",
        "    ax.axis('off')\n",
        "    ax.imshow(make_grid([undo_normalization(i) for i in imgs[:max_images]], nrow=8).permute(1, 2, 0))\n"
      ],
      "metadata": {
        "id": "ejPE4EZXvywC"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Previewing a batch from the data loader\n",
        "def preview_batch(data_loader, max_img=64):\n",
        "    imgs, _ = next(iter(data_loader))\n",
        "    visualize_images(imgs, max_img)"
      ],
      "metadata": {
        "id": "38v46mZ5v10B"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Preview a batch of training data\n",
        "preview_batch(training_loader)\n"
      ],
      "metadata": {
        "id": "XhltH141v4IL"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Select computation device: GPU if available, otherwise CPU\n",
        "comp_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')"
      ],
      "metadata": {
        "id": "wspF_fY6v54I"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Move data to selected device\n",
        "def shift_to_device(data, device):\n",
        "    if isinstance(data, (list, tuple)):\n",
        "        return [shift_to_device(d, device) for d in data]\n",
        "    return data.to(device, non_blocking=True)"
      ],
      "metadata": {
        "id": "7r5ilzWnv8Fq"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Class to handle loading data onto the selected device\n",
        "class DeviceLoaderWrapper:\n",
        "    def __init__(self, loader, device):\n",
        "        self.loader = loader\n",
        "        self.device = device\n",
        "\n",
        "    def __iter__(self):\n",
        "        for batch in self.loader:\n",
        "            yield shift_to_device(batch, self.device)\n",
        "\n",
        "    def __len__(self):\n",
        "        return len(self.loader)"
      ],
      "metadata": {
        "id": "mbzXSlbYv-JC"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Wrap the data loader to move batches to the selected device\n",
        "training_loader = DeviceLoaderWrapper(training_loader, comp_device)"
      ],
      "metadata": {
        "id": "-qBudroFwAMB"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Function to create a discriminator network\n",
        "def build_discriminator():\n",
        "    disc_net = nn.Sequential(\n",
        "        nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),\n",
        "        nn.BatchNorm2d(64),\n",
        "        nn.LeakyReLU(0.2, inplace=True),\n",
        "\n",
        "        nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),\n",
        "        nn.BatchNorm2d(128),\n",
        "        nn.LeakyReLU(0.2, inplace=True),\n",
        "\n",
        "        nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),\n",
        "        nn.BatchNorm2d(256),\n",
        "        nn.LeakyReLU(0.2, inplace=True),\n",
        "\n",
        "        nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),\n",
        "        nn.BatchNorm2d(512),\n",
        "        nn.LeakyReLU(0.2, inplace=True),\n",
        "\n",
        "        nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),\n",
        "        nn.Flatten(),\n",
        "        nn.Sigmoid()\n",
        "    )\n",
        "    return disc_net"
      ],
      "metadata": {
        "id": "mnF7uHIdwCYC"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Generator network using transposed convolutions\n",
        "class GeneratorNetwork(nn.Module):\n",
        "    def __init__(self, latent_dim):\n",
        "        super(GeneratorNetwork, self).__init__()\n",
        "        self.model = nn.Sequential(\n",
        "            nn.ConvTranspose2d(latent_dim, 512, kernel_size=4, stride=1, padding=0, bias=False),\n",
        "            nn.BatchNorm2d(512),\n",
        "            nn.ReLU(True),\n",
        "\n",
        "            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),\n",
        "            nn.BatchNorm2d(256),\n",
        "            nn.ReLU(True),\n",
        "\n",
        "            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),\n",
        "            nn.BatchNorm2d(128),\n",
        "            nn.ReLU(True),\n",
        "\n",
        "            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),\n",
        "            nn.BatchNorm2d(64),\n",
        "            nn.ReLU(True),\n",
        "\n",
        "            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),\n",
        "            nn.Tanh()\n",
        "        )\n",
        "\n",
        "    def forward(self, x):\n",
        "        return self.model(x)"
      ],
      "metadata": {
        "id": "pqN6bOMJwICh"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Instantiate and move the models to the selected device\n",
        "gen_model = GeneratorNetwork(LATENT_VECTOR_SIZE).to(comp_device)\n",
        "disc_model = build_discriminator().to(comp_device)"
      ],
      "metadata": {
        "id": "zq94D4f9wKaR"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Verify the output shape of the generator\n",
        "random_noise = torch.randn(BATCH_SIZE, LATENT_VECTOR_SIZE, 1, 1, device=comp_device)\n",
        "sample_fake_images = gen_model(random_noise)\n",
        "visualize_images(sample_fake_images)"
      ],
      "metadata": {
        "id": "0PJQYfqiwOkq"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Discriminator training function\n",
        "def train_discriminator(disc, real_data, optimizer):\n",
        "    optimizer.zero_grad()\n",
        "    real_outputs = disc(real_data)\n",
        "    real_labels = torch.ones(real_data.size(0), 1, device=comp_device)\n",
        "    loss_real = F.binary_cross_entropy(real_outputs, real_labels)\n",
        "    real_score = real_outputs.mean().item()\n",
        "\n",
        "    noise = torch.randn(BATCH_SIZE, LATENT_VECTOR_SIZE, 1, 1, device=comp_device)\n",
        "    fake_images = gen_model(noise)\n",
        "    fake_labels = torch.zeros(fake_images.size(0), 1, device=comp_device)\n",
        "    fake_outputs = disc(fake_images)\n",
        "    loss_fake = F.binary_cross_entropy(fake_outputs, fake_labels)\n",
        "    fake_score = fake_outputs.mean().item()\n",
        "\n",
        "    total_loss = loss_real + loss_fake\n",
        "    total_loss.backward()\n",
        "    optimizer.step()\n",
        "    return total_loss.item(), real_score, fake_score\n"
      ],
      "metadata": {
        "id": "DR8vcdkkwPIy"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Generator training function\n",
        "def train_generator(generator, discriminator, optimizer):\n",
        "    optimizer.zero_grad()\n",
        "    noise = torch.randn(BATCH_SIZE, LATENT_VECTOR_SIZE, 1, 1, device=comp_device)\n",
        "    fake_imgs = generator(noise)\n",
        "    predictions = discriminator(fake_imgs)\n",
        "    target_labels = torch.ones(BATCH_SIZE, 1, device=comp_device)\n",
        "    gen_loss = F.binary_cross_entropy(predictions, target_labels)\n",
        "    gen_loss.backward()\n",
        "    optimizer.step()\n",
        "    return gen_loss.item()"
      ],
      "metadata": {
        "id": "ujo_dgBOwRUB"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Save generated images during training\n",
        "def save_fake_samples(epoch, latent_vectors, show_images=True):\n",
        "    fake_samples = gen_model(latent_vectors)\n",
        "    filename = f'sample_images_{epoch:04d}.png'\n",
        "    save_image(undo_normalization(fake_samples), os.path.join(output_dir, filename), nrow=8)\n",
        "    if show_images:\n",
        "        plt.figure(figsize=(8, 8))\n",
        "        plt.imshow(make_grid(fake_samples.cpu().detach(), nrow=8).permute(1, 2, 0))\n",
        "        plt.axis('off')\n",
        "        plt.show()"
      ],
      "metadata": {
        "id": "vM158pbGwTGZ"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Set up a fixed set of random noise for generating consistent samples\n",
        "fixed_noise_vectors = torch.randn(64, LATENT_VECTOR_SIZE, 1, 1, device=comp_device)\n",
        "output_dir = 'generated_samples'\n",
        "os.makedirs(output_dir, exist_ok=True)\n"
      ],
      "metadata": {
        "id": "CyBpGp8UwWga"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Save initial generated images before training\n",
        "save_fake_samples(0, fixed_noise_vectors)\n"
      ],
      "metadata": {
        "id": "g_lddcMqwZ-p"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Training loop for the GAN\n",
        "def train_gan_model(epochs, learning_rate):\n",
        "    torch.cuda.empty_cache()\n",
        "    gen_loss_history, disc_loss_history, real_scores, fake_scores = [], [], [], []\n",
        "    disc_optimizer = optim.Adam(disc_model.parameters(), lr=learning_rate, betas=(0.5, 0.999))\n",
        "    gen_optimizer = optim.Adam(gen_model.parameters(), lr=learning_rate, betas=(0.5, 0.999))\n",
        "\n",
        "    for epoch in range(1, epochs + 1):\n",
        "        for batch_idx, (real_imgs, _) in enumerate(training_loader):\n",
        "            d_loss, r_score, f_score = train_discriminator(disc_model, real_imgs, disc_optimizer)\n",
        "            g_loss = train_generator(gen_model, disc_model, gen_optimizer)\n",
        "\n",
        "            if batch_idx % 50 == 0:\n",
        "                print(f\"Batch {batch_idx}/{len(training_loader)}, Disc_Loss: {d_loss:.4f}, Gen_Loss: {g_loss:.4f}\")\n",
        "\n",
        "        gen_loss_history.append(g_loss)\n",
        "        disc_loss_history.append(d_loss)\n",
        "        real_scores.append(r_score)\n",
        "        fake_scores.append(f_score)\n",
        "\n",
        "        save_fake_samples(epoch, fixed_noise_vectors, show_images=False)\n",
        "        print(f\"Epoch [{epoch}/{epochs}], Gen_Loss: {g_loss:.4f}, Disc_Loss: {d_loss:.4f}, Real_Score: {r_score:.4f}, Fake_Score: {f_score:.4f}\")\n",
        "\n",
        "    return gen_loss_history, disc_loss_history, real_scores, fake_scores\n"
      ],
      "metadata": {
        "id": "WrjMbXNPwcrr"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Hyperparameters for training\n",
        "LEARNING_RATE = 0.0002\n",
        "EPOCHS = 200"
      ],
      "metadata": {
        "id": "6jSep5YcwgGJ"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Start GAN training\n",
        "training_results = train_gan_model(EPOCHS, LEARNING_RATE)\n"
      ],
      "metadata": {
        "id": "FAcV_Woywgxa"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Function to zip and download the generated images\n",
        "def zip_and_download(directory, zip_filename):\n",
        "    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zf:\n",
        "        for root, _, files in os.walk(directory):\n",
        "            for file in files:\n",
        "                if not file.startswith('.'):\n",
        "                    zf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), directory))\n"
      ],
      "metadata": {
        "id": "Nte95CtdwjCB"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Zip and download the generated image samples\n",
        "zip_and_download(output_dir, 'sample_images.zip')\n",
        "files.download('sample_images.zip')"
      ],
      "metadata": {
        "id": "IKUWE9pMwmUh"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Save the models after training\n",
        "torch.save(gen_model.state_dict(), 'generator_model.pth')\n",
        "torch.save(disc_model.state_dict(), 'discriminator_model.pth')"
      ],
      "metadata": {
        "id": "21lD08utwoNS"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Display a sample generated image after training\n",
        "Image(f'./{output_dir}/sample_images_0001.png')"
      ],
      "metadata": {
        "id": "WYYdPB5RwqqS"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Create a video from generated images using OpenCV\n",
        "video_filename = 'training_evolution.avi'\n",
        "image_files = sorted([os.path.join(output_dir, f) for f in os.listdir(output_dir) if 'sample_images' in f])\n",
        "video_out = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'MP4V'), 1, (530, 530))\n",
        "for img in image_files:\n",
        "    video_out.write(cv2.imread(img))\n",
        "video_out.release()"
      ],
      "metadata": {
        "id": "uLuM2334wxVi"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Plotting loss curves for generator and discriminator\n",
        "plt.plot(disc_loss_history, '-')\n",
        "plt.plot(gen_loss_history, '-')\n",
        "plt.xlabel('Epoch')\n",
        "plt.ylabel('Loss')\n",
        "plt.legend(['Discriminator Loss', 'Generator Loss'])\n",
        "plt.title('Loss Progression')"
      ],
      "metadata": {
        "id": "vyScjtMFwx_h"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Plot real and fake scores\n",
        "plt.plot(real_scores, '-')\n",
        "plt.plot(fake_scores, '-')\n",
        "plt.xlabel('Epoch')\n",
        "plt.ylabel('Scores')\n",
        "plt.legend(['Real Score', 'Fake Score'])\n",
        "plt.title('Scores over Epochs')"
      ],
      "metadata": {
        "id": "GCgvwf24w0AZ"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}