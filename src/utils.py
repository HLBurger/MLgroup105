# This file contains functions to preview the data
import pandas as pd
import random
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from torchinfo import summary
from torchviz import make_dot
from config import device


def plot_transformed_images(image_dataset: pd.DataFrame, transform, n=3, seed=None):
    """Selects random images from a path of images and loads/transforms them then plots the original vs the transformed version."""
    image_paths = list(image_dataset['path'])
    if seed:
        random.seed(seed)

    random_image_paths = random.sample(image_paths, k=n)
    for image_path in random_image_paths:
        with Image.open(image_path) as f:
            fig, ax = plt.subplots(nrows=1, ncols=2)
            # Original Image
            ax[0].imshow(f)
            ax[0].set_title(f'Original Image \nSize: {f.size}')
            ax[0].axis(False)

            # Transformed Image
            transformed_image = transform(f).permute(1, 2, 0) # (C, H, W) -> (H, W, C) for matplotlib
            ax[1].imshow(transformed_image)
            ax[1].set_title(f'Transformed Image \nSize: {transformed_image.shape}')
            ax[1].axis('off')

            # Here a list comprehension because suptitle trys to find class information
            image_class = [x for x in image_dataset[image_dataset["path"]==image_path]['age']][0]
            fig.suptitle(f'Class: {image_class}', fontsize=14)

# plot_transformed_images(image_dataset=train_ageutk,
#                         transform=train_transform,
#                         n=3,
#                         seed=None)


# Visualize images with their labels
def visualize_batch(images, labels):
    """
    Visualizes a batch of images in an 4x4 grid.
    
    Parameters:
        images (torch.Tensor): Batch of images (B, C, H, W).
        labels (list): Corresponding labels [[age_idx, gender_idx, emotion_idx], ...].
    """
    # Define the grid size
    rows, cols = 4, 4
    fig, axes = plt.subplots(rows, cols, figsize=(16, 20))
    
    for i, ax in enumerate(axes.flatten()):
        if i >= len(images):
            ax.axis('off')
            continue
        
        image = images[i]
        
        # Display the image
        ax.imshow(image.permute(1, 2, 0))  # (C, H, W) -> (H, W, C)
        ax.set_title(f"Age Id: {labels[i]}")
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

# train_images, train_labels = next(iter(train_dataloader_NN))
# print(train_images.shape, train_labels.shape, train_labels[:5])
# visualize_batch(train_images, train_labels)


def show_distributions(data):
    # Creëer de figuur en assen voor 2 plots naast elkaar
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Leeftijdsverdeling histogram
    axes[0].hist(data['age'], bins=30, color='grey', alpha=0.7, edgecolor='black')
    axes[0].axvline(x=18, color='red', linestyle='dashed', linewidth=4)  # Verticale rode stippellijn op 25
    axes[0].axvline(x=25, color='darkgreen', linestyle='dashed', linewidth=4)  # Verticale rode stippellijn op 25

    axes[0].set_xlabel('Age')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Age Distribution')
    axes[0].grid()

    # Plot 2: Genderverdeling histogram
    gender_counts = data['gender'].value_counts()
    gender_counts.plot(kind='bar', ax=axes[1], color=['blue', 'pink'], alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Gender')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Gender Distribution')
    # Handmatige legenda maken
    legend_patches = [
        mpatches.Patch(color='blue', label='Man'),
        mpatches.Patch(color='pink', label='Woman')
    ]
    axes[1].legend(handles=legend_patches, title="Gender", loc='upper center')


    # Totaal boven de boxen plaatsen
    for i, v in enumerate(gender_counts):
        axes[1].text(i, v - 800, str(v), ha='center', fontsize=12, fontweight='bold')  # Plaats het aantal boven de boxen

    # Toon de figuur
    plt.tight_layout()
    plt.show()

    # Groepeer de data in de drie leeftijdscategorieën en tel het aantal mannen en vrouwen per groep
    age_groups = {
        '<12': data[data["age"] < 12]['gender'].value_counts(),
        '12-18': data[(data["age"] > 12) & (data["age"] < 18)]['gender'].value_counts(),
        '18-25': data[(data["age"] > 18) & (data["age"] < 25)]['gender'].value_counts(),
        '>25': data[data["age"] > 25]['gender'].value_counts()
    }

    # Zet de data in een dataframe voor makkelijker plotten
    age_group_df = pd.DataFrame(age_groups).T.fillna(0)  # Vul missende waarden (geen Male/Female in een categorie) met 0

    # Plot de gegevens als een gegroepeerde bar chart
    fig, ax = plt.subplots(figsize=(6, 5))
    age_group_df.plot(kind='bar', ax=ax, color=['blue', 'pink'], alpha=0.7, edgecolor='black')

    # Labels en titel instellen
    ax.set_xlabel('Age Groups')
    ax.set_ylabel('Count')
    ax.set_title('Gender Distribution Across Age Groups')
    ax.legend(title='Gender', labels=["man", "woman"])

    # Totaal boven de boxen plaatsen
    for age_idx, (age_group, values) in enumerate(age_group_df.iterrows()):
        for gender_idx, value in enumerate(values):
            ax.text(age_idx + gender_idx * 0.2 - 0.1, value + 15, str(int(value)), ha='center', fontsize=10, fontweight='bold')

    plt.xticks(rotation=0)  # Zorgt ervoor dat de leeftijdsgroepen horizontaal blijven staan
    plt.tight_layout()
    plt.show()




def visualize_model(model_class, num_age_bins, input_size=128, device=device):
    """
    Instantiates the model, prints a summary, and generates a computation graph.
    
    Parameters:
    model_class (torch.nn.Module): The class of the model to instantiate.
    num_age_bins (int): The number of age bins for classification.
    input_size (int, optional): The input image size (default is 128).
    device (str, optional): The device to use (default is "cpu").
    
    Returns:
    torchviz.Digraph: The computation graph of the model.
    """
    # Print model summary
    model = model_class(num_age_bins=num_age_bins, input_size=input_size).to(device)
    print(summary(model, input_size=(1, 3, input_size, input_size)))
    
    # Create a dummy input tensor
    dummy_input = torch.randn(1, 3, input_size, input_size).to(device)
    
    # Perform a forward pass and visualize the computation graph
    output = model(dummy_input)
    model_graph = make_dot(output, params=dict(model.named_parameters()))
    
    # Save and return the graph
    model_graph.render("AgePredictionCNN_Graph", format="png", cleanup=True)
    return model_graph

# model_graph = visualize_model(AgePredictionCNN, num_age_bins=len(age_idx), input_size=128, device=device)
