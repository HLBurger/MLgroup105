import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

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
    
