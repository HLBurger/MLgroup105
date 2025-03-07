import pandas as pd

def add_path(data, path):
    data["path"] = path + "\\UTKFace\\UTKFace\\UTKFace\\" + data["files"]
    data = data.drop("files", axis = "columns")
    return data

def add_labels(data):
    # age_bins = [0, 12, 18, 25, 55, 70]
    # age_labels = ['Child', 'Teen', 'Young Adult', 'Adult', 'Elderly']
    age_bins = [0, 18, 25, 70]
    age_labels = ['Not Old Enough', 'Not Sure', 'Old Enough']
    data['age'] = pd.cut(
        data['age'], bins=age_bins, 
        labels=age_labels, right=True
    )
    return data


