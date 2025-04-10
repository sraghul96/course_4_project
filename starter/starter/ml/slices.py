import pandas as pd


def data_slices(data: pd.DataFrame, label: str) -> dict:
    """
    Function to create slices of the data for each categorical feature.

    Inputs
    ------
    data : pd.DataFrame
        Dataframe containing the features and label.
    label : str
        Name of the label column in `data`.

    Returns
    -------
    slices : dict
        Dictionary containing the slices of the data for each categorical feature.
    """

    slices = {}

    # Define age buckets
    age_bins = [0, 18, 29, 44, 59, 100]
    age_labels = ["0-18", "19-29", "30-44", "45-59", "60+"]
    data["age_data_slice"] = pd.cut(
        data["age"],
        bins=age_bins,
        labels=age_labels,
        right=True)
    slices["age_data_slice"] = {
        label: data[data["age_data_slice"] == label] for label in age_labels}

    # Define marital status buckets
    marital_status_mapping = {
        " Married-civ-spouse": "Married",
        " Married-spouse-absent": "Married",
        " Married-AF-spouse": "Married",
        " Divorced": "Divorced",
        " Never-married": "Never-married",
        " Separated": "Separated",
        " Widowed": "Widowed",
    }
    data["marital_status_data_slices"] = data["marital-status"].map(
        lambda x: marital_status_mapping[x].replace(" ", "")
    )
    slices["marital_status_data_slice"] = {
        label: data[data["marital_status_data_slices"] == label] for label in marital_status_mapping.values()
    }

    # Define Race buckets
    data["race_data_slices"] = pd.Categorical(
        data["race"], categories=data["race"].unique())
    slices["race_data_slice"] = {
        label: data[data["race_data_slices"] == label] for label in data["race"].unique()}

    # Define Sex buckets
    data["sex_data_slices"] = pd.Categorical(
        data["sex"], categories=data["sex"].unique())
    slices["sex_data_slice"] = {
        label: data[data["sex_data_slices"] == label] for label in data["sex"].unique()}

    # Define combined buckets for occupation and workclass
    # data["occupation_workclass_bucket"] = data["occupation"] + "_" + data["workclass"]
    # data["occupation_workclass_data_slices"] = pd.Categorical(
    #     data["occupation_workclass_bucket"], categories=data["occupation_workclass_bucket"].unique()
    # )
    # slices["occupation_workclass_data_slice"] = {
    #     label: data[data["occupation_workclass_data_slices"] == label]
    #     for label in data["occupation_workclass_bucket"].unique()
    # }

    # Define native country buckets
    country_mapping = {
        "United-States": "North America",
        "Canada": "North America",
        "Mexico": "North America",
        "Puerto-Rico": "North America",
        "Outlying-US(Guam-USVI-etc)": "North America",
        "Cuba": "Central and South America",
        "Jamaica": "Central and South America",
        "Dominican-Republic": "Central and South America",
        "El-Salvador": "Central and South America",
        "Guatemala": "Central and South America",
        "Honduras": "Central and South America",
        "Nicaragua": "Central and South America",
        "Peru": "Central and South America",
        "Ecuador": "Central and South America",
        "Columbia": "Central and South America",
        "Trinadad&Tobago": "Central and South America",
        "England": "Europe",
        "Germany": "Europe",
        "Greece": "Europe",
        "Italy": "Europe",
        "Poland": "Europe",
        "Portugal": "Europe",
        "Ireland": "Europe",
        "France": "Europe",
        "Scotland": "Europe",
        "Yugoslavia": "Europe",
        "Holand-Netherlands": "Europe",
        "Cambodia": "Asia",
        "India": "Asia",
        "Japan": "Asia",
        "China": "Asia",
        "Iran": "Asia",
        "Philippines": "Asia",
        "Vietnam": "Asia",
        "Laos": "Asia",
        "Taiwan": "Asia",
        "Thailand": "Asia",
        "Hong": "Asia",
        "South": "Other",
        "Haiti": "Other",
        "Hungary": "Other",
    }

    country_mapping_space = country_mapping
    country_mapping_space = {
        f" {k}": v for k,
        v in country_mapping_space.items()}

    data["country_data_slices"] = data["native-country"].map(
        lambda x: country_mapping_space[x] if x in country_mapping_space else "Other")
    slices["country_data_slice"] = {
        label: data[data["country_data_slices"] == label] for label in country_mapping_space.values()
    }

    # Combine age buckets and education into a new column
    # data["age_education_data_slices"] = data["age_data_slice"].astype(str) + "_" + data["education"]
    # slices["age_education_data_slice"] = {
    #     label: data[data["age_education_data_slices"] == label] for label in data["age_education_data_slices"].unique()
    # }

    return slices
