import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder


def load_cover_dataset(path, share=0.1):
    data = pd.read_csv(path, sep=",", header=None)

    data.columns = (
        [
            "Elevation",
            "Aspect",
            "Slope",
            "Horizontal_Distance_To_Hydrology",
            "Vertical_Distance_To_Hydrology",
            "Horizontal_Distance_To_Roadways",
            "Hillshade_9am",
            "Hillshade_Noon",
            "Hillshade_3pm",
            "Horizontal_Distance_To_Fire_Points",
        ]
        + ["Rawah", "Neota", "Comanche Peak", "Cache la Poudre"]
        + [
            "ELU2702",
            "ELU2703",
            "ELU2704",
            "ELU2705",
            "ELU2706",
            "ELU2717",
            "ELU3501",
            "ELU3502",
            "ELU4201",
            "ELU4703",
            "ELU4704",
            "ELU4744",
            "ELU4758",
            "ELU5101",
            "ELU5151",
            "ELU6101",
            "ELU6102",
            "ELU6731",
            "ELU7101",
            "ELU7102",
            "ELU7103",
            "ELU7201",
            "ELU7202",
            "ELU7700",
            "ELU7701",
            "ELU7702",
            "ELU7709",
            "ELU7710",
            "ELU7745",
            "ELU7746",
            "ELU7755",
            "ELU7756",
            "ELU7757",
            "ELU7790",
            "ELU8703",
            "ELU8707",
            "ELU8708",
            "ELU8771",
            "ELU8772",
            "ELU8776",
        ]
        + ["Cover_Type"]
    )

    label_frequency = dict(data["Cover_Type"].value_counts())
    label_shares = {2: 0.5, 1: 0.5, 3: 1, 7: 1, 6: 1, 5: 1.5, 4: 3}
    np.random.seed(101)
    data = pd.concat(
        [
            data.loc[
                np.random.choice(
                    data.loc[data["Cover_Type"] == label, "Cover_Type"].index,
                    size=int(label_frequency[label] * label_share * share),
                    replace=int((label_share * share) > 1),
                )
            ]
            for label, label_share in label_shares.items()
        ],
        axis=0,
    )
    return (data[:, :-1].values, data[:, -1].values)
