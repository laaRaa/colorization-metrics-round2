import os

import pandas as pd


def main(
    hdfs_path: str = os.path.join("results", "dataframes", "unsplash.h5"),
    df_key: str = "df",
):
    try:
        # Lire le DataFrame depuis le fichier HDF5
        with pd.HDFStore(hdfs_path, mode="r") as store:
            if df_key in store:
                df = store[df_key]
            else:
                print(f"Aucune donnée trouvée pour la clé : {df_key}")
                return

        # Supprimer les doublons en gardant la première occurrence
        df_cleaned = df.drop_duplicates(
            subset=["Metric", "Image", "Alteration"], keep="first"
        )

        # Réécrire le fichier HDF5 avec le DataFrame nettoyé
        df_cleaned.to_hdf(
            hdfs_path, key=df_key, mode="w", format="table", data_columns=True
        )

        print("Fichier HDF5 réécrit avec succès sans doublons.")

    except Exception as e:
        print(f"Une erreur s'est produite : {e}")


if __name__ == "__main__":
    main()
