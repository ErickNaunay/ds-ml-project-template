"""
Módulo para limpieza y enriquecimiento (Feature Engineering) usando funciones simples.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    INSTRUCCIONES:
    1. Maneja los valores faltantes.
       Puedes llenarlos con la mediana de la columna.
    2. Retorna el DataFrame limpio.
    """
    df = df.copy()
    # Imputar total_bedrooms (única columna con faltantes) con la mediana
    for col in df.select_dtypes(include='number').columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    return df

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    INSTRUCCIONES:
    1. Agrega nuevas variables derivando las existentes, por ejemplo:
       - 'rooms_per_household' = total_rooms / households
       - 'population_per_household' = population / households
       - 'bedrooms_per_room' = total_bedrooms / total_rooms
    2. Retorna el DataFrame enriquecido.
    """
    df = df.copy()
    df['rooms_per_household'] = df['total_rooms'] / df['households']
    df['population_per_household'] = df['population'] / df['households']
    df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']
    return df

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Codifica variables categóricas con One-Hot Encoding (get_dummies).
    ocean_proximity es nominal (sin orden), por lo que OHE es la elección correcta.
    """
    df = pd.get_dummies(df, columns=['ocean_proximity'], dtype=float)
    return df

def scale_features(df: pd.DataFrame, target_col: str = 'median_house_value'):
    """
    Aplica StandardScaler a las variables numéricas (excepto la variable objetivo).
    Retorna el DataFrame escalado y el scaler ajustado (para usar en producción).
    """
    df = df.copy()
    feature_cols = [c for c in df.select_dtypes(include='number').columns if c != target_col]
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df, scaler

def preprocess_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Función orquestadora que toma el DataFrame crudo y aplica limpieza y enriquecimiento.
    """
    df_clean = clean_data(df)
    df_featured = create_features(df_clean)
    df_encoded = encode_categoricals(df_featured)
    return df_encoded

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3:
        input_path, output_path = sys.argv[1], sys.argv[2]
        df = pd.read_csv(input_path)
        df_processed = preprocess_pipeline(df)
        df_processed.to_csv(output_path, index=False)
        print(f"Procesado guardado en: {output_path}")
    else:
        print("Uso: python build_features.py <input_csv> <output_csv>")