import psycopg2
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Configurações do banco de dados
# Usado somente para salvar localmente um dataset com as informações do BD
DB_CONFIG = {
    "dbname": "santoscart",
    "user": "postgres",
    "password": "********",
    "host": "********",
    "port": "5432"
}

# Nome do arquivo para armazenamento local dos dados históricos
LOCAL_DATA_FILE = "product_history.csv"

# Função para conectar ao banco de dados PostgreSQL
def connect_db():
    """Estabelece conexão com o banco de dados usando as configurações definidas."""
    return psycopg2.connect(**DB_CONFIG)

# Função para buscar dados históricos dos últimos 6 meses
def fetch_historical_data():
    """Recupera os dados históricos dos últimos 6 meses do banco de dados.
    Se os dados já estiverem armazenados localmente, carrega a partir do arquivo CSV."""
    if os.path.exists(LOCAL_DATA_FILE):
        print("Carregando dados locais...")
        return pd.read_csv(LOCAL_DATA_FILE, parse_dates=["datetime"])
    
    print("Baixando dados do banco de dados...")
    conn = connect_db()
    query = """
    SELECT 
        ph.product_barcode, 
        ph.price, 
        ph.location, 
        ph.datetime,
        p.name AS product_name
    FROM product_history ph
    JOIN product p ON p.barcode = ph.product_barcode
    WHERE ph.datetime >= NOW() - INTERVAL '6 months'
    ORDER BY ph.datetime DESC;
    """
    df = pd.read_sql(query, conn)
    conn.close()
    
    df.to_csv(LOCAL_DATA_FILE, index=False)
    print(f"Dados salvos localmente em {LOCAL_DATA_FILE}")
    return df

# Função para preparar os dados para o modelo de classificação
def prepare_data(df):
    """Transforma os dados em um formato adequado para treinamento do modelo de machine learning."""
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['timestamp'] = df['datetime'].astype('int64') // 10**9  # Converte datetime para timestamp
    df['location_code'] = df['location'].astype('category').cat.codes  # Converte local para código numérico
    X = df[['timestamp', 'location_code']]
    
    # Determina o melhor local baseado no menor preço por produto
    df['best_location'] = df.groupby('product_barcode', group_keys=False)['price'].transform(
        lambda x: df.loc[x.idxmin(), 'location']
    )
    df['best_location'] = df['best_location'].astype('category')
    y = df['best_location'].cat.codes  # Converte a variável de saída para formato numérico
    return X, y, df['best_location'].cat.categories

# Função para treinar um modelo de classificação para um produto específico
def train_model_for_product(X, y, product_name):
    """Treina um modelo de Random Forest para prever o melhor local para um produto."""
    if len(X) < 2:
        print("Dados insuficientes para treinar o modelo.")
        return None
    
    test_size = min(0.2, max(0.1, 1 - 1 / len(X)))  # Garante que sempre há pelo menos um dado para treino
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Treinando modelo para {product_name}")
    
    return model

# Função para recomendar o melhor local para todos os produtos
def recommend_best_locations(df):
    """Para cada produto, treina um modelo e recomenda a melhor localização baseada nos dados."""
    results = []
    products = df['product_barcode'].unique()
    
    for product_barcode in products:
        product_data = df[df['product_barcode'] == product_barcode].copy()
        product_name = product_data['product_name'].iloc[0]
        
        if len(product_data) < 2:
            print(f"Produto {product_name} ({product_barcode}) tem poucos dados. Ignorando.")
            continue
        
        X, y, location_categories = prepare_data(product_data)
        model = train_model_for_product(X, y, product_name)
        if model is None:
            continue
        
        future_timestamp = pd.Timestamp.now().timestamp()
        X_pred = pd.DataFrame({'timestamp': [future_timestamp] * len(location_categories),
                               'location_code': range(len(location_categories))})
        if X_pred.empty:
            continue
        
        predicted_location_code = model.predict(X_pred[['timestamp', 'location_code']])[0]
        predicted_location = location_categories[predicted_location_code]
        results.append({'product_barcode': product_barcode, 'product_name': product_name, 'best_location': predicted_location})
    
    return pd.DataFrame(results)

# Função para salvar os resultados em um arquivo Excel
def save_to_excel(df, file_path):
    """Salva os resultados das recomendações em um arquivo Excel."""
    df.to_excel(file_path, index=False)
    print(f"Arquivo salvo em: {file_path}")

# Função principal do script
def main():
    """Executa o fluxo completo: coleta dados, treina modelos e salva recomendações."""
    historical_data = fetch_historical_data()
    recommendations = recommend_best_locations(historical_data)
    if not recommendations.empty:
        save_to_excel(recommendations, 'purchase_recommendations.xlsx')


if __name__ == "__main__":
    main()
