import generate_embeddings
import pandas as pd
class AnalogyDB:
    def __init__(self, distance_metric, embedding_model, data, store_path, text_column):
        self.distance_metric = distance_metric
        self.embedding_model = embedding_model
        self.data = data
        self.store_path = store_path
        self.text_column = text_column

    def create_embeddings(self):
        data_with_embeddings = generate_embeddings.generate_embeddings(self.embedding_model, self.data)
        data_with_embeddings.save(self.store_path)

    def update_embeddings(self, new_data):
        data_with_embeddings = generate_embeddings.generate_embeddings(self.embedding_model, new_data)
        df = pd.read_csv(self.store_path)
        df = df.append(data_with_embeddings)
        df.save(self.store_path)

    def delete_embeddings(self,column_name_to_delete, column_value_to_delete):
         df = pd.read_csv(self.store_path)
         df = df[df[column_name_to_delete] == column_value_to_delete]
         df.save(self.store_path)
    
    def add_column(self, column_name, column_values):
        df = pd.read_csv(self.store_path)
        df[column_name] = column_values
        df.save(self.store_path)

    def delete_column(self, column_name):
        df = pd.read_csv(self.store_path)
        df = df.drop(column_name, axis=1)
        df.save(self.store_path)

         

db = AnalogyDB()
db.distance_metric = 'cosine'
db.embedding_model = 'paraphrase-MiniLM-L6-v2'
db.data = pd.read_csv('data.csv')
db.store_path = 'data_with.csv'
db.text_column = 'text'
db.create_embeddings()

    
