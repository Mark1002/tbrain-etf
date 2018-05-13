import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

class DataVisualization:
    def compare_predict_and_actual_plot(self, pred, actual):
        plt.figure(1)
        plt.subplot(211)
        plt.title("predcit price")
        plt.plot(pred)
        plt.subplot(212)
        plt.title("actual price")
        plt.plot(actual)
        plt.tight_layout()
        plt.show()
      
    def visualize_feature(self, df):
        # visualize feature
        for i, col_name in enumerate(df.columns):
            plot_index = i % 4
            if plot_index == 0:
                plt.figure(1)
            plt.subplot(221 + plot_index)
            plt.title(str(i) + "," + col_name)
            plt.plot(df[col_name])
            plt.xlabel("Time")
            plt.grid(True)
            if plot_index == 3:
                plt.tight_layout()
                plt.show()
            elif i == len(df.columns)-1:
                plt.tight_layout()
                plt.show()
                
class DataPreproecss:
    def trans_time_series_to_supervised(self, df, shift_range, target_feature):
        df = df.copy()
        # shift 時間序列轉換成監督式學習 X, y 回歸預測任務
        df['y'] = df.shift(-shift_range)[target_feature]
        df = df.dropna()
        return df
    
    def make_slide_windows(self, df, windows_size):
        columns = df.columns
        df_window = df.copy()
        for i in range(1, windows_size):
            shift_df = df.shift(-i)
            shift_df.columns = columns + "+" + str(i)
            df_window = pd.concat([df_window, shift_df], axis=1).dropna()
        return df_window
    
    def standardize(self, df):
        scaler = StandardScaler()
        scaler.fit(df)
        df_scale = scaler.transform(df)
        return df_scale, scaler

class MongoBase:
    def __init__(self, uri, db_name):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        
    def insert_document(self, collection_name, df):
        collection = self.get_collection(collection_name)
        return collection.insert_many(df.T.to_dict().values())
    
    def get_collection(self, collection_name):
        return self.db[collection_name]
    
    def delete_collection(self, collection_name):
        return self.db[collection_name].drop()
    
    def select_document(self, collection_name, query={}, no_id=True):
        collection = self.get_collection(collection_name)
        if collection.count()==0:
            return None
        df = pd.DataFrame(list(collection.find()))
        if no_id:
            del df['_id']
        return df
    
    def close_db(self):
        self.client.close()
