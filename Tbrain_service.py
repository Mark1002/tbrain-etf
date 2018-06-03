import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from stockstats import StockDataFrame
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
                
    def visual_predict_and_actual_plot(self, plot_info_list, test_y):
        handle_list = []
        actual_line, = plt.plot(test_y, label='actual')
        handle_list.append(actual_line)
        for plot_info in plot_info_list:
            line, = plt.plot(plot_info['pred'], label=plot_info['algo_name'])
            handle_list.append(line)
            print("{}, MSE:{}".format(plot_info['algo_name'], plot_info['MSE']))
        plt.legend(handles=handle_list)
                
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

class Util:
    def etf_data_preprocess(self, df):
        df['代碼'] = df['代碼'].str.strip()
        df['中文簡稱'] = df['中文簡稱'].str.strip()
        df['日期'] = pd.to_datetime(df['日期'])
        for col in df.columns[3:]:
            df[col] = df[col].map(lambda x: float("".join("".join(x.split()).split(','))))
        return df
    
    def decide_up_and_down(self, x):
        if x > 0:
            result = 1
        elif x < 0:
            result = -1
        else:
            result = 0
        return result
    
    def make_submission_record(self, etf_code, etf_df, pred):
        # 預測前上一個紀錄的值
        pev_price = etf_df.iloc[-1:, etf_df.columns == 'adj close'].values.flatten()
        r1 = np.round(np.concatenate([pev_price, pred])[:5],2)
        r2 = np.round(pred, 2)

        # make up submission csv
        up_list = list(map(self.decide_up_and_down, r2 - r1))
        cprice_list = r2.tolist()

        submission_records = []
        for i in range(1,11):
            if i % 2 == 1:
                submission_records.append(up_list.pop(0))
            else:
                submission_records.append(cprice_list.pop(0))
        submission_records = [int(etf_code)] + submission_records
        return submission_records
    
class Evaluation:
    def test_stationarity(self, timeseries):
        # Determing rolling statistics
        rolmean = timeseries.rolling(center=False,window=12).mean()
        # Plot rolling statistics:
        orig = plt.plot(timeseries, color='blue',label='Original')
        mean = plt.plot(rolmean, color='red', label='Rolling Mean')
        plt.legend(loc='best')
        plt.title('Rolling Mean')
        plt.show(block=False)

        # Perform Augmented Dickey-Fuller test:
        print('Results of Augmented Dickey-Fuller test:')
        dftest = adfuller(timeseries)
        dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
        for key,value in dftest[4].items():
            dfoutput['Critical Value (%s)'%key] = value
        print(dfoutput)

class FeatureExtraction:
    def perform_stock_stat(self, df):
        stock = StockDataFrame.retype(df.copy())
        # volume delta against previous day
        # stock['volume_delta']

        # open delta against next 2 day
        # stock['open_2_d']

        # open price change (in percent) between today and the day before yesterday
        # 'r' stands for rate.
        # stock['open_-2_r']

        # CR indicator, including 5, 10, 20 days moving average
        stock['cr']
        # stock['cr-ma1']
        # stock['cr-ma2']
        # stock['cr-ma3']

        # volume max of three days ago, yesterday and two days later
        # stock['volume_-3,2,-1_max']

        # volume min between 3 days ago and tomorrow
        # stock['volume_-3~1_min']

        # KDJ, default to 9 days
        # stock['kdjk']
        # stock['kdjd']
        # stock['kdjj']

        # 2 days simple moving average on open price
        stock['open_2_sma']

        # MACD
        stock['macd']
        # MACD signal line
        # stock['macds']
        # MACD histogram
        # stock['macdh']

        # bolling, including upper band and lower band
        # stock['boll']
        # stock['boll_ub']
        # stock['boll_lb']

        # close price less than 10.0 in 5 days count
        # stock['close_10.0_le_5_c']

        # CR MA2 cross up CR MA1 in 20 days count
        # stock['cr-ma2_xu_cr-ma1_20_c']

        # 6 days RSI
        # stock['rsi_6']
        # 12 days RSI
        stock['rsi_12']

        # 10 days WR
        stock['wr_10']
        # 6 days WR
        # stock['wr_6']

        # CCI, default to 14 days
        stock['cci']
        # 20 days CCI
        # stock['cci_20']

        # TR (true range)
        stock['tr']
        # ATR (Average True Range)
        stock['atr']

        # DMA, difference of 10 and 50 moving average
        stock['dma']

        # DMI
        # +DI, default to 14 days
        # stock['pdi']
        # -DI, default to 14 days
        # stock['mdi']
        # DX, default to 14 days of +DI and -DI
        # stock['dx']
        # ADX, 6 days SMA of DX, same as stock['dx_6_ema']
        # stock['adx']
        # ADXR, 6 days SMA of ADX, same as stock['adx_6_ema']
        # stock['adxr']

        # TRIX, default to 12 days
        # stock['trix']
        # MATRIX is the simple moving average of TRIX
        # stock['trix_9_sma']

        # VR, default to 26 days
        # stock['vr']
        # MAVR is the simple moving average of VR
        # stock['vr_6_sma']
        return stock