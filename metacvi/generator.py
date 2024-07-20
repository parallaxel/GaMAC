import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from metacvi.collector import DatasetForMetaCVI, DatasetInfoCollector
from metacvi.producers import ProducerProvider
from metacvi.reducers import ReducerProvider


def launch(data_name):
    dataframe = pd.read_csv(f'data/{data_name}/orig.csv')
    data = dataframe.drop(columns=['class']).values
    normalised = MinMaxScaler().fit_transform(data)

    for reducer in ReducerProvider().get_all():
        print(f"===== {reducer.name} =====")
        dataset = DatasetForMetaCVI(data_name, reducer, normalised)
        collector = DatasetInfoCollector(dataset)
        for producer in ProducerProvider().get_all():
            print(f"--- {producer.name} ---")
            partition = producer.fit_predict(dataset)
            collector.register(partition, producer)
        collector.save()


if __name__ == "__main__":
    launch("wine-quality-red")
