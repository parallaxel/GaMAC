import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from metacvi.collector import DatasetForMetaCVI, DatasetInfoCollector
from metacvi.producers import ProducerProvider
from metacvi.reducers import ReducerProvider


def launch(data_name):
    dataframe = pd.read_csv(f'data/{data_name}/orig.csv')
    data = dataframe.drop(columns=['class']).values
    normalised = MinMaxScaler().fit_transform(data)

    for reducer in ReducerProvider.get_all():
        print(f"===== {reducer.name} =====")
        try:
            dataset = DatasetForMetaCVI(data_name, reducer, normalised)
        except ValueError:
            continue
        collector = DatasetInfoCollector(dataset)
        for producer in ProducerProvider.get_all():
            print(f"--- {producer.name} ---")
            partition = producer.fit_predict(dataset)
            collector.save(partition, producer)
        collector.persist()

# READY:
# 100-plants-margin
# abalone
# banknote-authentication
# blocks
# blood-transfusion-service
# cardiotocography
# character
# climate-model-simulation-crashes
# cnae-9
# diabetes
# eye-movements
# first-order-theorem-proving
# gesture-phase-segmentation-processed
# gina-agnostic
# hiva-agnostic
# kc1
# madelon
# mammography
# mfeature
# micro-mass
# optdigits
# ozone-level-8hr
# pc4
# pendigits
# phoneme
# pollen
# qsar-biodeg
# satellite
# satimage
# segment
# semeion
# spambase
# speech
# steel-plates-fault
# texture
# vehicle
# volcanoes-d4
# wall-robot-navigation
# waveform-5000
# wdbc
# wilt
# wine-quality-red
# wine-quality-white
# yeast


# MISSING MDS
# ailerons (-mds)
# eeg-eye-state (-mds)
# gas-drift (-mds)
# sylva-prior (-mds)
# visualizing-soil (-mds)


# GENERATE FROM SCRATCH
# covertype
# elevators
# fried
# har
# helena
# houses
# jannis
# letter
# numerai-28-6
# shuttle
# walking-activity


if __name__ == "__main__":
    launch("elevators")
