import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import gzip,pickle
from scipy.io import wavfile
from pandas import read_excel, read_csv
import pandas as pd
import zipfile
import _pickle as cPickle
import gzip
import os
import io
import json

import gzip
import shutil
import datetime
import time
# ---------------------------------------------------------------------------------------------------------------------
def read_raw_data(file_path):
    with gzip.open(file_path) as fp:
        datapkl_raw = pickle.load(fp)

    return datapkl_raw

# ---------------------------------------------------------------------------------------------------------------------
def creat_dictionnay():

    # List of all participants IDs
    ListPartcipants = ['221732']

    for ParticipentID in ListPartcipants:
        RawFilePath = 'data/Raw_Dataset/' + ParticipentID
        TagFilePath = 'data/Tags/Tags_' + ParticipentID + '.xlsx'

        # Read Wav files
        frAccX, dataAccX = wavfile.read(RawFilePath + '/acceleration_X.wav')
        frAccY, dataAccY = wavfile.read(RawFilePath + '/acceleration_Y.wav')
        frAccZ, dataAccZ = wavfile.read(RawFilePath + '/acceleration_Z.wav')
        frRespA, dataRespA = wavfile.read(RawFilePath + '/respiration_abdominal.wav')
        frRespT, dataRespT = wavfile.read(RawFilePath + '/respiration_thoracic.wav')

        # Read "statistics.csv" file
        statistics = pd.read_csv(RawFilePath + '/statistics.csv')

        # Substruct 4 Hours to get GMT-4 time
        HexoskinCorrectedTime = datetime.datetime.strptime(statistics.iloc[1, 1],'%Y-%m-%d %H:%M:%S.%f') - datetime.timedelta(hours=4)

        # Read the Tag events file
        Tags = pd.read_excel(TagFilePath)

        # Get start and the end time of the hexoskin device
        Hexoskin_Start_Time = datetime.datetime.strptime(Tags.iloc[0]['HexoskinTime'] ,'%H.%M.%S.%f').replace(year=HexoskinCorrectedTime.year,
                                                          month=HexoskinCorrectedTime.month,
                                                          day=HexoskinCorrectedTime.day)

        Hexoskin_End_Time   = datetime.datetime.strptime(Tags.iloc[-1]['HexoskinTime'],'%H.%M.%S.%f').replace(year=HexoskinCorrectedTime.year,
                                                          month=HexoskinCorrectedTime.month,
                                                          day=HexoskinCorrectedTime.day)

        # Get the start and the End time of the activity
        Start_Aquisition = round((Hexoskin_Start_Time - HexoskinCorrectedTime).total_seconds())
        End_Aquisition   = round((Hexoskin_End_Time - HexoskinCorrectedTime).total_seconds())+1

        # Calcul Offset between the Hexoskin and the video
        offset = datetime.datetime.strptime(Tags.iloc[0]['VideoTime'] ,'%H.%M.%S.%f') - datetime.datetime.strptime(Tags.iloc[0]['HexoskinTime'] ,'%H.%M.%S.%f')

        # Fill the correct hexoskin time events using the video time
        for i in np.arange(3,Tags.shape[0]-3):
            Tags.loc[i,'HexoskinTime'] = (datetime.datetime.strptime(Tags.iloc[i]['VideoTime'] ,'%H.%M.%S.%f') - offset).time().strftime("%H.%M.%S.%f")

        datapkl_raw = {
            'datatype_spec': {
                'acceleration_Y': {'freq': 64},
                'acceleration_X': {'freq': 64},
                'acceleration_Z': {'freq': 64},
                'respiration_abdominal': {'freq': 128},
                'respiration_thoracic' : {'freq': 128}
            },

            # This part will change One time we have all the subjects
            'record_specs': {ParticipentID: {'offset_Hexoskin_Video': str(offset),
                                             'Hexoskin_start_date'  : Hexoskin_Start_Time.strftime('%Y-%m-%d %H:%M:%S.%f')}},

            # This part will change One time we have all the subjects
            'data': {ParticipentID: {'acceleration_X': dataAccX[Start_Aquisition*64:End_Aquisition*64],
                                     'acceleration_Y': dataAccY[Start_Aquisition*64:End_Aquisition*64],
                                     'acceleration_Z': dataAccZ[Start_Aquisition*64:End_Aquisition*64],
                                     'respiration_abdominal': dataRespA[Start_Aquisition*128:End_Aquisition*128],
                                     'respiration_thoracic': dataRespT[Start_Aquisition*128:End_Aquisition*128]}
                     },
            # This part will change One time we have all the subjects
            'annotation': {ParticipentID: Tags}
        }

    PklName = 'data/Cough_dataset_2021/data_raw_2021.pkl'
    output = open(PklName, 'wb')
    pickle.dump(datapkl_raw, output)
    output.close()

    ZipName = 'data/Cough_dataset_2021/data_raw_2021.pkl.zip'
    with open(PklName, 'rb') as f_in:
        with gzip.open(ZipName, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    return datapkl_raw

# ---------------------------------------------------------------------------------------------------------------------
def run():
    # ---  Initialization  ---
    ColorDictionary = {'respiration_thoracic': '#1f77b4',
                       'respiration_abdominal': '#ff7f0e',
                       'acceleration_X': '#2ca02c',
                       'acceleration_Y': '#d62728',
                       'acceleration_Z': '#9467bd'}

    CoughColorDictionary = {0: '#581845',
                           1: '#d62728',
                           2: '#9467bd',
                           3: '#8c564b',
                           4: '#e377c2',
                           5: '#7f7f7f',
                           6: '#bcbd22',
                           7: '#17becf',
                           8: '#8c564b',
                           9: '#e377c2',
                           10: '#7f7f7f',
                           11: '#bcbd22',
                           12: '#17becf'}

    TypeOfCoughDictionary = {0: 'Tape début',
                            1: 'Minute respiration normal',
                            2: 'Toux volume normal',
                            3: 'Toux double',
                            4: 'Raclage de gorge',
                            5: 'Toux volume faible',
                            6: 'Rire 2-5 secondes',
                            7: 'Toux volume normal',
                            8: 'roles (3 phrases)',
                            9: 'Toux volume fort',
                            10: 'Cycle de respiration profonde',
                            11: 'Reniflement',
                            12: 'Tape fin'}

    #Creat Dictionnary
    datapkl_raw=creat_dictionnay()

    #la lecture du disctionnaire
    pkl_raw='data/Cough_dataset_2021/data_raw_2021.pkl.zip'
    datapkl_raw   = read_raw_data(pkl_raw)

    datatype_spec = datapkl_raw['datatype_spec']
    record_specs  = datapkl_raw['record_specs']
    patients_data = datapkl_raw['data']
    annotation    = datapkl_raw['annotation']


    for Patient in patients_data:
        fig = plt.figure()
        ax = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)
        ax3 = ax2.twinx()
        ax4 = ax.twinx()

        # Time in seconds for both type of signals
        Time1 = np.arange(0, len(patients_data[Patient]['acceleration_X'])) * (
                    1 / datatype_spec['acceleration_X']['freq'])
        Time2 = np.arange(0, len(patients_data[Patient]['respiration_thoracic'])) * (
                    1 / datatype_spec['respiration_thoracic']['freq'])

        # Read annotations
        Cough = annotation[Patient]

        HexoskinStart = datetime.datetime.strptime(Cough.loc[0, "HexoskinTime"], '%H.%M.%S.%f')

        for i in range(len(Cough)):
            x = np.where(Time1 <= (datetime.datetime.strptime(Cough.loc[i, "HexoskinTime"], '%H.%M.%S.%f') - HexoskinStart).total_seconds())
            pltCough1 = ax4.axvline(x=x[0][-1] * (1 / datatype_spec['acceleration_X']['freq']),
                                    label=TypeOfCoughDictionary[Cough.loc[i,'ActivityID']],
                                    c=CoughColorDictionary[Cough.loc[i,'ActivityID']], linestyle=':', linewidth=1.5, alpha=0.7)

            x = np.where(Time2 <=(datetime.datetime.strptime(Cough.loc[i, "HexoskinTime"], '%H.%M.%S.%f') - HexoskinStart).total_seconds())
            pltCough2 = ax2.axvline(x=x[0][-1] * (1 / datatype_spec['respiration_thoracic']['freq']),
                                    label=TypeOfCoughDictionary[Cough.loc[i,'ActivityID']],
                                    c=CoughColorDictionary[Cough.loc[i,'ActivityID']], linestyle=':', linewidth=1.5, alpha=0.7)

        pltAccX = ax.plot(Time1, patients_data[Patient]['acceleration_X'], label='acceleration_X', c=ColorDictionary['acceleration_X'], linewidth=0.5)
        pltAccY = ax.plot(Time1, patients_data[Patient]['acceleration_Y'], label='acceleration_Y', c=ColorDictionary['acceleration_Y'], linewidth=0.5)
        pltAccZ = ax.plot(Time1, patients_data[Patient]['acceleration_Z'], label='acceleration_Z', c=ColorDictionary['acceleration_Z'], linewidth=0.5)

        pltResT = ax2.plot(Time2, patients_data[Patient]['respiration_thoracic'], label='respiration_thoracic', c=ColorDictionary['respiration_thoracic'], linewidth=0.5)
        pltResA = ax3.plot(Time2, patients_data[Patient]['respiration_abdominal'], label='respiration_abdominal', c=ColorDictionary['respiration_abdominal'], linewidth=0.5)

        #ax.set_xlabel("Time (S)", color="black", fontsize=10)
        ax2.set_xlabel("Time (S)", color="black", fontsize=10)
        #ax2.set_xlabel("Time Respiration (S)", color="black", fontsize=10)
        ax.set_ylabel(r"Acceleration $(m/s^2)$", color="black", fontsize=10)
        ax2.set_ylabel("Respiration Thoracic", color="black", fontsize=10)
        ax3.set_ylabel("Respiration Abdominal", color="black", fontsize=10)

        ax.set_title("Acceleration", fontsize=12)
        ax2.set_title("Respiration", fontsize=12)
        fig.suptitle("Patient (" + str(Patient) + ")", fontsize=18)
        plt.grid()

        leg = pltAccX+pltAccY+pltAccZ+pltResT+pltResA
        labs = [l.get_label() for l in leg]
        Legend1 = ax.legend(leg, labs, ncol=5, bbox_to_anchor=(0.80, -1.35))

        for legobj in Legend1.legendHandles:
            legobj.set_linewidth(2.0)

        Legend2 = ax4.legend(*[*zip(*{l: h for h, l in zip(*ax4.get_legend_handles_labels())}.items())][::-1], ncol=8, bbox_to_anchor=(0.90, 1.20))
        for legobj in Legend2.legendHandles:
            legobj.set_linewidth(2.0)

        # Add second legend "Legend2" will be removed from figure
        ax.add_artist(Legend1)

        StartIter=0
        EndIter=len(patients_data[Patient]['acceleration_X'])*(1/64)
        majorStepIter=10
        minorStepIter = 5

        major_ticks = np.arange(StartIter, EndIter, majorStepIter)
        minor_ticks = np.arange(StartIter, EndIter, minorStepIter)
        ax.set_xticks(major_ticks)
        ax2.set_xticks(major_ticks)
        ax.tick_params(axis='x', rotation=45)
        ax2.tick_params(axis='x', rotation=45)
        ax.set_xticks(minor_ticks, minor=True)
        ax2.set_xticks(minor_ticks, minor=True)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax2.tick_params(axis='both', which='major', labelsize=8)
        ax3.tick_params(axis='both', which='major', labelsize=8)
        # ax3.spines["right"].set_position(("axes", 1.05))

        ax.grid(which='both')
        ax.grid(which='minor', alpha=0.2)
        ax.grid(which='major', alpha=0.5)
        ax2.grid(which='both')
        ax2.grid(which='minor', alpha=0.2)
        ax2.grid(which='major', alpha=0.5)

        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
        plt.show(block=False)

        # Save figure into specific folder
        MyFolder = "data/Figures/" + str(Patient) + "/"
        Path(MyFolder).mkdir(parents=True, exist_ok=True)
        filename = "Signals_Patient_" + str(Patient)
        fig.savefig(MyFolder + filename + ".png", dpi=600)  # Change is over here
        fig.savefig(MyFolder + filename + ".eps", format='eps')

        # Close Plot
        plt.close()

# ----------------------------------------------------------------------------------------------------------------------
# Main
if __name__ == '__main__':
    run()
