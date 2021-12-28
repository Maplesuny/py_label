from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import xml.etree.ElementTree as ET
from scipy import signal
import neurokit2 as nk

import os
import sys
import psycopg2 as pg2
import json
import datetime


class xml_struct:
    def __init__(self, xml_root, cur=0):
        # xml_root is the product of ElementTree class
        self.name = xml_root.tag
        self.text = xml_root.text
        self.element = xml_root.attrib
        self.child = []
        self.child_name = []
        self.cur = cur
        if len(xml_root) > 0:
            for the_child in xml_root:
                self.child.append(xml_struct(the_child, self.cur+1))
                self.child_name.append(self.child[-1].name)
            self.child_name = dict(
                zip(self.child_name, list(range(len(self.child_name)))))
        self._concat()
        if self.cur == 0:
            self._to_element()

    def __len__(self):
        return len(self.child)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.child[idx]
        if isinstance(idx, str):
            if idx in self.child_name:
                return self.child[self.child_name[idx]]
            else:
                return self.element[idx]

    def _concat(self):
        if len(self.child) > 0 and not self.element:
            if any([len(the_child.child) > 0 for the_child in self.child]) or any([the_child.element for the_child in self.child]):
                return
            else:
                self.element = {}
                for the_child in self.child:
                    self.element[the_child.name] = the_child.text
                self.child = []
                self.child_name = []

    def _to_element(self):
        if self.text:
            self.element['text'] = self.text
        if len(self) > 0:
            for the_child in self:
                the_child._to_element()

    def get_element(self, name, key=None):
        if self.name == name:
            if key is not None:
                try:
                    return self.element[key]
                except:
                    print(f'The cell with name : {name} dont have key :{key} ')
                    return -1
            else:
                return self.element
        elif len(self.child) == 0:
            return -1
        else:
            for the_child in self.child:
                _tmp_text = the_child.get_element(name, key)
                if not _tmp_text == -1:
                    return _tmp_text
        return -1

    def to_Dataframe(self, name, the_tag):
        if self.name == name:
            try:
                keys = self[the_tag].element.keys()
                element = [the_child.element.values()
                           for the_child in self.child if the_tag == the_child.name]
                return pd.DataFrame(element, columns=keys)
            except:
                print('Cant build DataFrame')
                return -1
        elif len(self.child) == 0:
            return -1
        else:
            for the_child in self.child:
                _tmp_df = the_child.to_Dataframe(name, the_tag)
                if not type(_tmp_df) == int:
                    return _tmp_df
            return -1

    def print_item(self, cur=0):
        sky_blue, red, tail = '\x1b[96m', '\x1b[31m', '\x1b[0m'

        print('\t'*cur, self.name, ":")
        if self.element:
            for key, element in self.element.items():
                print('\t'*(cur+1),
                      f"{sky_blue}{key}{tail}\t: {red}{element}{tail}")
        if len(self.child) > 0:
            for the_child in self.child:
                the_child.print_item(cur+1)


class np_read_file:
    def __init__(self, filepath):
        self.offset = 0
        self.filepath = filepath

    def read(self, read_num, read_type):

        element = np.fromfile(self.filepath, count=read_num,
                              dtype=read_type, offset=self.offset)
        self.offset = self.offset+read_num*np.dtype(read_type).itemsize
        return element

# Input :
#     the top Folder of EEG raw data
# Output :
#     the address of the files we need


def get_subfile_path(the_path):
    eegFolderPath = the_path + '/'
    rdaFilePath = glob(eegFolderPath + 'EEGData/*.rda')[0]
    patientFilePath = os.path.join(eegFolderPath, 'EEG4PatientInfo.xml')
    sensorFilePath = glob(eegFolderPath + '*.sdy')[0]
    headerFilePath = os.path.join(eegFolderPath, 'EEGData', 'EEGData.ini')
    posFilePath = glob(eegFolderPath + 'ElectrodePlacements/*.xml')[0]
    eventFilePath = os.path.join(eegFolderPath, 'EEGStudyDB.mdb')
    deviceFilePath = os.path.join(
        eegFolderPath, "DeviceConfigurations", 'Grael LT - Default.xml')
    return eegFolderPath, rdaFilePath, patientFilePath, sensorFilePath, headerFilePath, posFilePath, eventFilePath, deviceFilePath

# Get the element in SensorFile
# Input :
#      SensorFile Address
# Outpput :
#     Sensor         : type = xml_struct ,for debug
#     Channels_info  : type = df , the channels info
#     chnames        : the_channels name
#     nchannels      : number of channels
#     sfreq          : Frequence of signal
#     record_start_time : start time of record (UTC+8)


def get_sensor_part(sensorFilePath):
    tree = ET.parse(sensorFilePath)
    root = tree.getroot()
    sensor = xml_struct(root)
    channels_info = sensor.to_Dataframe('Channels', 'Channel')
    chnames = channels_info.name.to_numpy()
    nchannels = len(channels_info)
    sfreq = float(sensor['Study']['eeg_sample_rate'])
    record_start_time = sensor['Study']['recording_start_time']

    return sensor, channels_info, chnames, nchannels, sfreq, record_start_time

# Get the element in PatientFile
# Input :
#     PatientFile Address
# Outpput :
#     Patient_Info   : type = xml_struct ,for debug
#     Patient_Name   : Name of patient
#     Patient_ID     : ID of patient
#     Patient_Birth  : Birth of patient


def get_patient_part(patientFilePath):
    tree = ET.parse(patientFilePath)
    root = tree.getroot()
    patient = xml_struct(root)
    patient_name = patient.get_element('GivenName', 'text')
    patient_ID = patient.get_element('Reference', 'text')
    patient_birth = patient.get_element('DOB', 'text')
    return patient, patient_name, patient_ID, patient_birth

# Get the header in RdaFile
# Input :
#     RdaFile Address
#     sfreq          : Frequence of signal , output from "get_sensor_part"
# Outpput :PatientFile
#     hdr_pos        : The offset of file reader
#     prop_times     : The head and tail of data we need


def get_rda_head(rdaFilePath, sfreq):
    f = np_read_file(rdaFilePath)
    rda_sealed = f.read(1, bool)[0]
    rda_pdel = f.read(1, 'int32')[0]
    unused = f.read(95, bool)
    rda_magic = f.read(1, "int64")[0]
    rda_first_sample = f.read(1, "int64")[0]
    rda_num_samples = f.read(1, "int64")[0]
    rda_closed = f.read(1, "bool")[0]
    unused = f.read(175, bool)
    hdr_pos = f.offset
    prop_times = np.array([0, rda_num_samples-1]) / sfreq
    return hdr_pos, prop_times

# Get the Signal in RdaFile
# Input :
#     RdaFile Address
#     nchannels     : Frequence of signal , output from "get_sensor_part"
#     prop_head     : First element in prop_times   , output from "get_rda_head"
#     prop_tail     : Second element in prop_times  , output from "get_rda_head"
#     sfreq         : Frequence of signal , output from "get_sensor_part"
#     hdr_pos       : The offset of file reader , output from "get_rda_head"
# Outpput :PatientFile
#     data          : The data we need


def get_rda_data(rdaFilePath, nchannels, prop_head, prop_tail, sfreq, hdr_pos):
    nChannels = float(nchannels)
    channelsRange = [0, nchannels]
    bytesPerVal = 4
    dataClass = 'single'
    nReadTimes = int(prop_tail*sfreq-prop_head*sfreq)
    nReadChannels = channelsRange[1]-channelsRange[0]

    offsetHeader = hdr_pos
    offsetTime = int(prop_head*nChannels*bytesPerVal)
    offsetChannelStart = int(channelsRange[0] * bytesPerVal)
    offsetChannelEnd = (nChannels - channelsRange[1]) * bytesPerVal
    offsetStart = offsetHeader + offsetTime + offsetChannelStart
    offsetSkip = offsetChannelStart + offsetChannelEnd

    data = []
    f = np_read_file(rdaFilePath)
    f.offset = offsetStart
    data = f.read(nReadTimes*nReadChannels, dataClass)
    data = np.reshape(data, [nReadTimes, nReadChannels]).T
    return data


def get_data_info(the_EEG_path, get_data=True):
    #EegFolderPath, RdaFile, PatientFile, SensorFile, HeaderFile, PosFile, EventFile, DeviceFile = get_subfile_path(the_EEG_path)
    eegFolderPath, rdaFilePath, patientFilePath, sensorFilePath, headerFilePath, posFilePath, eventFilePath, deviceFilePath = get_subfile_path(
        the_EEG_path)

    #Sensor_XML, Channels_info, chnames, nchannels, sfreq, record_start_time = get_sensor_part(sensorFilePath)
    sensor, channels_info, chnames, nchannels, sfreq, record_start_time = get_sensor_part(
        sensorFilePath)

    #Patient_XML, Patient_Name, Patient_ID, Patient_Birth = get_patient_part(patientFilePath)
    patient, patient_name, patient_ID, patient_birth = get_patient_part(
        patientFilePath)

    hdr_pos, prop_times = get_rda_head(rdaFilePath, sfreq)
    print('hdr_pos: ', hdr_pos)
    print('prop_times type: ', type(prop_times))
    print('prop_times: ', prop_times)
    print('prop_times_nick  ,', prop_times[1])
    print('length of prop_times : ', len(prop_times))

    patient_info = {
        'Patient_Name': patient_name,
        'Patient_ID': patient_ID,
        'Patient_Birth': patient_birth,
        'Record_Start_Time': record_start_time,
        'EegFolder': eegFolderPath,
        'rdaFile': rdaFilePath
    }
    rda_header = {
        'rdaFile': rdaFilePath,
        'chnames': ','.join(chnames),
        'nchannels': nchannels,
        'prop_head': prop_times[0],
        'prop_tail': prop_times[1],
        'sfreq': sfreq,
        'hdr_pos': hdr_pos,
    }
    if get_data:
        data = get_rda_data(rdaFilePath, nchannels,
                            prop_times[0], prop_times[1], sfreq, hdr_pos)
        return data, patient_info, rda_header
    else:
        return patient_info, rda_header


def simple_filter(data, fs, lowcut, highcut, notch):
    quality_factor = 20.0
    b_notch, a_notch = signal.iirnotch(notch, quality_factor, fs)
    freq, h = signal.freqz(b_notch, a_notch, fs=fs)
    data = signal.filtfilt(b_notch, a_notch, data)
    data = nk.signal_filter(signal=data, sampling_rate=fs, lowcut=lowcut,
                            highcut=highcut,  method="butterworth", powerline=60, order=3)
    return data


def calculateMontage(data, channel_dict, seg_num, plot_config):
    plot_sensitivity = plot_config['Sensitive']
    Seconds = plot_config['Time Window']
    the_Montage = plot_config['Montage']
    Montage_Input, Montage_Ref, Montage_Color, Montage_Name = the_Montage.Input, the_Montage.Ref, the_Montage.Colour, the_Montage.Name

    channel_range = 20*plot_sensitivity
    windows_size = int(plot_config['FS'] * Seconds)
    plot_channel_num = len(Montage_Input)

    calculatList = []

    for channel_idx, (the_input, the_ref, the_color) in enumerate(zip(Montage_Input, Montage_Ref, Montage_Color)):
        the_shift = (plot_channel_num-channel_idx)*channel_range
        #the_data = data[channel_dict[the_input], seg_num * windows_size : (seg_num+1) * windows_size]
        #the_data = the_data - data[channel_dict[the_ref], seg_num * windows_size : (seg_num+1) * windows_size]  if the_ref is not None else the_data
        the_data = data[channel_dict[the_input], :]
        the_data = the_data - data[channel_dict[the_ref],
                                   :] if the_ref is not None else the_data
        targetCalculatChannel = {}
        if the_ref is not None:
            targetCalculatChannel['id'] = the_input + '-' + the_ref
        else:
            targetCalculatChannel['id'] = the_input
        #d = dict(enumerate(the_data.flatten(), 1))
        targetCalculatChannel['value'] = list(the_data)
        calculatList.insert(channel_idx, targetCalculatChannel)
        # calculatList.append(targetCalculatChannel)
        #plt.plot(the_data + the_shift , c = Decimal_to_RGB(int(the_color)))
    return calculatList


'''
def get_data():
    print('=== start ===')
    #folder_path = '/home/leonard/project/leonard/t0.eeg'
    folder_path = '/tmp/t0.eeg'
    #Example_data, patient_info, rda_header  = get_data_info(folder_path , get_data=True)
    #print (Example_data.shape)

    patient_info, rda_header = get_data_info(folder_path, get_data=False)
    Example_data = get_rda_data(rda_header['rdaFile'], rda_header['nchannels'], rda_header['prop_head'], rda_header['prop_tail'], rda_header['sfreq'], rda_header['hdr_pos'])
    print('=== rda data ===')
    print(Example_data.shape)

    print('=== Patient Info ===')
    print(patient_info)

    print('=== RDA header ===')
    print(rda_header)

    # Select the signals which have valid record
    FS = float(rda_header['sfreq'])
    Channel_Name = rda_header['chnames'].split(',')
    Channel_Name = np.array(Channel_Name)
    invalid_index = np.array([the_name.lower()==the_name.upper() for the_name in Channel_Name])
    print('Invalid Channel',Channel_Name[invalid_index])
    Channel_Name = Channel_Name[~invalid_index]
    Example_data = Example_data[~invalid_index] 
    Channel_Dict = dict(zip(Channel_Name , np.arange(len(Channel_Name))))
    print('Channel_Dict', Channel_Dict)
    Example_data = Example_data*1e6*-1
    Example_data[Channel_Dict['ECG']] = Example_data[Channel_Dict['ECG']] * 1e-1
    print(Channel_Name.shape , Example_data.shape)
    print(Channel_Name)
    print(Example_data)

    Montage_type = ['A1-A2 Montage' , 'Cz Montage' , 'Double Banana']
    Example_type = 0
    the_montage_file = f'{folder_path}/Montages/10-20/{Montage_type[Example_type]}.xml'
    tree = ET.parse(the_montage_file)
    root = tree.getroot()
    Montage_xml_struct = xml_struct(root)
    Montage_channel_info = Montage_xml_struct.to_Dataframe('Traces' , 'Trace')
    Montage_Input , Montage_Ref , Montage_Color , Montage_Name =   Montage_channel_info.Input , Montage_channel_info.Ref , Montage_channel_info.Colour , Montage_channel_info.Name
    print('Montage_Input: ', Montage_Input)
    print('Montage_Ref: ', Montage_Ref)
    print('Montage_Color: ', Montage_Color)


    plot_config = {
        'FS' : FS , 
        'Sensitive' : 7 ,
        'Time Window' : 10 ,
        'Montage' : Montage_channel_info
    }

    seg_num = 7
    plot_segment_data(Example_data , seg_num, plot_config, Channel_Dict)
    #print('iiiiiiiiiiiiiii ', len(Example_data[20].tolist()))
    print('=== end ===')
    #return Example_data[0].tolist()
    return 'Example_data[0].tolist()'
'''


def get_data_sec():
    print('----------------get Sec-----------------')
    # 當前資料夾路徑
    curr_file_path = os.getcwd()
    folder_path = curr_file_path + '/t0.eeg'
    #EegFolderPath, RdaFile, PatientFile, SensorFile, HeaderFile, PosFile, EventFile, DeviceFile = get_subfile_path(the_EEG_path)
    eegFolderPath, rdaFilePath, patientFilePath, sensorFilePath, headerFilePath, posFilePath, eventFilePath, deviceFilePath = get_subfile_path(
        folder_path)

    #Sensor_XML, Channels_info, chnames, nchannels, sfreq, record_start_time = get_sensor_part(sensorFilePath)
    sensor, channels_info, chnames, nchannels, sfreq, record_start_time = get_sensor_part(
        sensorFilePath)

    #Patient_XML, Patient_Name, Patient_ID, Patient_Birth = get_patient_part(patientFilePath)
    patient, patient_name, patient_ID, patient_birth = get_patient_part(
        patientFilePath)

    # prop_times =>取得秒數
    hdr_pos, prop_times = get_rda_head(rdaFilePath, sfreq)

    # 取得筆數
    patient_info, rda_header = get_data_info(folder_path, get_data=False)
    rda_data = get_rda_data(rda_header['rdaFile'], rda_header['nchannels'], rda_header['prop_head'],
                            rda_header['prop_tail'], rda_header['sfreq'], rda_header['hdr_pos'])
    print('rda_data_per_len_nick', len(rda_data[0]))
    data_pen = len(rda_data[0])

    export_timeData = [data_pen, prop_times[1]]
    print('prop_times_nick  ,', prop_times[1])

    return export_timeData


def get_eeg_data(start_time, end_time, montage_type):
    print('=== start ===')

    # read data
    #folder_path = '/home/leonard/project/leonard/t0.eeg'
    # 當前資料夾路徑
    curr_file_path = os.getcwd()
    folder_path = curr_file_path + '/t0.eeg'
    print('dsfdfsdfdfsdf', folder_path)
    # folder_path = '/t0.eeg'
    patient_info, rda_header = get_data_info(folder_path, get_data=False)
    rda_data = get_rda_data(rda_header['rdaFile'], rda_header['nchannels'], rda_header['prop_head'],
                            rda_header['prop_tail'], rda_header['sfreq'], rda_header['hdr_pos'])
    print('rda_data type:', type(rda_data))
    print('rda_data len:', len(rda_data))
    print('rda_data_per_len', len(rda_data[0]))
    print(rda_data[:, start_time*512:end_time*512].shape)
    rda_data = rda_data[:, start_time*512:end_time*512]
    print('seg rda_data:', rda_data)
    # Select the signals which have valid record
    fs = float(rda_header['sfreq'])
    channel_name = rda_header['chnames'].split(',')
    channel_name = np.array(channel_name)
    invalid_index = np.array(
        [the_name.lower() == the_name.upper() for the_name in channel_name])

    # remove array by index
    channel_name = channel_name[~invalid_index]
    rda_data = rda_data[~invalid_index]
    channel_dict = dict(zip(channel_name, np.arange(len(channel_name))))
    rda_data = rda_data*1e6*-1
    rda_data[channel_dict['ECG']] = rda_data[channel_dict['ECG']] * 1e-1
    print('rda_data: ', len(rda_data[0]))
    print('fs:', fs)
    rda_data = simple_filter(rda_data, fs, lowcut=0.5, highcut=70, notch=60)
    # montage read file and parsing
    montage_list = ['A1-A2 Montage', 'Cz Montage', 'Double Banana']
    the_montage_file = f'{folder_path}/Montages/10-20/{montage_list[montage_type]}.xml'
    tree = ET.parse(the_montage_file)
    root = tree.getroot()
    montage_xml_struct = xml_struct(root)
    montage_channel_info = montage_xml_struct.to_Dataframe('Traces', 'Trace')
    montage_Input, montage_Ref, montage_Color, montage_Name = montage_channel_info.Input, montage_channel_info.Ref, montage_channel_info.Colour, montage_channel_info.Name

    window_time = end_time - start_time
    plot_config = {
        'FS': fs,
        'Sensitive': 7,
        'Time Window': window_time,
        'Montage': montage_channel_info
    }

    seg_num = 7

    result_list = calculateMontage(
        rda_data, channel_dict, seg_num, plot_config)
    return result_list


def get_patient_info(patient_id):
    # read data
    #folder_path = '/home/leonard/project/leonard/t0.eeg'
    folder_path = '/t0.eeg'
    patient_info, rda_header = get_data_info(folder_path, get_data=False)

    return patient_info

# if __name__ == "__main__":
