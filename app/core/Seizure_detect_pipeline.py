import os
import sys
import mne
import numpy as np
import pandas as pd
from glob import glob
from scipy import signal
from tqdm.auto import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from PIL import Image
from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs,
                               corrmap)
from mne.time_frequency import tfr_morlet, psd_multitaper, psd_welch, tfr_stockwell
import tensorflow as tf
import tensorflow.keras.layers as L
import efficientnet.tfkeras as efn


# 院內RDA檔轉EDF
class xml_struct:
    def __init__(self, xml_root, cur=0):
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


def ismember(A, B):
    return np.array([np.sum(a == B) for a in A]).astype(bool)


def get_subfile_path(the_path):
    EegFolder = the_path+'/'
    RdaFile = glob(EegFolder + 'EEGData/*.rda')[0]
    PatientFile = os.path.join(EegFolder, 'EEG4PatientInfo.xml')
    SensorFile = glob(EegFolder + '*.sdy')[0]
    HeaderFile = os.path.join(EegFolder, 'EEGData', 'EEGData.ini')
    PosFile = glob(EegFolder + 'ElectrodePlacements/*.xml')[0]
    EventFile = os.path.join(EegFolder, 'EEGStudyDB.mdb')
    DeviceFile = os.path.join(
        EegFolder, "DeviceConfigurations", 'Grael LT - Default.xml')
    return EegFolder, RdaFile, PatientFile, SensorFile, HeaderFile, PosFile, EventFile, DeviceFile


def get_sensor_part(SensorFile):
    tree = ET.parse(SensorFile)
    root = tree.getroot()
    Sensor = xml_struct(root)
    Channels_info = Sensor.to_Dataframe('Channels', 'Channel')
    chnames = list(Channels_info.name.to_numpy())
    nchannels = len(Channels_info)
    sfreq = float(Sensor['Study']['eeg_sample_rate'])
    record_start_time = Sensor['Study']['recording_start_time']
    return Sensor, Channels_info, chnames, nchannels, sfreq, record_start_time


def get_patient_part(PatientFile):
    tree = ET.parse(PatientFile)
    root = tree.getroot()
    Patient_Info = xml_struct(root)
    Patient_Name = Patient_Info.get_element('GivenName', 'text')
    Patient_ID = Patient_Info.get_element('Reference', 'text')
    Patient_Birth = Patient_Info.get_element('DOB', 'text')
    return Patient_Info, Patient_Name, Patient_ID, Patient_Birth


def get_pos_part(PosFile, chnames):
    tree = ET.parse(PosFile)
    root = tree.getroot()
    Pos = xml_struct(root)
    Pos_info = Pos.to_Dataframe('Electrodes', 'Electrode')
    Pos_info = Pos_info[ismember(Pos_info.Label, chnames)]
    chnames_pos = Pos_info.Label.to_numpy()
    chposX = Pos_info.XCoordinate.to_numpy().astype(float)
    chposY = Pos_info.YCoordinate.to_numpy().astype(float)
    chpos = Pos_info[['XCoordinate', 'YCoordinate']].astype(float)
    chpos = chpos-chpos.mean()
    chpos = chpos / chpos.abs().max()
    return Pos, Pos_info,  chnames_pos, chpos


def get_rda_head(RdaFile, sfreq):
    f = np_read_file(RdaFile)
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


def get_rda_data(RdaFile, nchannels, prop_head, prop_tail, sfreq, hdr_pos):
    nChannels = float(nchannels)
    ChannelsRange = [0, nchannels]
    bytesPerVal = 4
    dataClass = 'single'
    nReadTimes = int(prop_tail*sfreq-prop_head*sfreq)
    nReadChannels = ChannelsRange[1]-ChannelsRange[0]

    offsetHeader = hdr_pos
    offsetTime = int(prop_head*nChannels*bytesPerVal)
    offsetChannelStart = int(ChannelsRange[0] * bytesPerVal)
    offsetChannelEnd = (nChannels - ChannelsRange[1]) * bytesPerVal
    offsetStart = offsetHeader + offsetTime + offsetChannelStart
    offsetSkip = offsetChannelStart + offsetChannelEnd

    data = []
    f = np_read_file(RdaFile)
    f.offset = offsetStart
    data = f.read(nReadTimes*nReadChannels, dataClass)
    data = np.reshape(data, [nReadTimes, nReadChannels]).T
    return data


def get_data_info(the_EEG_path, get_data=True):
    EegFolder, RdaFile, PatientFile, SensorFile, HeaderFile, PosFile, EventFile, DeviceFile = get_subfile_path(
        the_EEG_path)
    Sensor_XML, Channels_info, chnames, nchannels, sfreq, record_start_time = get_sensor_part(
        SensorFile)
    Patient_XML, Patient_Name, Patient_ID, Patient_Birth = get_patient_part(
        PatientFile)
    Pos_XML, Pos_info,  chnames_pos, chpos = get_pos_part(PosFile, chnames)
    hdr_pos, prop_times = get_rda_head(RdaFile, sfreq)
    ch_names = chnames
    Patient_Info = {
        'Patient_Name': Patient_Name,
        'Patient_ID': Patient_ID,
        'Patient_Birth': Patient_Birth,
        'Record_Start_Time': record_start_time,
        'EegFolder': EegFolder,
        'RdaFile': RdaFile
    }
    rda_header = {
        'RdaFile': RdaFile,
        'chnames': ','.join(chnames),
        'nchannels': nchannels,
        'prop_head': prop_times[0],
        'prop_tail': prop_times[1],
        'sfreq': sfreq,
        'hdr_pos': hdr_pos,
    }
    if get_data:
        data = get_rda_data(RdaFile, nchannels,
                            prop_times[0], prop_times[1], sfreq, hdr_pos)
        return data, Patient_Info, rda_header, ch_names
    else:
        return Patient_Info, rda_header, ch_names


def RDA2MNE(path):
    data, Patient_Info, RDA_header, ch_names = get_data_info(
        path, get_data=True)
    info = mne.create_info(
        ch_names=ch_names,
        ch_types=['eeg']*RDA_header['nchannels'],
        sfreq=RDA_header['sfreq']
    )
    data = mne.io.RawArray(data, info)
    return data, Patient_Info['Patient_ID']


# EEG Preprocessing
def EEG_PreProcessing(data):
    raw_copy = data.copy()
    raw_pick = raw_copy.copy().pick_channels(['Fp1', 'Fp2', 'F3', 'F4',
                                              'C3', 'C4', 'P3', 'P4',
                                              'O1', 'O2', 'F7', 'F8',
                                              'T3', 'T4', 'T5', 'T6',
                                             'Fz', 'Cz',
                                              'Pz'], ordered=True)
    raw_pick = raw_pick.load_data()
    raw_doublebanana = mne.set_bipolar_reference(raw_pick.copy(),
                                                 anode=['Fp1',
                                                        'F3',
                                                        'C3',
                                                        'P3',
                                                        'Fp1',
                                                        'F7',
                                                        'T3',
                                                        'T5',
                                                        'Fz',
                                                        'Fp2',
                                                        'F4',
                                                        'C4',
                                                        'P4',
                                                        'Fp2',
                                                        'F8',
                                                        'T4',
                                                        'T6',
                                                        'Cz'],
                                                 cathode=['F3',
                                                          'C3',
                                                          'P3',
                                                          'O1',
                                                          'F7',
                                                          'T3',
                                                          'T5',
                                                          'O1',
                                                          'Cz',
                                                          'F4',
                                                          'C4',
                                                          'P4',
                                                          'O2',
                                                          'F8',
                                                          'T4',
                                                          'T6',
                                                          'O2',
                                                          'Pz'])

    raw_resam = raw_doublebanana.copy().resample(250)
    raw_banpas = raw_resam.copy().filter(0.1, 40, fir_design='firwin')

    return raw_banpas


# EEG Processing
def rescale_255(x):
    return(((x - x.min()) * (1/(x.max() - x.min()) * 255)).astype('int'))


def TF_analysis(raw_banpas, ID):
    epochs = mne.make_fixed_length_epochs(
        raw_banpas, duration=30, preload=True)
    freqs = np.arange(1., 41., 1.)
    n_cycles = freqs / 2.
    power = tfr_morlet(epochs, freqs=freqs, use_fft=True, decim=30,
                       n_cycles=n_cycles, return_itc=False, average=False)
    tf_data = power.data

    # string=ID[-22:-4]
    # string=string.replace("/",'_')
    for j, signal in enumerate(tf_data):
        tf_nor_1 = []
        for i in range(6):
            tf_nor_1.append(signal[i])
        tf_nor_1 = np.array(tf_nor_1)
        tf_nor_1 = np.reshape(tf_nor_1, (-1, 250))
        tf_nor_1 = np.log(tf_nor_1)

        tf_nor_2 = []
        for i in range(6):
            tf_nor_2.append(signal[i+6])
        tf_nor_2 = np.array(tf_nor_2)
        tf_nor_2 = np.reshape(tf_nor_2, (-1, 250))
        tf_nor_2 = np.log(tf_nor_2)

        tf_nor_3 = []
        for i in range(6):
            tf_nor_3.append(signal[i+12])
        tf_nor_3 = np.array(tf_nor_3)
        tf_nor_3 = np.reshape(tf_nor_3, (-1, 250))
        tf_nor_3 = np.log(tf_nor_3)

        a = rescale_255(tf_nor_1)
        b = rescale_255(tf_nor_2)
        c = rescale_255(tf_nor_3)
        # c=np.ones([360,60],int)
        # c=c*255
        d = np.array([a, b, c])
        d = d.transpose(1, 2, 0)
        n_im = Image.fromarray(d.astype(np.uint8))
        # print('ID', type(ID))
        # print('j', type(j))
        n_im.save(f'{TF_save_address}/%d_%d.png' % (int(ID), (j)))


# files為整個資料夾，需要設置路徑
print('sdfsdfsdf', os.getcwd())
mypath = os.getcwd() + '\\t0.eeg'
print('mypath:', mypath)
print(type(mypath))
data, Patient_ID = RDA2MNE(mypath)
TF_save_address = 'TF_Figure/'+str(Patient_ID)+'/'
os.makedirs(TF_save_address, exist_ok=True)
raw_banpas = EEG_PreProcessing(data)
TF_analysis(raw_banpas, Patient_ID)


# ----------Data Preparation--------------
BATCH_SIZE = 64
IMG_SIZE = 224
CLASS_NUM = 1
LR = 1e-3
N_EPOCHS = 15
input_shape = (224, 224, 3)
model_save_address = 'Best_model/'
os.makedirs(model_save_address, exist_ok=True)


All_file_paths = glob(f'{TF_save_address}/*.png')
print(len(All_file_paths))


def build_decoder(with_labels=True, target_size=(IMG_SIZE, IMG_SIZE), ext='png'):
    def decode(path):
        file_bytes = tf.io.read_file(path)
        if ext == 'png':
            img = tf.image.decode_png(file_bytes, channels=3)
        elif ext in ['jpg', 'jpeg']:
            img = tf.image.decode_jpeg(file_bytes, channels=3)
        else:
            raise ValueError("Image extension not supported")
        img = tf.cast(img, tf.float32) / 255.0
        img = tf.image.resize(img, target_size)
        return img

    def decode_with_labels(path, label):
        return decode(path), label
    return decode_with_labels if with_labels else decode


def build_augmenter(with_labels=True):
    def augment(img):
        img = tf.image.random_brightness(img, max_delta=0.6)
        img = tf.image.random_contrast(img, lower=0.8, upper=3)
        img = tf.image.random_hue(img, max_delta=0.5)
        img = tf.image.random_saturation(img, lower=0, upper=5)
        #img = tf.image.random_flip_left_right(img)
        #img = tf.image.random_flip_up_down(img)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        return img

    def augment_with_labels(img, label):
        return augment(img), label

    return augment_with_labels if with_labels else augment


def build_dataset(paths, labels=None, bsize=128, cache=True,
                  decode_fn=None, augment_fn=None,
                  repeat=True, shuffle=54,
                  cache_dir=""):
    if cache_dir != "" and cache is True:
        os.makedirs(cache_dir, exist_ok=True)
    augment = True if augment_fn is not None else False
    AUTO = tf.data.experimental.AUTOTUNE
    slices = paths if labels is None else (paths, labels)
    dset = tf.data.Dataset.from_tensor_slices(slices)
    dset = dset.map(decode_fn, num_parallel_calls=AUTO)
    dset = dset.cache(cache_dir) if cache else dset
    dset = dset.map(augment_fn, num_parallel_calls=AUTO) if augment else dset
    dset = dset.repeat() if repeat else dset
    dset = dset.shuffle(shuffle) if shuffle else dset
    dset = dset.batch(bsize).prefetch(AUTO)
    return dset


Test_dataset = build_dataset(paths=All_file_paths, labels=None, bsize=BATCH_SIZE,
                             decode_fn=build_decoder(with_labels=False, target_size=(
                                 IMG_SIZE, IMG_SIZE), ext='png'),
                             repeat=False, shuffle=False
                             )


def build_model(IMG_SIZE=IMG_SIZE, CLASS_NUM=CLASS_NUM, weight_path=None):
    if CLASS_NUM == 1:
        activation = 'sigmoid'
    else:
        activation = 'softmax'
    model = tf.keras.Sequential([
        efn.EfficientNetB0(
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            weights='noisy-student',
            # weights='imagenet',
            include_top=False),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(CLASS_NUM, activation=activation)
    ])
    if weight_path is not None:
        model.load_weights(weight_path)
    return model


print('當前資料夾路徑', os.getcwd())
curr_file_path = os.getcwd()
folder_path = curr_file_path + ''

model = build_model(IMG_SIZE=IMG_SIZE, CLASS_NUM=CLASS_NUM,
                    weight_path='Best_model/best_loss_model_20211208.h5')

pred_result = model.predict(Test_dataset, verbose=1)

y_pred = (pred_result > 0.5).astype("int32")


def return_pred_result():
    Pred_result = []
    for tmp in y_pred:
        if tmp == 0:
            Pred_result.append('Normal')
        elif tmp == 1:
            Pred_result.append('Seizure')
    return Pred_result
