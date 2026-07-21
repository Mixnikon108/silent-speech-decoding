# -*- coding: utf-8 -*-

"""
Preprocesado del dataset TOL (Inner Speech) hacia epochs "raw".

Para cada sujeto y bloque: carga el BDF, corrige y valida los eventos, calcula
un residual ultra-low (señal original menos su versión filtrada en paso alto a
0.5 Hz) sobre los canales de datos, extrae epochs de EEG y guarda tanto los
epochs (.fif) como los eventos etiquetados (.dat). Aplica además la corrección
ad hoc del sujeto 3.
"""

# Imports modules
import mne
import pickle
import numpy as np

from Events_analysis import Event_correction, add_condition_tag, add_block_tag
from Events_analysis import check_baseline_tags, delete_trigger
from Events_analysis import cognitive_control_check, standardize_labels
from Data_extractions import extract_subject_from_bdf
from Utilitys import ensure_dir
from AdHoc_modification import adhoc_subject_3


# =============================================================================
# Processing Variables
# =============================================================================

root_dir = r'/path/to/project/data/raw/TOL/'
save_dir = root_dir + "derivatives/"

N_Subj_arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
N_block_arr = [1, 2, 3]

Low_cut = 0.5
High_cut = 100

Notch_bool = False
Filter_bool = False
DS_rate = 4

ICA_bool = False
ICA_Components = None
ica_random_state = 23
ica_method = 'infomax'
max_pca_components = None
fit_params = dict(extended=True)

low_f = 1
high_f = 20
window_len = 0.5
window_step = 0.05
std_times = 3

t_min_baseline = 0
t_max_baseline = 15

t_min = 1
t_max = 3.5

event_id = dict(Arriba=31, Abajo=32, Derecha=33, Izquierda=34)
baseline_id = dict(Baseline=13)

report = dict(Age=0, Gender=0, Recording_time=0, Ans_R=0, Ans_W=0)

aquisition_eq = "biosemi128"
montage = mne.channels.make_standard_montage(aquisition_eq)

Ref_channels = ['EXG1', 'EXG2']
Gaze_channels = ['EXG3', 'EXG4']
Blinks_channels = ['EXG5', 'EXG6']
Mouth_channels = ['EXG7', 'EXG8']

Subject_age = [56, 50, 34, 24, 31, 29, 26, 28, 35, 31]
Subject_gender = ['F', 'M', 'M', 'F', 'F', 'M', 'M', 'F', 'M', 'M']

# =============================================================================
# Processing loop
# =============================================================================
for N_S in N_Subj_arr:
    report['Age'] = Subject_age[N_S - 1]
    report['Gender'] = Subject_gender[N_S - 1]

    for N_B in N_block_arr:
        print('Subject: ' + str(N_S))
        print('Session: ' + str(N_B))

        # ---- Load data ----
        rawdata, Num_s = extract_subject_from_bdf(root_dir, N_S, N_B)

        rawdata.set_eeg_reference(ref_channels=Ref_channels)

        if Notch_bool:
            rawdata = mne.io.Raw.notch_filter(rawdata, freqs=50)
        if Filter_bool:
            rawdata.filter(Low_cut, High_cut)

        # ---- Events ----
        if (N_S == 10 and N_B == 1):
            events = mne.find_events(rawdata, initial_event=True,
                                     consecutive=True, min_duration=0.002)
        else:
            events = mne.find_events(rawdata, initial_event=True,
                                     consecutive=True)

        events = check_baseline_tags(events)
        events = Event_correction(events=events)
        rawdata.event = events

        report['Recording_time'] = int(np.round(rawdata.last_samp / rawdata.info['sfreq']))
        report['Ans_R'], report['Ans_W'] = cognitive_control_check(events)

        file_path = save_dir + Num_s + '/ses-0' + str(N_B)
        ensure_dir(file_path)
        file_name = f"{file_path}/{Num_s}_ses-0{N_B}_report.pkl"
        with open(file_name, 'wb') as output:
            pickle.dump(report, output, pickle.HIGHEST_PROTOCOL)

        # ---------------------- SOLO RESIDUAL ULTRA-LOW ----------------------
        # Cargar en memoria y calcular residual solo en canales de datos
        rawdata.load_data()
        picks_data = mne.pick_types(rawdata.info, eeg=True, eog=True, ecg=True,
                                    emg=True, meg=False, misc=True, stim=False)
        picks_stim = mne.pick_types(rawdata.info, stim=True)

        raw_hp = rawdata.copy()
        raw_hp.filter(l_freq=0.5, h_freq=None, picks=picks_data, verbose=False)

        data_orig = rawdata.get_data()
        data_hp = raw_hp.get_data()
        data_res = data_orig.copy()
        data_res[picks_data, :] = data_orig[picks_data, :] - data_hp[picks_data, :]
        if picks_stim.size > 0:
            data_res[picks_stim, :] = data_orig[picks_stim, :]

        raw_residual = rawdata.copy()
        raw_residual._data = data_res

        # Epoching con el residual
        picks_eeg = mne.pick_types(raw_residual.info, eeg=True,
                                   exclude=['EXG1', 'EXG2', 'EXG3', 'EXG4',
                                            'EXG5', 'EXG6', 'EXG7', 'EXG8'],
                                   stim=False)
        epochs_residual = mne.Epochs(raw_residual, events, event_id=event_id,
                                     tmin=-0.5, tmax=4, picks=picks_eeg,
                                     preload=True, detrend=0,
                                     decim=DS_rate, baseline=None)

        # Guardar solo el residual
        file_name_res = f"{file_path}/{Num_s}_ses-0{N_B}_eeg-epo.fif"
        epochs_residual.save(file_name_res, fmt='double', split_size='2GB', overwrite=True)
        # ----------------------------------------------------------------------

        # ---- Events save ----
        print(">>>>IDs únicos de evento y conteos:", np.unique(events[:, 2], return_counts=True))
        print(">>>>Total events rows:", events.shape[0])
        events = add_condition_tag(events)
        print(">>>>Total tags generated:", events.shape[0])
        events = add_block_tag(events, N_B=N_B)
        events = delete_trigger(events)
        events = standardize_labels(events)

        file_name = f"{file_path}/{Num_s}_ses-0{N_B}_events.dat"
        events.dump(file_name)

# ========================= Ad Hoc & EMG Control ==============================
adhoc_subject_3(root_dir=root_dir)
