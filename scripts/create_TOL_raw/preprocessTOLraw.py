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
# Processing Variables
# =============================================================================

# Root where the raw data are stored
root_dir = r'/home/w314/w314139/PROJECT/silent-speech-decoding/data/raw/TOL/'

# Root where the structured data will be saved
# It can be changed and saved in other direction
save_dir = root_dir + "derivatives/"

# Subjects and blocks
N_Subj_arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
N_block_arr = [1, 2, 3]

# -------------------- Filtering (disabled) -----------------------------------
# Cut‑off frequencies (kept for reference but **not used**)
Low_cut = 0.5  # Hz
High_cut = 100 # Hz

# Notch filter in 50 Hz
Notch_bool = False  # MODIFICACION: desactivar notch

# NEW: flag to apply band‑pass filtering
Filter_bool = False  # MODIFICACION: desactivar filtrado pasa‑banda

# Downsampling rate (still useful para reducir tamaño de epocas)
DS_rate = 4

# -------------------- ICA (disabled) -----------------------------------------
ICA_bool = False        # MODIFICACION: desactivar ICA
ICA_Components = None
ica_random_state = 23
ica_method = 'infomax'
max_pca_components = None
fit_params = dict(extended=True)

# -------------------- EMG Control (se mantiene) ------------------------------
low_f = 1
high_f = 20
# Slide window design
window_len = 0.5  # sec
window_step = 0.05  # sec

# Threshold for EMG control
std_times = 3

# Baseline
t_min_baseline = 0
t_max_baseline = 15

# Trial time
t_min = 1
t_max = 3.5

# =============================================================================
# Fixed Variables
# =============================================================================
# Events ID
# 31 = Arriba / Up
# 32 = Abajo / Down
# 33 = Derecha / Right
# 34 = Izquierda / Left
event_id = dict(Arriba=31, Abajo=32, Derecha=33, Izquierda=34)

# Baseline id
baseline_id = dict(Baseline=13)

# Report initialization
report = dict(Age=0, Gender=0, Recording_time=0, Ans_R=0, Ans_W=0)

# Montage
aquisition_eq = "biosemi128"
montage = mne.channels.make_standard_montage(aquisition_eq)

# External channels
Ref_channels = ['EXG1', 'EXG2']

# Gaze detection
Gaze_channels = ['EXG3', 'EXG4']

# Blinks detection
Blinks_channels = ['EXG5', 'EXG6']

# Mouth Moving detection
Mouth_channels = ['EXG7', 'EXG8']

# Demographic information
Subject_age = [56, 50, 34, 24, 31, 29, 26, 28, 35, 31]
Subject_gender = ['F', 'M', 'M', 'F', 'F', 'M', 'M', 'F', 'M', 'M']

# =============================================================================
# Processing loop
# =============================================================================
for N_S in N_Subj_arr:
    # Get Age and Gender
    report['Age'] = Subject_age[N_S - 1]
    report['Gender'] = Subject_gender[N_S - 1]

    for N_B in N_block_arr:
        print('Subject: ' + str(N_S))
        print('Session: ' + str(N_B))

        # ---------------- Load data from BDF file --------------------------
        rawdata, Num_s = extract_subject_from_bdf(root_dir, N_S, N_B)

        # Referencing
        rawdata.set_eeg_reference(ref_channels=Ref_channels)

        # ----------------------- OPTIONAL Filtering ------------------------
        # MODIFICACION: los filtros se aplican solo si las banderas están
        # activadas. Por defecto, ambos están desactivados para dejar la señal
        # lo más cruda posible.
        if Notch_bool:
            rawdata = mne.io.Raw.notch_filter(rawdata, freqs=50)

        if Filter_bool:
            rawdata.filter(Low_cut, High_cut)

        # ----------------------- Events handling ---------------------------
        # Subject 10 Block 1 has a spurious trigger
        if (N_S == 10 and N_B == 1):
            events = mne.find_events(rawdata, initial_event=True,
                                     consecutive=True, min_duration=0.002)
        else:
            events = mne.find_events(rawdata, initial_event=True,
                                     consecutive=True)

        events = check_baseline_tags(events)
        events = Event_correction(events=events)
        rawdata.event = events  # replace raw events with corrected events

        report['Recording_time'] = int(np.round(rawdata.last_samp / rawdata.info['sfreq']))

        # Cognitive Control
        report['Ans_R'], report['Ans_W'] = cognitive_control_check(events)

        # ---------------------- Save report ------------------------------
        file_path = save_dir + Num_s + '/ses-0' + str(N_B)
        ensure_dir(file_path)
        file_name = f"{file_path}/{Num_s}_ses-0{N_B}_report.pkl"
        with open(file_name, 'wb') as output:
            pickle.dump(report, output, pickle.HIGHEST_PROTOCOL)

        # ---------------------- EXG channels ------------------------------
        picks_eog = mne.pick_types(rawdata.info, eeg=False, stim=False,
                                   include=['EXG1', 'EXG2', 'EXG3', 'EXG4',
                                            'EXG5', 'EXG6', 'EXG7', 'EXG8'])
        epochsEOG = mne.Epochs(rawdata, events, event_id=event_id, tmin=-0.5,
                               tmax=4, picks=picks_eog, preload=True,
                               detrend=0, decim=DS_rate)

        file_name = f"{file_path}/{Num_s}_ses-0{N_B}_exg-epo.fif"
        epochsEOG.save(file_name, fmt='double', split_size='2GB', overwrite=True)
        del epochsEOG

        # ---------------------- Baseline ---------------------------------
        t_baseline = (events[events[:, 2] == 14, 0] - events[events[:, 2] == 13, 0]) / rawdata.info['sfreq']
        t_baseline = t_baseline[0]
        Baseline = mne.Epochs(rawdata, events, event_id=baseline_id, tmin=0,
                              tmax=round(t_baseline), picks='all',
                              preload=True, detrend=0, decim=DS_rate,
                              baseline=None)

        file_name = f"{file_path}/{Num_s}_ses-0{N_B}_baseline-epo.fif"
        Baseline.save(file_name, fmt='double', split_size='2GB', overwrite=True)
        del Baseline

        # ---------------------- EEG Epoching -----------------------------
        picks_eeg = mne.pick_types(rawdata.info, eeg=True,
                                   exclude=['EXG1', 'EXG2', 'EXG3', 'EXG4',
                                            'EXG5', 'EXG6', 'EXG7', 'EXG8'],
                                   stim=False)
        epochsEEG = mne.Epochs(rawdata, events, event_id=event_id, tmin=-0.5,
                               tmax=4, picks=picks_eeg, preload=True,
                               detrend=0, decim=DS_rate, baseline=None)

        # ---------------------- ICA (disabled) ---------------------------
        if ICA_bool:
            # El bloque completo de ICA se mantiene por referencia, pero no se 
            # ejecutará ya que ICA_bool=False. Si se quisiera reactivar,
            # bastaría con cambiar la bandera.
            picks_vir = mne.pick_types(rawdata.info, eeg=True,
                                       include=['EXG1', 'EXG2', 'EXG3', 'EXG4',
                                                'EXG5', 'EXG6', 'EXG7', 'EXG8'],
                                       stim=False)
            epochsEEG_full = mne.Epochs(rawdata, events, event_id=event_id,
                                        tmin=-0.5, tmax=4,
                                        picks=picks_vir, preload=True,
                                        detrend=0, decim=DS_rate,
                                        baseline=None)
            del rawdata  # liberar memoria

            ica = mne.preprocessing.ICA(n_components=ICA_Components,
                                        random_state=ica_random_state,
                                        method=ica_method,
                                        fit_params=fit_params)
            ica.fit(epochsEEG)
            ica.exclude = []
            # ... detección y exclusión de componentes ...
            print("Applying ICA")
            ica.apply(epochsEEG)

        # ---------------------- Save EEG ---------------------------------
        file_name = f"{file_path}/{Num_s}_ses-0{N_B}_eeg-epo.fif"
        epochsEEG.save(file_name, fmt='double', split_size='2GB', overwrite=True)

        # ---------------------- Events save ------------------------------

        # justo antes de add_condition_tag…
        print(">>>>IDs únicos de evento y conteos:", np.unique(events[:,2], return_counts=True))
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


