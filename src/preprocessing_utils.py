"""
Module Name: preprocessing_utils
Description: This module contains all data preprocessing functions.

Classes & Methods:
- FeatureExtractor: it extracts handcrafted, oS features and wav2vec embeddings 
- SyllableNucleiExtractor: extracts handcrafted features
- Wav2VecEmbeddingsExtractor: extracts wav2vec embeddings
- TextEmbeddingsExtractor: extract bert embeddings

#! to-do: typing
#! to-do: documentation
"""

import os
import re
import sys
import math

from tqdm import tqdm
import pandas as pd
import numpy as np

from audiomentations import (
    Compose,
    AddGaussianNoise,
    PitchShift,
    TimeStretch,
    Shift,
)
import audiofile
from opensmile import FeatureLevel, FeatureSet, Smile
from sklearn.decomposition import PCA
import parselmouth
from parselmouth.praat import call
import pickle
import torch
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
from transformers import BertTokenizer, BertModel
import whisper


def label_encoder():
    label2id = {"control": 0, "MCI": 1, "dementia": 2}
    id2label = {id: label for label, id in label2id.items()}
    return label2id, id2label


class FeatureExtractor:
    def __init__(self, device):
        self.device = device

    def load_and_extract_features(
        self,
        db: pd.DataFrame,
        features_type: str,
        target: str,
        output: str,
        number_variance: float,
        number_components: int,
        pca: str,
    ):
        """
        Description: extracts corresponding features and loads them if they already exist.
        Args:
            db(pd.DataFrame): data for processing
            features_type(str): name of the feature set to extract
            target(str): name of the target column
            output(str): path where to save all the features
            number_variance(float):number of variance to keep for pca
            pca(str): whether to apply pca or not
        Returns:
            df_all
            feat_columns
        """

        if pca == "apply_pca":
            files4features = f"{output}/features_{features_type}_pca_{number_variance}_{number_components}.pkl"
            file_feature_names = f"{output}/list_{features_type}_pca_{number_variance}_{number_components}.pkl"
        else:
            files4features = f"{output}/features_{features_type}.pkl"
            file_feature_names = f"{output}/list_{features_type}.pkl"

        db.loc[:, "label"] = db[target]

        df_feat, feat_columns = None, None
        if re.search("wav2vec", features_type) and not os.path.exists(files4features):
            frame, feat_columns, updated_num_components = self.extract_acoustic_embeddings(
                db, features_type, number_variance, number_components, pca
            )
            if updated_num_components != number_components and pca == "apply_pca":
                files4features = f"{output}/features_{features_type}_pca_{number_variance}_{updated_num_components}.pkl"
                file_feature_names = f"{output}/list_{features_type}_pca_{number_variance}_{updated_num_components}.pkl"
            else:
                pass
            df_all = self.save_processed_data(
                frame, feat_columns, files4features, file_feature_names
            )
        elif features_type == "rhythmic":
            df_feat, feat_columns = self.extract_handcrafted(db, features_type, output)
        elif features_type == "os":
            df_feat, feat_columns = self.extract_os(db, output)
        elif features_type == "combined_handcrafted":
            df_feat, feat_columns = self.combined_handcrafted(db, output)

        if not re.search("wav2vec", features_type) and not os.path.exists(
            files4features
        ):
            if "start" in df_feat.columns:
                frame = db.merge(df_feat, on=["file", "start", "end"])
            else:
                frame = db.merge(df_feat, on=["file"])

            if pca == "apply_pca":
                frame, feat_columns, updated_num_components = self.apply_pca(
                    frame,
                    feat_columns,
                    n_variance=number_variance,
                    n_components=number_components,
                )
                if updated_num_components != number_components:
                    files4features = f"{output}/features_{features_type}_pca_{number_variance}_{updated_num_components}.pkl"
                    file_feature_names = f"{output}/list_{features_type}_pca_{number_variance}_{updated_num_components}.pkl"
                print("PCA is applied")
            else:
                print("PCA is NOT applied")

            self.check_file_percentage(frame, db, df_feat)
            df_all = self.save_processed_data(
                frame, feat_columns, files4features, file_feature_names
            )
        else:
            print(f"Data is already processed and save in {files4features}")

        return df_all, feat_columns

    @staticmethod
    def check_file_percentage(frame, db, df_feat):
        initial_file_count = len(db)
        feature_file_count = len(df_feat)
        merged_file_count = len(frame)
        percentage_files_left = (merged_file_count / initial_file_count) * 100
        if percentage_files_left < 99:
            print(f"Initial files in metadata: {initial_file_count}")
            print(f"Initial files in feature daframe: {feature_file_count}")
            print(f"Files left after merging: {merged_file_count}")
            print(f"Percentage of files left: {percentage_files_left:.2f}%")
            sys.exit(f"Many files are being dropped. Investigation is needed.")

    def extract_acoustic_embeddings(
        self, db, model_wav2vec, number_variance, number_components, pca
    ):
        """
        Description: extract wav2vec embeddings from either wav2vec multilingual or base.
        Args:
            db(pd.DataFrame):dataframe from where the features are extracted. It should contain a column called "file"
            model_wav2vec: either "wav2vec_base" or "wav2vec_multilingual"
            number_variance(float):number of variance to apply for pca
            pca(boolean): whether to apply pca or not
        Returns:
            df_feat(pd.DataFrame): dataframe containing "file", and feature columns
        """
        if model_wav2vec == "wav2vec_multilingual":
            model_path = "facebook/wav2vec2-large-xlsr-53"
        elif model_wav2vec == "wav2vec_base":
            model_path = "facebook/wav2vec2-base-960h"

        else:
            sys.exit(
                f"{model_wav2vec} is not supported. Please choose between `wav2vec_multilingual` or `wav2vec_base`."
            )

        # feature extraction is at feature level
        # db = self.dt2sec(db)
        wav2vec_processor = Wav2VecEmbeddingsExtractor(model_path, self.device)
        df_feat, feat_columns = wav2vec_processor.process_audio(db)

        if pca == "apply_pca":
            df_feat, feat_columns, number_components = self.apply_pca(
                df_feat, feat_columns, number_variance, number_components
            )
            print("PCA is applied")
        else:
            number_components = None
            print("PCA is NOT applied")
        return df_feat, feat_columns, number_components

    def combined_handcrafted(self, db, output_dir) -> pd.DataFrame:
        """
        Description: function to combine openSMILE and rythmic features
        Args:
            db(pd.DataFrame):data for feature extraction
            output_dir(str): where to store the features
        Returns:
            df_feat(pd.DataFrame):data with features combined
        """

        feat_list = list()
        df_rhythimc, feat_rhythmic = self.extract_handcrafted(
            db, "rhythmic", output_dir
        )
        df_os, feat_os = self.extract_os(db, output_dir)

        if "start" in df_rhythimc.columns:
            df_feat = df_os.merge(df_rhythimc, on=["file", "start", "end"])
        else:
            df_feat = df_os.merge(df_rhythimc, on=["file"])

        feat_list.extend(feat_os)
        feat_list.extend(feat_rhythmic)
        print(len(feat_list))

        return df_feat, feat_list

    def apply_pca(
        self, db, feat_col, n_variance=None, n_components=None, random_state=42
    ):
        """
        Description: applies Principal Component Analysis (PCA) to the features.
        Args:
            db(pd.DataFrame): input data for PCA
            n_variance(float): variance to keep. If None, all components are kepts.
            random_state(int)
        Returns:
            pca_df(pd.DataFrame): the transformed data"""

        if "file" not in db.columns:
            df.reset_index(inplace=True)

        db_features = db[feat_col]
        file_column = db["file"]
        df_metadata = db[
            ["label", "ID", "file", "dataset", "age", "gender", "mmse", "split"]
        ]

        pca = PCA()
        transformed_data = pca.fit_transform(db_features)

        if n_variance is not None:
            cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
            n_components = np.argmax(cumulative_variance >= n_variance) + 1
            print(
                f"Final number of components to explain {n_variance * 100:.2f}% variance: {n_components}"
            )
            pca = PCA(n_components=n_components, random_state=random_state)

            transformed_data = pca.fit_transform(db_features)

        elif n_components is not None:
            print(f"Final number of components specified by user: {n_components}")
            pca = PCA(n_components=n_components, random_state=random_state)
            transformed_data = pca.fit_transform(db_features)
            print("Explained variance ratio for each component:")
            total_explained_variance = 0.0
            for i, explained_variance in enumerate(pca.explained_variance_ratio_):
                print(f"Component {i+1}: {explained_variance:.4f}")
                total_explained_variance+=explained_variance
            print (f"The total explained variance with this number of components is: {total_explained_variance}")

        columns = [f"PC{i+1}" for i in range(transformed_data.shape[1])]
        pca_df = pd.DataFrame(data=transformed_data, columns=columns)
        pca_df["file"] = file_column.reset_index(drop=True)

        # merge the components with the metadata
        df = pca_df.merge(df_metadata, on="file")

        return df, columns, n_components

    def extract_os(self, db, output_dir):
        """
        Description: extract eGeMAPS features from the openSMILE feature extractor.
        Args:
            db(pd.DataFrame):dataframe from where the features are extracted. It should contain a column called "file"
            output_dir(str): path to where the extracted features are stored
        Returns:
            df_feat(pd.DataFrame): dataframe containing "file","start", "end", and feature columns
            feat_columns(list): list of the feature names available in "df_feat"
        """
        feature_set = "eGeMAPSv02"

        if "start" not in db.index.names and "start" in db.columns:
            db.reset_index(inplace=True)
            db.set_index(["file", "start", "end"], inplace=True)
        elif "file" not in db.index.names:
            db.reset_index(inplace=True)
            db.set_index(["file"], inplace=True)

        smile = Smile(
            feature_set=getattr(FeatureSet, feature_set),
            feature_level=FeatureLevel.Functionals,
            num_workers=16,
            verbose=True,
            logfile="log",
        )

        path = os.path.join(output_dir, f"{feature_set}.pkl")

        if not os.path.exists(path):
            df_feat = smile.process_index(index=db.index)
            df_feat.to_pickle(path)
            print(f"Features are extracted and saved to:", path)
            assert len(db) == len(df_feat)
        else:
            df_feat = pd.read_pickle(path)

        feat_columns = df_feat.columns

        # sanity check how similar is the index of both dataframe
        if df_feat.index.equals(db.index):
            print("Both dataframes have the same index.")
        else:
            print("Dataframes have different indices.")

        df_feat.reset_index(["file"], inplace=True)

        return df_feat, feat_columns

    def extract_handcrafted(self, db, feature_set, output_dir):
        """
        Description: extract handcrafted features.
        Args:
            db(pd.DataFrame):dataframe from where the features are extracted. It should contain a column called "file"
            output_dir(str): path in which the extracted features are stored
        Returns:
            df_feat(pd.DataFrame): dataframe containing "file" and feature columns
            feat_columns(list): list of the feature names available in "df_feat"
        """
        if "file" in db.index.names:
            db.reset_index(inplace=True)

        path = os.path.join(output_dir, f"{feature_set}.pkl")
        if not os.path.exists(path):
            df_list = []
            nuclei_extractor = SyllableNucleiExtractor(
                silencedb=-25, mindip=2, minpause=0.3
            )
            # feature extraction is performed at file-level
            for sound in tqdm(db["file"]):
                s = parselmouth.Sound(sound)
                speech_rate_dict = nuclei_extractor.syllable_nuclei(s)
                speech_rate_dict["file"] = sound
                df_list.append(speech_rate_dict)

            df_feat = pd.DataFrame(df_list)
            # assert whether the original dataframe and the feature dataframe have the same length
            assert len(db) == len(df_feat)

            df_feat.set_index(["file"], inplace=True)
            df_feat.to_pickle(path)
            print(f"Features are extracted and saved to:", path)
        else:
            df_feat = pd.read_pickle(path)

        feat_columns = df_feat.columns

        return df_feat, feat_columns

    @staticmethod
    def dt2sec(df: pd.DataFrame, reset_index=True) -> pd.DataFrame:
        """
        Description: converts start and end index values from datetime to seconds.
        Args:
            df(pd.DataFrame): file, start, and end should be in the dataframe
            reset_index (boolean): if True, "file, start, end" will be
                columns. If False, they will be index
        Returns:
            df(pd.DataFrame)
        """
        print(df)
        if "start" not in df.columns:
            df.reset_index(inplace=True)

        # print(df['start'].dtype)
        df.loc[:, "start"] = df["start"].dt.total_seconds()
        df.loc[:, "end"] = df["end"].dt.total_seconds()

        if not reset_index:
            df.set_index(["file", "start", "end"], inplace=True)

        return df

    @staticmethod
    def save_processed_data(
        data: pd.DataFrame, feat_columns: list, file_name: str, lst_feature_file: str
    ) -> pd.DataFrame:
        """
        Description: function to save the processed data into a dict. It preserves the tensors
        Args:
            data (pd.DataFrame): data with the processed input
            feature_columns(list): list of the name of the feature columns
            file_name(str): absolute filename where dictionary will be stored
            lst_feature_file(str): name of file where to store the list
        Returns:
            df_all(pd.DataFrame)
        """

        # * extended the list of columns that are saved with the features to make analysis of results per subgroup easier
        columns_to_save = [
            "label",
            "ID",
            "file",
            "dataset",
            "age",
            "gender",
            "mmse",
            "split",
        ]
        columns_to_save.extend(feat_columns)

        df_all = data[columns_to_save]
        df_all.to_pickle(file_name)

        with open(lst_feature_file, "wb") as file:
            pickle.dump(feat_columns, file)

        print(f"Data has been processed and saved in {file_name}")

        return df_all


class SyllableNucleiExtractor:
    def __init__(self, silencedb, mindip, minpause):
        self.silencedb = silencedb
        self.mindip = mindip
        self.minpause = minpause

    def calculate_intensity_threshold(self, intensity, max_intensity, min_intensity):
        max_99_intensity = np.percentile(intensity, 99)
        threshold = max_99_intensity + self.silencedb
        threshold2 = max_intensity - max_99_intensity
        threshold3 = self.silencedb - threshold2
        return max(threshold, min_intensity), threshold3

    def get_silence_and_speech_durations(
        self, silencetable_sounds, silencetable_sil, n_speech_segs, n_sil_segs
    ):
        # speakingtot = sum(
        #    call(silencetable_sounds, "Get value", i, 2)
        #    - call(silencetable_sounds, "Get value", i, 1)
        #    for i in range(1, len(silencetable_sounds) + 1)
        # )
        #
        # sil_pauses_dur = [
        #    call(silencetable_sil, "Get value", i, 2)
        #    - call(silencetable_sil, "Get value", i, 1)
        #    for i in range(1, len(silencetable_sil) + 1)
        # ]
        # silenttot = sum(sil_pauses_dur)

        speakingtot = 0
        speech_segs_dur = []
        for ispeech in range(1, n_speech_segs + 1):
            beginsound = call(silencetable_sounds, "Get value", ispeech, 1)
            endsound = call(silencetable_sounds, "Get value", ispeech, 2)
            speakingdur = endsound - beginsound
            speakingtot += speakingdur
            speech_segs_dur.append(speakingdur)

        # pauses duration
        silenttot = 0
        sil_pauses_dur = []
        first_sil_dur = 0  # 0 in case there is no initial silence
        last_sil_dur = 0  # 0 in case there is no final silence
        for ipause in range(1, n_sil_segs + 1):
            beginsil = call(silencetable_sil, "Get value", ipause, 1)
            endsil = call(silencetable_sil, "Get value", ipause, 2)
            if beginsil == 0:  # excludes first silence if it occurs before speech
                first_sil_dur = endsil - beginsil
                continue
            if endsil > endsound:  # excludes last silence if it occurs after speech
                last_sil_dur = endsil - beginsil
                continue
            sildur = endsil - beginsil
            silenttot += sildur
            sil_pauses_dur.append(sildur)

        return (
            speakingtot,
            speech_segs_dur,
            sil_pauses_dur,
            silenttot,
            first_sil_dur,
            last_sil_dur,
        )

    def get_intensity_peaks(
        self, intensity_matrix, sound_from_intensity_matrix, threshold
    ):
        point_process = call(
            sound_from_intensity_matrix,
            "To PointProcess (extrema)",
            "Left",
            "yes",
            "no",
            "Sinc70",
        )
        numpeaks = call(point_process, "Get number of points")

        t = [call(point_process, "Get time from index", i + 1) for i in range(numpeaks)]
        intensities = [
            call(sound_from_intensity_matrix, "Get value at time", t[i], "Cubic")
            for i in range(numpeaks)
            if call(sound_from_intensity_matrix, "Get value at time", t[i], "Cubic")
            > threshold
        ]

        return t, intensities

    def calculate_valid_peaks(
        self, validtime, peakcount, timepeaks, intensity, currenttime, currentint
    ):
        for p in range(peakcount - 1):
            following = p + 1
            dip = call(intensity, "Get minimum", currenttime, timepeaks[p + 1], "None")
            diffint = abs(currentint - dip)
            if diffint > self.mindip:
                validtime.append(timepeaks[p])
        return validtime

    def calculate_voiced_parts(self, validtime, textgrid, pitch):
        voicedcount = 0
        voicedpeak = []

        for time in range(len(validtime)):
            querytime = validtime[time]
            whichinterval = call(textgrid, "Get interval at time", 1, querytime)
            whichlabel = call(textgrid, "Get label of interval", 1, whichinterval)
            value = pitch.get_value_at_time(querytime)
            if not math.isnan(value) and whichlabel == "sounding":
                voicedcount += 1
                voicedpeak.append(validtime[time])

        return voicedcount, voicedpeak

    def insert_voiced_peaks(self, textgrid, timecorrection, voicedpeak):
        call(textgrid, "Insert point tier", 1, "syllables")
        for i in range(len(voicedpeak)):
            position = voicedpeak[i] * timecorrection
            call(textgrid, "Insert point", 1, position, "")

    def calculate_silence_features(
        self,
        sil_pauses_dur,
        speech_segs_dur,
        dur_between_speech,
        originaldur,
        speakingtot,
    ):
        mean_pause_dur = np.mean(sil_pauses_dur) if len(sil_pauses_dur) else 0
        mean_speech_dur = np.mean(speech_segs_dur) if len(speech_segs_dur) else 0
        silence_rate = (
            0 if dur_between_speech == 0 else sum(sil_pauses_dur) / dur_between_speech
        )
        silence_speech_ratio = (
            len(sil_pauses_dur) / (len(speech_segs_dur) - 1)
            if len(speech_segs_dur) > 1
            else 0
        )
        mean_sil_count = (
            sum(sil_pauses_dur) / dur_between_speech if dur_between_speech > 0 else 0
        )

        longsils = [s for s in sil_pauses_dur if s > 1]
        lsil_rate = sum(longsils) / dur_between_speech if dur_between_speech > 0 else 0
        lsil_speech_ratio = (
            len(longsils) / (len(speech_segs_dur) - 1)
            if len(speech_segs_dur) > 1
            else 0
        )
        mean_lsil_count = (
            len(longsils) / dur_between_speech if dur_between_speech > 0 else 0
        )

        return (
            mean_pause_dur,
            mean_speech_dur,
            silence_rate,
            silence_speech_ratio,
            mean_sil_count,
            lsil_rate,
            lsil_speech_ratio,
            mean_lsil_count,
        )

    def syllable_nuclei(self, sound):
        originaldur = sound.get_total_duration()
        intensity = sound.to_intensity(50)
        min_intensity = call(intensity, "Get minimum", 0, 0, "Parabolic")
        max_intensity = call(intensity, "Get maximum", 0, 0, "Parabolic")

        threshold, threshold3 = self.calculate_intensity_threshold(
            intensity, max_intensity, min_intensity
        )

        textgrid = call(
            intensity,
            "To TextGrid (silences)",
            threshold3,
            self.minpause,
            0.1,
            "silent",
            "sounding",
        )
        silencetier = call(textgrid, "Extract tier", 1)
        silencetable_sounds = call(silencetier, "Down to TableOfReal", "sounding")
        n_speech_segs = call(silencetable_sounds, "Get number of rows")

        if n_speech_segs > 1:
            silencetable_sil = call(silencetier, "Down to TableOfReal", "silent")
            n_sil_segs = call(silencetable_sil, "Get number of rows")
        else:
            silencetable_sil = 0
            n_sil_segs = 0

        (
            speakingtot,
            speech_segs_dur,
            sil_pauses_dur,
            silenttot,
            first_sil_dur,
            last_sil_dur,
        ) = self.get_silence_and_speech_durations(
            silencetable_sounds, silencetable_sil, n_speech_segs, n_sil_segs
        )

        dur_between_speech = originaldur - first_sil_dur - last_sil_dur

        intensity_matrix = call(intensity, "Down to Matrix")
        sound_from_intensity_matrix = call(intensity_matrix, "To Sound (slice)", 1)
        intensity_duration = call(sound_from_intensity_matrix, "Get total duration")
        intensity_max = call(
            sound_from_intensity_matrix, "Get maximum", 0, 0, "Parabolic"
        )

        t, intensities = self.get_intensity_peaks(
            intensity_matrix, sound_from_intensity_matrix, threshold
        )

        validtime = self.calculate_valid_peaks(
            [], len(intensities), t, intensity, t[0], intensities[0]
        )

        pitch = sound.to_pitch_ac(0.02, 30, 4, False, 0.03, 0.25, 0.01, 0.35, 0.25, 450)
        voicedcount, voicedpeak = self.calculate_voiced_parts(
            validtime, textgrid, pitch
        )

        timecorrection = originaldur / intensity_duration
        self.insert_voiced_peaks(textgrid, timecorrection, voicedpeak)

        speakingrate = voicedcount / dur_between_speech
        articulationrate = voicedcount / speakingtot

        (
            mean_pause_dur,
            mean_speech_dur,
            silence_rate,
            silence_speech_ratio,
            mean_sil_count,
            lsil_rate,
            lsil_speech_ratio,
            mean_lsil_count,
        ) = self.calculate_silence_features(
            sil_pauses_dur,
            speech_segs_dur,
            dur_between_speech,
            originaldur,
            speakingtot,
        )

        asd = speakingtot / voicedcount if voicedcount > 0 else None

        speechrate_dictionary = {
            "number_syllable_nuclei": voicedcount,
            "total_pauses": n_speech_segs - 1,
            "duration": originaldur,
            "phonationtime": speakingtot,
            "silence": silenttot,
            "speaking_rate": speakingrate,
            "articulation_rate": articulationrate,
            "average_syllable": asd,
            "mean_pauses": mean_pause_dur,
            "mean_duration": mean_speech_dur,
            "silence_rate": silence_rate,
            "silence_speech_ratio": silence_speech_ratio,
            "mean_syllables": mean_sil_count,
            "long_syllables_rate": lsil_rate,
            "long_syllables_speech_ratio": lsil_speech_ratio,
            "mean_long_syllables": mean_lsil_count,
        }

        return speechrate_dictionary


class Wav2VecEmbeddingsExtractor:
    """
    Description:facilitates the extraction of Wav2Vec feature embeddings from audio files.
    it also applies data augmentation techniques.

    Attributes:
    - processor: Wav2Vec2FeatureExtractor instance for feature extraction
    - model_wav: Wav2Vec2Model instance for Wav2Vec model
    - target_sampling_rate: target sampling rate for audio processing
    """

    def __init__(self, model_path, device):
        """
        Description: initialization for extracting Wav2vec feature embeddings
        Args:
            model_path(str): the path to the pre-trained wav2vec model
            device(str): the device on which to load the mode (e.g. "cuda" or "cpu")
        """
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
        self.model_wav = Wav2Vec2Model.from_pretrained(model_path).to(device)
        self.target_sampling_rate = self.processor.sampling_rate
        self.device = device

    def process_audio(self, examples):
        """
        Description: proccess a dataframe containing audio files paths, start and end. it applies data augmentation
        Args:
            examples(pd.DataFrame): dataframe that contains a column "file", "start" and "end"
        """
        input_column = "file"

        speech_list = [
            self.augmented_speech_file_to_array(path)
            for path in tqdm(
                examples[input_column],
                desc="Augmenting data",
            )
        ]

        features_list = [
            self.extract_wav2vec_features(audio).squeeze()
            for audio in tqdm(speech_list, desc="Extracting features")
        ]

        features_np = np.vstack([tensor.numpy() for tensor in features_list])
        artificial_feature_names = [f"e{i+1}" for i in range(features_np.shape[1])]

        features = pd.DataFrame(data=features_np, columns=artificial_feature_names)
        examples = pd.concat([examples, features], axis=1)

        return examples, artificial_feature_names

    def augmented_speech_file_to_array(self, path):
        """
        Description:  Loads and augments a portion of an audio file based on specified start and end times
        Args:
            path(str):absolute path for the audio file
        """
        augment = Compose(
            [
                AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
                TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
                PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
                Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
            ]
        )
        max_duration = 8.0
        speech_array, sampling_rate = audiofile.read(path, duration=max_duration)
        augmented_array = augment(samples=speech_array, sample_rate=sampling_rate)
        speech = augmented_array.squeeze()
        return speech

    def extract_wav2vec_features(self, audio):
        """
        Description: extracts Wav2Vec feature embeddings from the provided
        audio using the pre-trained model and feature extractor
        Args:
            audio: absolute path for audio file
        """
        inputs = self.processor(
            audio,
            sampling_rate=self.target_sampling_rate,
            return_tensors="pt",
            padding="longest",
        ).to(self.device)
        with torch.no_grad():
            feat = self.model_wav(inputs["input_values"])
            features = feat["last_hidden_state"].squeeze()
            features = features[0]
        return features.to("cpu")


class TextEmbeddingsExtractor:
    """
    Description:use the Whisper ASR model to transcribe audio files
    and then extracts BERT embeddings from the obtained transcriptions


    - extract_transcriptions_and_embeddings_per_file(audio_file_path): Transcribes the given
      audio file and extracts BERT embeddings for the obtained transcription.
    - extract_bert_embeddings(text): Tokenizes the provided text, obtains BERT embeddings using
      a pre-trained model, and returns the embeddings.

    Attributes:
    - asr_model: Whisper ASR model loaded for speech transcription.
    - bert_tokenizer: BERT tokenizer for text tokenization.
    - bert_model: Pre-trained BERT model for extracting embeddings.

    """

    def __init__(self, whisper_model, bert_model_name, device):
        """
        Description:Initialization
        Args:
            whisper_model (str): path or model name of the Whisper ASR model.
            bert_model_name (str): BERT model name or path.
            device (str): the device on which to load the models (e.g., "cuda" for GPU or "cpu").
        """
        self.asr_model = whisper.load_model(whisper_model, device)

        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.bert_model = BertModel.from_pretrained(bert_model_name)

    def extract_embeddings(self, df):
        """
        Description: process a dataframe that contains audio file paths, transcribes them with whisper and extracts BERT embeddings
        Args:
            df(pd.DataFrame): dataframe from which the audio files can be found"""
        if "file" not in df.columns:
            raise ValueError("DataFrame must have a 'file' column.")

        embeddings = []

        for audio_file_path in tqdm(
            enumerate(df["file"]), desc="Processing", total=len(df)
        ):
            bert_embeddings = self.extract_transcriptions_and_embeddings_per_file(
                audio_file_path
            )
            embeddings.append(bert_embeddings)

        df["bert_features"] = embeddings

        return df

    def extract_transcriptions_and_embeddings_per_file(self, audio_file_path):
        """
        Description:transcribes the given audio file and extracts BERT embeddings for the obtained transcription
        Args
            audio_file_path(str):absolute path to the audio file
        """
        transcription = self.asr_model.transcribe(audio_file_path)

        bert_embeddings = self.extract_bert_embeddings(transcription)

        return bert_embeddings

    def extract_bert_embeddings(text):
        """
        Description:extracts BERT embeddings
        Args:
            text(str):transcribed text to extract BERT embeddings
        """
        tokens = self.bert_tokenizer(
            text, return_tensors="pt", truncation=True, padding=True
        )

        with torch.no_grad():
            outputs = self.bert_model(**tokens)

        embeddings = outputs.last_hidden_state[:, 0, :].numpy()

        return embeddings
