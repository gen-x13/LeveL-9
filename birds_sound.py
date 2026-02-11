# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 08:21:42 2026

@author: GenXCode
"""

# Import handling files
import pandas as pd

# Import for calculations
import numpy as np

# Import plotting results
from matplotlib import pyplot as plt

# Imports turning sample to audio
from scipy.io.wavfile import write

# Imports handling audio
import librosa # is also a library with audio samples
import librosa.display

# Import handling the path to files
from pathlib import Path

# Import handling the audio files in the path
import os

# Import handling the model
from birds_sound_model import BirdModel

# Import grabbing the audios from Xeno-Canto API
import requests

# Import JSON
import json

# Import time to wait before making another request
import time

# ------------------------- GETTING THE BIRDS AUDIOS ------------------------ #

# Path to the audio files data
file_path = Path(__file__).parent / "json_xeno_canto"

# Path to the audios files for extraction loop
audios_path = Path(__file__).parent / "audios_xeno_canto"

# Path to the audio samples for prediction
sample_audio_path = Path(__file__).parent / "sample_audios_xeno_canto"

# Path to store the audio graph pictures
image_path = Path(__file__).parent / "pictures"

# Path to store the dataset from the birdsong samples infos
data_path = Path(__file__).parent / "data"

# The Class running the Birdsongs downloading, filtering, transformation,
# and slicing into smaller samples
class BirdSongs():
    
    def __init__(self):
        
        # Lists inside the dictionnary containing the files informations
        self.names = []
        self.species = []
        self.audio_sample = []
        self.amplitudes = []
        self.frequencies = []
        self.spectral_descs_mean = []
        self.spectral_descs_var = []
        self.spectral_centroids_max = []
        
        self.data = None

    def fetching_data(self):
        
        # Storing an increasing counter to meet the threshold and stop the loop    
        download = 0
        
        # Request and Dowloading Loops to fetch data and download the 50 audios
        
        # The URL = Any name, any species, in english, 
        # with good quality and the audio length is under 2 minutes
        base_url = 'https://www.xeno-canto.org/api/3/recordings'
        query = 'len:"<120"'
        url = f'{base_url}?query={query}&per_page=50&page=1&key=3277920db0ddaf85b5d307dd738c931776b387d3'
        
        # Sending a request to store the data inside the r variable
        r = requests.get(url)
        if r.status_code == 200:
            print("OK")
        else:
            print(r.json())
            print("Erreur HTTP :", r.status_code)
            
        # Getting the json from the response
        self.data = r.json()
        print(self.data.keys())
        
        pages = int(self.data.get("numPages", 1))
        
        # Storing the JSON inside the folder
        filename = file_path / 'page 1.json'
        with open(filename, 'w') as saved:
            json.dump(self.data, saved)
            

        # Rate limit of one request per second
        time.sleep(1)
        
        for page in range(2, pages + 1):
        
            # Request and Dowloading Loops to fetch data and download the 50 audios
            
            # The URL = Any name, any species, in english, 
            # with good quality and the audio length is under 2 minutes
            url = f'{base_url}?query={query}&per_page=50&page={page}&key=3277920db0ddaf85b5d307dd738c931776b387d3'
            
            # Sending a request to store the data inside the r variable
            r = requests.get(url)
            if r.status_code == 200:
                print("OK")
            else:
                print(r.json())
                print("Erreur HTTP :", r.status_code)
                
            # Getting the json from the response
            self.data = r.json()
            print(self.data.keys())
                
            print("Download started : ", page )
            
            # Storing the JSON inside the folder
            filename = file_path / f'page {page}.json'
            with open(filename, 'w') as saved:
                json.dump(self.data, saved)
                
    
            # Rate limit of one request per second
            time.sleep(1)
            
            # For each recordings in each pages download if the name spec is specified
            # download the audio, else, print there's an issue.
            if self.data.get("recordings") is not None:
                for recordings in self.data.get("recordings"):
                    
                    # If not download reached 100, continue the loop
                    if download >= 600:
                        break
                        
                    file = recordings.get("file")
                    file_name = recordings.get("gen")
                    file_spec = recordings.get("sp")
                    
                    # Checking if there's any existing file
                    existing_files = [f.stem for f in audios_path.glob(f"*{file_name}*{file_spec}*")]
                    if existing_files:
                        print(f"Skipping {file_name} - {file_spec}: already exists")
                        continue
                    
                    try:
                        # Filtering to avoid audio with missing names or species
                        if file_spec and file_name:
                            
                            # Audio file
                            audio_filename = audios_path / recordings["file-name"].split("/")[-1]
                            
                            # Download URL
                            if file.startswith("//"):
                                audio_url = "https:" + file
                            elif file.startswith("/"):
                                audio_url = "https://www.xeno-canto.org" + file
                            else:
                                audio_url = file
                            
                            print(f"\nStarting download {file_name}, {file_spec} → {audio_filename.name}")
                                
                            
                            # Downloading the audio file in a precise file
                            with requests.get(audio_url, stream=True) as x_audio:
                                
                                print("Requesting:", audio_url)
                                print("Status code:", x_audio.status_code)
                                
                                # Stop if can't find any audio
                                x_audio.raise_for_status()
                                
                                # Downloading inside the file
                                with open(audio_filename, "wb") as f:
                                    for chunk in x_audio.iter_content(chunk_size=8192):
                                        f.write(chunk)
                                        
                                # Getting the id of the audio
                                recording_id = recordings['id']
                                
                                # Retrieving the initial suffix
                                original_name = recordings["file-name"]
                                extension = Path(original_name).suffix
                                
                                # Renaming with a new name 
                                new_name = f"{recording_id} - {file_name} - {file_spec}{extension}"
                                new_path = audios_path / new_name
                                audio_filename.rename(new_path)
                                    
                                # Increasing the downloading variable
                                download += 1
                                print("Downloading to:", audio_filename)
                                    
                        else:
                            print("No download because of missing informations.")
                            
                    except Exception as e:
                        print("An error occured : ", e)
                    
        else:
            print("Missing informations.")
            
        print("\nStarting the extraction loop")
        
        # Calling the function after the function downloaded 50 audios
        self.extraction_loop()

    # Extraction loop : Loop extracting necessary informations and implement them
    # inside a dictionnary at the end.
    def extraction_loop(self):
        for file in os.scandir(audios_path):
            
            print(file)
            
            # Extract file name from the title
            file_name = file.name.split("-")[-2]
            # Remove all whitespaces except between words
            file_name = " ".join(file_name.split())
            # To standardize every file name
            file_name = file_name.capitalize()
            
            
            # Extract file species from the title
            file_specie = file.name.split("-")[-1]
            # Removing the .mp3 after the specie name
            file_specie_filt = file_specie.split(".")[0]
            # Remove all whitespaces except between words
            file_specie_filt = " ".join(file_specie_filt.split())
            
            print("\nStarting the handling audios loop")
            
            if not any(data_path.iterdir()):
            
                # Function handling, segmenting and transforming the audios 
                fft_audio = self.handling_audios(file, file_name)
                
                if fft_audio is not None:
                
                    # Separating the result inside the handling_audio function
                    # using the same order that was written after the return funtion
                    amplitude, freq, mean_sd, variance_sd, centroid_max, audio_path = fft_audio
                    
                else :
                    
                    print("No data, passing to the next audio")
                    continue
                    
                # Appending each list to their attended column
                self.names.append(file_name)
                self.species.append(file_specie_filt)
                self.amplitudes.append(amplitude)
                self.frequencies.append(freq)
                self.spectral_descs_mean.append(mean_sd)
                self.spectral_descs_var.append(variance_sd)
                self.spectral_centroids_max.append(centroid_max)
                self.audio_sample.append(audio_path)
                
                # Dictionnary containing audios data
                dict_audio = {
                                "Name": self.names,
                                "Species": self.species,
                                "Amplitude":self.amplitudes,
                                "Frequency":self.frequencies,
                                "Spectral_Desc_Mean" : self.spectral_descs_mean,
                                "Spectral_Desc_Var":self.spectral_descs_var,
                                "Spectral_Centroid":self.spectral_centroids_max,
                                "Audio": self.audio_sample
                            }
                
                # Transform the dictionnary into a dataframe to be used by the model
                self.df_audio = pd.DataFrame.from_dict(dict_audio)
                # print(self.df_audio.head(5))
                
                # Transforming the dataframe to a csv file to be used without calling
                # this program again
                # Pandas handle the dataframe to csv, no need to use "with open"
                csv_file = data_path / 'bird_sound_data.csv'
                self.df_audio.to_csv(csv_file, index=False, encoding='utf-8')
            
            else:
                csv_file = data_path / 'bird_sound_data.csv'
                self.df_audio = pd.read_csv(csv_file)
        
        # Calling the class from the bird_sound_model file
        self.bird_model = BirdModel(self.df_audio)
        # Starting the model process from the running function inside the class
        self.bird_model.run()
                   
    
    # Handling Audios : Function transforming and slicing each audios into small samples,
    # extracting informations from each audios.
    def handling_audios(self, file, file_name):
        
        # Read file using librosa
        signal, sample_rate = librosa.load(file)
        #normal_length = signal.shape[0] / sample_rate
        #print(f"\nNormal length {file_name} = {normal_length}s")
        
        # Compute the Short-Time Fourier Transform (STFT)
        stft = librosa.stft(signal)
        
        # Frames of the signal 
        S = np.abs(stft)
    
        # Find maximum amplitude across frequencies for each time step
        amplitude_envelope = np.max(np.abs(stft), axis=0)
        
        # Onset Detection with frames (more stable than unit=time)
        onset_frames = librosa.onset.onset_detect(y=signal, sr=sample_rate, units='frames')
        # Converting the frames to time
        onset_time = librosa.frames_to_time(onset_frames)
        
        # Creation of the figure
        plt.figure(figsize=(12, 6))
        
        # Plot the waveform
        plt.subplot(3, 1, 1)
        librosa.display.waveshow(signal, sr=sample_rate, alpha=0.5)
        plt.title(f'Waveform {file_name}')
        
        # Plot the peak amplitude envelope
        plt.subplot(3, 1, 2)
        plt.plot(librosa.frames_to_time(np.arange(len(amplitude_envelope)), 
                                        sr=sample_rate), amplitude_envelope, 
                                        label='Max Amplitude')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title(f'Max amplitude {file_name}')
        plt.legend()
        
        # Plot onset times as vertical lines
        plt.subplot(3, 1, 3)
        librosa.display.waveshow(signal, sr=sample_rate, alpha=0.5)
        plt.vlines(onset_time, -1, 1, color='g', linestyle='dashed', label='Onsets')
        plt.legend()
        plt.title(f'Onset Detection {file_name}')
        
        # Reduce the figure size
        plt.tight_layout()
        
        # Saving every figures as pictures inside a file
        plt.savefig(f'{image_path}/{file_name}.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        
        # Plot the spectrogram
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), 
                                 sr=sample_rate, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Spectrogram {file_name}')
        plt.tight_layout()
        
        # Saving every figures as pictures inside a file
        plt.savefig(f'{image_path}/Spectrogram {file_name}.png', dpi=600, bbox_inches='tight')
        plt.show()
        plt.close()
        
        # Now, using the onset results to find the highest peak of each file
        # cut a sample from it and store it inside the dictionnary
        # Where there's onset -> take a small sample at [before, after]
        
        # print(f"\nOnset Time of {file_name} = {onset_time}")
        
        # Onset Time is a list of floats in time (secs)
        # We need to convert it to samples index and integers
        # To take the necessary audio part, we'll slice it.
        
        
        # Time * sample_rate to retrieve the index sample, same type as the
        # audio at the start of the function
        sample_index = onset_time * sample_rate
        
        try:
        
            # Using mean for the slicing as a threshold to be above of
            onset_mean = int(sample_index.mean())
            
            # Slicing sample_index with only the numbers/variations above the mean
            # Only the first mean, not the entire list
            int(sample_index[sample_index > onset_mean][0])
            
        except Exception as e:
            
            print("Error, pass the audio : ", file)
            print(e)
            
            return None
        
        # Using new_sample first index as a start
        start = int(sample_index[0])
        
        # Converting 2 secs to frames from the start
        end = int(start + 2 * sample_rate)
        
        # To avoid overtake the signal length
        end = min(end, len(signal))
        
        # The new signal ready to be incorporated inside the dataset
        new_signal = signal[start:end]
        
        # Compute the Short-Time Fourier Transform (STFT)
        stft_new = librosa.stft(new_signal)
    
        # Find maximum amplitude across frequencies for each time step
        amplitude_envelope_new = np.max(np.abs(stft_new), axis=0)
        
        # Variance of the amplitude of the audio
        new_amplitude = np.var(amplitude_envelope_new)
        
        # Creation of a new figure to compare normal and new signal
        plt.figure(figsize=(12, 6))
        
        # Plot the waveform
        plt.subplot(2, 1, 1)
        plt.plot(librosa.frames_to_time(np.arange(len(amplitude_envelope)), 
                                        sr=sample_rate), amplitude_envelope, 
                                        label='Max Amplitude')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title(f'Max amplitude Normal Signal {file_name}')
        plt.legend()
        
        # Plot the peak amplitude envelope
        plt.subplot(2, 1, 2)
        plt.plot(librosa.frames_to_time(np.arange(len(amplitude_envelope_new)), 
                                        sr=sample_rate), amplitude_envelope_new, 
                                        label='Max Amplitude')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title(f'Max amplitude New Signal {file_name} - Start {start}')
        plt.legend()
        
        # Saving every figures as pictures inside a file
        plt.savefig(f'{image_path}/Comparison {file_name}.png', dpi=600, bbox_inches='tight')
        plt.show()
        plt.close()
        
        # Extracting the informations we need in our dictionnary :
            
        # The Frequency, since we have the amplitude of the sample
        freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=2048)
        
        # Storing the maximum frequency only
        new_signal_freqs = int(freqs.mean())
        
        # Having a description of the spectral audio to be added inside the dict
        # from time-series :
        
        # Using Mel-frequency cepstral coefficients (MFCCs)
        # From the feature extraction function found in the librosa library
        
        # Output : The tone/pitch of a sound
        
        # The spectral description 
        spec_desc = librosa.feature.mfcc(y=new_signal, sr=sample_rate)
        
        # Mean to be added to the dictionnary : New Signal Mean Spectral Desc
        # Résume les sons : gloablement ça ressemble à quoi ? Où est le centre ?
        nsmsd = int(spec_desc.mean())
        
        #print(f"\n{file_name} Mean Spectral Description : {nsmsd}")
        
        # Variance of the NSMSD - Output : stable sound or not (birdsong or scream).
        nsvsd = int(np.var(spec_desc))
        
        #print(f"{file_name} Variance Spectral Description : {nsvsd}")
        
        # Using Spectral Centroid to find the center
        # Each frame of a magnitude spectrogram is normalized and 
        # treated as a distribution over frequency bins, from which the mean 
        # (centroid) is extracted per frame.
        
        # Output : The audio/birdsong is either high or low pitch
        new_signal_centroid = librosa.feature.spectral_centroid(y=new_signal, 
                                                                sr=sample_rate)
        
        # Storing only its maximum
        new_signal_centroid_max = int(new_signal_centroid.max())
        
        #print(f"{file_name} Centroid Spectral Description : {new_signal_centroid_max}")
        
        print("\nStarting Sample To Audio")
        
        self.sample_to_audio(sample_rate, new_signal, file_name)
        
        print("\nFinished Sample To Audio")
        
        # Return all the values to be stored in the variables inside the dictionnary
        return new_amplitude, new_signal_freqs, nsmsd, nsvsd, new_signal_centroid_max, self.output_path
    
    # Turning the sample into fragmented audios
    def sample_to_audio(self, sample_rate, sample, filename):
        
        # Output Path for samples
        self.output_path = sample_audio_path / f"{filename}.wav"
        
        # Rewrite the sample into audio shape
        write(self.output_path, sample_rate, sample)
    
    # Running the program
    def run(self):
        
        # If there's nothing in the file path, run the fetching function
        # if there's something, run the extraction function
        if not any(file_path.iterdir()):
            self.fetching_data()
            
        else:
            self.extraction_loop()
 
            
if __name__ == "__main__":
    
    birdsong = BirdSongs()
    birdsong.run()
