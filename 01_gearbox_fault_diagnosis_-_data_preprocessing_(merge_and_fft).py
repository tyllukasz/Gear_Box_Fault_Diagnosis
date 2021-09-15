#imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#data import from .csv files
#-----------------------------------------------------------------------------------------------------------
datas_broken = ['b30hz0.csv','b30hz10.csv','b30hz20.csv','b30hz30.csv','b30hz40.csv',
         'b30hz50.csv','b30hz60.csv','b30hz70.csv','b30hz80.csv','b30hz90.csv',]

dfs_broken = []

load = 0

for data_name in datas_broken:
    
    df = pd.read_csv('RAW_data/BrokenTooth/'+data_name)
    df['load'] = load
    df['failure'] = 1
    dfs_broken.append(df)
    
    load += 10

#-----------------------------------------------------------------------------------------------------------
datas_healthy = ['h30hz0.csv','h30hz10.csv','h30hz20.csv','h30hz30.csv','h30hz40.csv',
         'h30hz50.csv','h30hz60.csv','h30hz70.csv','h30hz80.csv','h30hz90.csv',]

dfs_healthy = []

load = 0

for data_name in datas_healthy:
    
    df = pd.read_csv('RAW_data/Healthy/'+data_name)
    df['load'] = load
    df['failure'] = 0
    dfs_healthy.append(df)
    
    load += 10
#-----------------------------------------------------------------------------------------------------------

#FFT parameters

# time step [s] (in this case arbitrary choice - no info from data set about sampling frequency)
t_step = 0.001

# sampling frequency [Hz]
sample_rate = 1/t_step

# buffer size (number of samples)
buffer_size = 128

# frequency interval [Hz] (in frequency domain)
f_step = sample_rate / buffer_size
#-----------------------------------------------------------------------------------------------------------


# columns in new data frame

columns_a1 = []
columns_a2 = []
columns_a3 = []
columns_a4 = []
other_labels = ['load','failure']

for i in range (0, int(buffer_size / 2 + 1)):
    columns_a1.append('a1_freq_'+str(i))
    columns_a2.append('a2_freq_'+str(i))
    columns_a3.append('a3_freq_'+str(i))
    columns_a4.append('a4_freq_'+str(i))

#-----------------------------------------------------------------------------------------------------------



dfs = [dfs_healthy,dfs_broken]

new_cols = columns_a1 + columns_a2 + columns_a3 + columns_a4 + other_labels

processed_df = pd.DataFrame(columns=new_cols)


for i0 in range(0,2):
    
    df_list = dfs[i0] # broken or healthy
    
    #------------------------------------------------------
    for i1 in range(0,len(df_list)):
        
        df = df_list[i1] # loads: 0%, 10%, 20% ... 90%
        
        #--------------------------------------------------
        for i2 in range(0 , int(len(df) // buffer_size) ):
            
            i_start = int(0 + i2 * buffer_size) # range start
            i_end = int(buffer_size + i2 * buffer_size) # range end
            
            df_buffer = df.iloc[i_start:i_end] # batch from data frame with size defined by buffer size
            
            amplitude_all = np.array([]) # all values from 4 sensors in one array
            
            #----------------------------------------------
            for i3 in range(0,4):
                
                #FFT calculation
                current_fourier = np.fft.fft(df_buffer['a'+str(i3+1)])
                freq_samples = int(buffer_size / 2 + 1)

                amplitude = (2 * (np.abs(current_fourier) / buffer_size))[:freq_samples] #Nyquist frequency (sampling_rate / 2)
                amplitude[0] = amplitude[0] / 2
                amplitude = amplitude.reshape(1,len(amplitude)) #for append to data frame function
                
                amplitude_all = np.append(amplitude_all,amplitude)
            
            #----------------------------------------------
            
            all_features = np.append(amplitude_all,i1)
            all_features = np.append(all_features,i0) #all features (FFT for a1,a2,a3,a4 and load, failure) size should be 262 for 128 buffer size
            all_features = all_features.reshape(1,len(all_features))

            temp_df = pd.DataFrame(all_features,columns = new_cols)
            
            processed_df = processed_df.append(temp_df,ignore_index=True)
            
        #--------------------------------------------------
    #------------------------------------------------------


processed_df.to_csv('processed_df.csv')