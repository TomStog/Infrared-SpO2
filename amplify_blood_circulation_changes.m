clear ;
close all;
clc;

% Step 
% This code applies spatial Gdown temporal function in order to amplify the
% color changes that blood circulation create on the face, so to make it
% visible.

% Define video paths
%video_path = '..\Forehead_Att\Forehead\';
%video_path = '..\Left_Cheek_Att\Left_Cheek\';
video_path = '..\Right_Cheek_Att\Right_Cheek\';

%day_subfolders = ['Day1\'; 'Day2\'; 'Day3\'];
day_subfolders = ['Right_out\'];

day_len = size(day_subfolders, 1);
person_subfolders = ['01\';'02\';'03\';'04\';'05\';'06\';'07\';'08\';'09\';'10\';'11\';'12\';'13\'];

person_len = size(person_subfolders, 1);

% Define out video path
%out_video_path = '..\Forehead_Amp\Forehead\';
%out_video_path = '..\Left_Cheek_Amp\Left_Cheek\';
out_video_path = '..\Right_Cheek_Amp\Right_Cheek\';

alpha = 120;
level = 4;
chromAttenuation = 1;
fl = 0.4;
fh = 4.0;

% For every day 
for i=1:day_len

    % For every person
    for j=1:person_len

        % Define video path
        vpath = [video_path day_subfolders(i, :) person_subfolders(j, :)];
        files = dir(vpath);

        % For every video in vpath
        for k=3:length(files)

            % Load video
            vfilename = [vpath files(k).name];
            v = VideoReader(vfilename);
            samplingRate = v.FrameRate;

            % Construct video writer
            [fpath, name, ext] = fileparts(files(k).name);
            out_vfilename = [out_video_path day_subfolders(i, :) person_subfolders(j, :)];

            tic

            amplify_spatial_Gdown_temporal_ideal_Only_Amplification(vfilename,out_vfilename,alpha,level, ...
                     fl,fh,samplingRate, chromAttenuation);

            toc

        end
    end
end
