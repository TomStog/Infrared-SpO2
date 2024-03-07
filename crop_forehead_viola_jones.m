clear all
close all
clc

addpath(genpath('..\Codes\face-release1.0-basic'));
addpath(genpath('..\Codes'));

% Load model for face detection
load face_p99.mat; % model

% Load AAM model
load cGN_DPM.mat;

cd '..\Codes';

% Define video paths
video_path = '..\Video_Person_ID\Experiment_Jun_2022\';
day_subfolders = ['Day1\'];

day_len = size(day_subfolders, 1);
person_subfolders = ['03\'];

person_len = size(person_subfolders, 1);

% Define out video path
out_video_path = '..\Forehead_out\';

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
            
            % Construct video writer
            [fpath, name, ext] = fileparts(files(k).name);
            
            % Forehead
            out_vfilename = [out_video_path 'Forehead\' day_subfolders(i, :) person_subfolders name] 
            vwriter = VideoWriter(out_vfilename, 'Uncompressed AVI');
            vwriter.FrameRate = v.FrameRate;
            
            open(vwriter);
            
            tic
            
            % Extract 1st frame
            frame = readFrame(v);
            
            % Check if the first frame was read successfully
            if isempty(frame)
                warning('Failed to read the first frame from the video.');
                continue; % Skip to the next video
            end
            
            faceDetector = vision.CascadeObjectDetector('FrontalFaceCART'); %Create a detector object
            img = rgb2gray(frame(:,:,:)); % convert to gray
            BB = step(faceDetector,img); % Detect faces
            [numRows,~] = size(BB);
            if numRows==1
                fitbox(1,1) = BB(1,2);
                fitbox(1,2) = BB(1,1);
                fitbox(1,3) = BB(1,3);
                fitbox(1,4) = BB(1,4);
            else
                [~, idx] = max(BB(:,3));
                fitbox(1,1) = BB(idx,2);
                fitbox(1,2) = BB(idx,1);
                fitbox(1,3) = BB(idx,3);
                fitbox(1,4) = BB(idx,4);
            end
            
            % Create forehead rectangle
            my_w = fitbox(1,3)*(1/3-sqrt(5)/9);
            forehead_rect = [(fitbox(1,2)+round(fitbox(1,3)/5)), fitbox(1,1), round(3*fitbox(1,3)/5), round(fitbox(1,3)/3-my_w)];
            
            % For every frame
            while hasFrame(v)
                
                % Cropped face 
                cropped_frame = imcrop(frame, forehead_rect);
                
                % Write cropped frame in new video
                writeVideo(vwriter, cropped_frame);
                
                % Read new frame
                if hasFrame(v)
                    frame = readFrame(v);
                else
                    break; % Break the loop if no more frames are available
                end
            end
            
            close(vwriter);
            toc
            %break
        end        
    end
end