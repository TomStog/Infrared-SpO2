clear;
clc;
close all;
format short

mypath1 = '..\Forehead_Att\Forehead\Day1\03\';
info1 = dir(mypath1);

mypath2 = '..\Forehead_Amp\Forehead\Day1\03\';
info2 = dir(mypath2);

mypath3 = '..\Left_Cheek_Att\Left_Cheek\Day1\03\';
info3 = dir(mypath3);

mypath4 = '..\Left_Cheek_Amp\Left_Cheek\Day1\03\';
info4 = dir(mypath4);

mypath5 = '..\Right_Cheek_Att\Right_Cheek\Day1\03\';
info5 = dir(mypath5);

mypath6 = '..\Right_Cheek_Amp\Right_Cheek\Day1\03\';
info6 = dir(mypath6);

M =[];
channel_sel = 2;

for i = 1:length(info4)-2
    
    str1 = string(mypath1);
    str2 = string(info1(i+2).name);
    str = strcat(str1,str2);
    vidObj = VideoReader(str);
    allFrames1 = read(vidObj);
    var1 = vidObj.Duration*vidObj.FrameRate;

    str1 = string(mypath2);
    str2 = string(info2(i+2).name);
    str = strcat(str1,str2);
    vidObj = VideoReader(str);
    allFrames2 = read(vidObj);
    var2 = vidObj.Duration*vidObj.FrameRate;

    str1 = string(mypath3);
    str2 = string(info3(i+2).name);
    str = strcat(str1,str2);
    vidObj = VideoReader(str);
    allFrames3 = read(vidObj);
    var3 = vidObj.Duration*vidObj.FrameRate;

    str1 = string(mypath4);
    str2 = string(info4(i+2).name);
    str = strcat(str1,str2);
    vidObj = VideoReader(str);
    allFrames4 = read(vidObj);
    var4 = vidObj.Duration*vidObj.FrameRate;

    str1 = string(mypath5);
    str2 = string(info5(i+2).name);
    str = strcat(str1,str2);
    vidObj = VideoReader(str);
    allFrames5 = read(vidObj);
    var5 = vidObj.Duration*vidObj.FrameRate;

    str1 = string(mypath6);
    str2 = string(info6(i+2).name);
    str = strcat(str1,str2);
    vidObj = VideoReader(str);
    allFrames6 = read(vidObj);
    var6 = vidObj.Duration*vidObj.FrameRate;
    
    varm1 = min([var1,var2]);
    varm2 = min([var3,var4]);
    varm3 = min([var5,var6]);    

    for j=1:varm1
        if channel_sel==2
            G1 = rgb2gray(allFrames2(:,:,:,j));
        else
            G1 = rgb2gray(allFrames1(:,:,:,j)) + rgb2gray(allFrames2(:,:,:,j));
        end
        G1 = im2double(G1);
        meanG1(j) = mean(G1(:));
        stdevG1(j) = std(G1(:));   
    end

    for j=1:varm2
        if channel_sel==2
            G2 = rgb2gray(allFrames4(:,:,:,j));
        else
            G2 = rgb2gray(allFrames3(:,:,:,j)) + rgb2gray(allFrames4(:,:,:,j));
        end
        G2 = im2double(G2);
        meanG2(j) = mean(G2(:));
        stdevG2(j) = std(G2(:));
    end

    for j=1:varm3
        if channel_sel==2
            G3 = rgb2gray(allFrames6(:,:,:,j));
        else
            G3 = rgb2gray(allFrames5(:,:,:,j)) + rgb2gray(allFrames6(:,:,:,j));
        end
        G3 = im2double(G3);
        meanG3(j) = mean(G3(:));
        stdevG3(j) = std(G3(:));
    end

    temp_1 = mean(meanG1(:));
    temp_2 = mean(stdevG1(:));
    temp_3 = std(meanG1(:));
    temp_4 = std(stdevG1(:));

    temp_5 = mean(meanG2(:));
    temp_6 = mean(stdevG2(:));
    temp_7 = std(meanG2(:));
    temp_8 = std(stdevG2(:));

    temp_9 = mean(meanG3(:));
    temp_10 = mean(stdevG3(:));
    temp_11 = std(meanG3(:));
    temp_12 = std(stdevG3(:));

    M = [M;temp_1 temp_2 temp_3 temp_4 temp_5 temp_6 temp_7 temp_8 temp_9 temp_10 temp_11 temp_12];

end
xlswrite('filename_12vars_3rd_group_mag.xlsx',M)