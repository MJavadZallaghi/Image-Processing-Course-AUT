% DIP Course - fall 2020 - HW: 5
% Student: M. J. Zallaghi    -    ID: 98126079

% Question: 2 code
clc;
clear all;
%% reading image and resizing
I =  rgb2gray(imread('fMRI.jpg'));
I = imresize(I,[512 512]);

%% using from developed function
segments = [512 256 128 64 32 16 8 4 2 1];
Similarity_Threshold = 0.4;
segmented_img = split_and_merge(I, segments, Similarity_Threshold);

%% showing results
figure(1);
subplot(1,2,1);
imshow(I);
title('main Image');
subplot(1,2,2);
imshow(segmented_img,[]);
title(['segmented Image, with segments:', num2str(segments)]);
%% Split and merge algorithm
function blocks =  split_and_merge(I,segment_size, Similarity_Threshold)
S = qtdecomp(I, Similarity_Threshold);
blocks = repmat(uint8(0), size(S));

for dim = segment_size
  numblocks = length(find(S==dim));
  if (numblocks > 0)
    values = repmat(uint8(1), [dim dim numblocks]);
    values(2:dim,2:dim,:) = 0;
    blocks = qtsetblk(blocks,S,dim,values);
  end
end
blocks(end, 1:end) = 1;
blocks(1:end, end) = 1;
end
