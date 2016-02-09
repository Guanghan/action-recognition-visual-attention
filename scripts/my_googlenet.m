function my_googlenet

addpath('~/Projects/segRCNN/caffe-master/matlab/caffe/');
addpath('~/Projects/segRCNN/deep-rcnn/');

% extract raw CNN features per frame and save to feat_cache

output_dir = ['feat_cache/' ];
if ~exist(output_dir,'dir')
    mkdir(output_dir);
end

%% init caffe
try
  id = obtain_gpu_lock_id;
  disp(id)
catch
  disp('cannot find a working GPU')
end

keyboard;

caffe('set_device', id);
model_def_file = '/u/yukun/Projects/segRCNN/caffe-master/caffe-models/forward_GoogLeNet_outputconv.prototxt';
model_file = '/u/yukun/Projects/segRCNN/caffe-master/caffe-models/GoogLeNet';
matcaffe_init(true,model_def_file,model_file);

load('/u/yukun/Projects/segRCNN/caffe-master/caffe-models/GoogLeNet_mean.mat');

load('imdb.mat')

data = cat(1, imdb.trainset, imdb.testset);

% for each shot, cache output from hybrid CNN
tic;
for i=1:numel(data)
  if toc>3
    fprintf('%d/%d\n', i,numel(data));
    tic;
  end
  
  file_name_raw = data{i}{1}; 
  [~, file_name] = fileparts(file_name_raw);

  if exist([output_dir filesep file_name '.mat'],'file')
    continue;
  end
    
    readerobj = VideoReader(file_name_raw);
    
    vidFrames = read(readerobj);
    
    imseq = cell(1,size(vidFrames,4));
    for j=1:size(vidFrames,4)
        imseq{j} =  vidFrames(:,:,:,j);
    end
    
    feat = caffe_forward_googlenet(imseq,image_mean);
    feat = sparse(feat);
    
    save([output_dir filesep file_name '.mat'],'feat','-v7.3');
end



function [feat] = caffe_forward_googlenet(imseq,image_mean)

% prepare oversampled input
% input_data is Height x Width x Channel x Num
input_data = prepare_image(imseq,image_mean);

% do forward pass to get scores
% scores are now Width x Height x Channels x Num
feat = zeros(numel(imseq),7*7*1024);
cnt = 0;
for i=1:numel(input_data)
    scores = caffe('forward', {input_data{i}});
    
    if cnt+128<=numel(imseq)
    feat(cnt+1:cnt+128,:)=reshape(scores{1},[7*7*1024,128])';
    cnt = cnt + 128;
  else
    T = reshape(scores{1},[7*7*1024,128])';
    feat(cnt+1:end,:) = T(1:numel(imseq)-cnt,:);
  end

    %keyboard;
end



% ------------------------------------------------------------------------
function batch = prepare_image(imbatch,IMAGE_MEAN)
% ------------------------------------------------------------------------

IMAGE_DIM = 256;
CROPPED_DIM = 224;

% resize to fixed input size
cnt = 1;
batch{cnt} = zeros(CROPPED_DIM, CROPPED_DIM, 3, 128, 'single');

for i=1:numel(imbatch)
  if i>128*cnt
    cnt = cnt + 1;
    batch{cnt} = zeros(CROPPED_DIM, CROPPED_DIM, 3, 128, 'single');
  end
  im = single(imbatch{i});
  im = imresize(im, [IMAGE_DIM IMAGE_DIM], 'bilinear');
  % permute from RGB to BGR (IMAGE_MEAN is already BGR)
  im = im(:,:,[3 2 1]) - IMAGE_MEAN;
  im = imresize(im, [CROPPED_DIM CROPPED_DIM], 'bilinear');
  batch{cnt}(:,:,:,i-128*(cnt-1)) = permute(im, [2 1 3]);
end

