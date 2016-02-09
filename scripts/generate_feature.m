clear;
load imdbTmV.mat;

frames = dlmread('train_framenum.txt');
totalframes = sum(frames);

matObj = matfile('train_features.h5','Writable',true);
matObj.features(50176, totalframes) = single(0);

frames_seen = 0;
for i = 1:size(frames,1)
        fprintf('%d\n',i);
	a = strsplit(imdb.trainset{i},'/');
        a = strsplit(a{end},'.');
        a = a{1};
        filename = strcat('/ais/gobi3/u/shikhar/hmdb/dataset/split1/feat_cache/',a,'.mat');
        load(filename);
	feat = full(feat);
        matObj.features(1:50176,frames_seen+1:frames_seen+frames(i)) = single(feat');
        clear('feat');
        frames_seen = frames_seen + frames(i);
end

