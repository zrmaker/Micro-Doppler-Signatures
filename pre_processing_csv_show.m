clear all; close all;

winsize = 40;

% cd ./data/nothing

[x,y] = meshgrid(0:(winsize-1),-512:511);
[xq,yq] = meshgrid(0:(winsize-1),linspace(-512,511,winsize));
data_x = [];
data_y = [];

cd ./data/nothing
disp('Processing nothing...')
mat = dir('*.mat');
for i = 1:length(mat) 
    load(mat(i).name);
    for j = 1:(size(DS,2)-winsize+1)
        vq = griddata(x,y,DS(:,j:j+winsize-1),xq,yq);
        data_x = [data_x; reshape(vq,1,[])];
        data_y = [data_y; 0];
    end
    disp(['Progress: ',num2str(i/length(mat)*100),'%'])
end 

cd ../walking
disp('Processing walking...')
mat = dir('*.mat');
for i = 1:length(mat) 
    load(mat(i).name);
    for j = 1:(size(DS,2)-winsize+1)
%         imagesc(DS(:,j:j+winsize-1))
%         figure
        vq = griddata(x,y,DS(:,j:j+winsize-1),xq,yq);
%         imagesc(vq)
        data_x = [data_x; reshape(vq,1,[])];
        data_y = [data_y; 1];
    end
    disp(['Progress: ',num2str(i/length(mat)*100),'%'])
end 
cd ../..
disp('Saving...')
csvwrite('./database/train_x.csv',data_x)
csvwrite('./database/train_y.csv',data_y)
datasize = length(y);
uy = unique(data_y);
s = struct('datasize',datasize,'unique_values',uy)
save('./database/info','s')