clear all; close all;

yuan = [[ 1, 1, 1];
    [ 1, 1,.9625];
    [.8,.8,.8625];
    [.6,.6,.7625];
    [.4,.4,.6625];
    [.2,.2,.5625];
    jet;];

dopplerfreq = [-1024./2:1024./2-1].'./1024.*(1/256e-6);

cd ./data/20180508/waving
mat = dir('*.mat');
for i = 1:length(mat) 
    load(mat(i).name);
    t = 0:128e-4:(size(DS,2)-1)*128e-4;
    figure(1)
%     imagesc(t,dopplerfreq,db(DS))
    imagesc(db(DS))
    grid on
    xlabel('Time (s)')
    ylabel('Doppler frequency (Hz)')
    caxis([-45 -15])
    colormap(yuan)
    colorbar
    title(mat(i).name, 'Interpreter', 'none')
    key
end 
close all;
cd ../../..

function key
    k = waitforbuttonpress;
    if k == 1
        return
    else
        key
    end
end
