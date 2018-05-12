clear all;

filename = 'proc_test_1.mat';
filepath = ['./data/test/',filename];
load(filepath)

slide = [0;86;242;300;399];
% slide = [0;25;200;250;399];

for i = 1:length(slide)-1
    tmp = DS(:,(slide(i)+1):slide(i+1));
    save(['./data/test_',num2str(i),'.mat'],'tmp')
end