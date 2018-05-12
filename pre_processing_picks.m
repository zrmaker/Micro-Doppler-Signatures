clear all; close all;

filename = '/20180508/waving/proc_feng_waving_2';
filepath = ['./data/',filename];
load(filepath)

DS = DS(:,43:end);
save(filepath,'DS')