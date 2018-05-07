clear all; close all;

filename = 'walking/proc_feng_walking_4';
filepath = ['./data/',filename];
load(filepath)

DS = DS(:,40:174);
save(filepath,'DS')