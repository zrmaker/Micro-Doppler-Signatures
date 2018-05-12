% function trained_show()
clear all; close all;

filename = 'test_1.mat';
filepath = ['./data/',filename];
load(filepath)

pred = csvread('./output/pred.csv');

dopplerfreq = [-512:511].'./1024.*(1/256e-6);

DS = zeros(NFFTVel, NrFrms);
rdprev = squeeze(rdcali(1,:,:));
DSshow = zeros(NFFTVel, 200);
figure(1)
set(gcf,'units','normalized','outerposition',[0 0 1 1])
for MeasIdx     =   1:NrFrms
	data = squeeze(DataStore(MeasIdx).rd_scan);
    RP          =   fft(data(2:end,:).*Win2D,NFFT,1).*FuSca/ScaWin;
    RPExt       =   RP(RMinIdx:RMaxIdx,:);
    RD          =   fftshift(fft(RPExt.*WinVel2D, NFFTVel, 2)./ScaWinVel,2);
    RD2      =   RD - rdprev;
    rdprev = RD;
    RDrev=abs(RD2).*repmat(vRangeExt.^2,1,NFFTVel);

    DP = sum(RDrev,1).';
    DS(:,MeasIdx) = DP;
%     t = (-199*128e-4:128e-4:0)+128e-4*MeasIdx;
%     DSshow(:,1:199) = DSshow(:,2:200);
%     DSshow(:,200) = DP;
%     DSshow2=abs(DSshow)./max(max(abs(DSshow)));
%     
%     subplot(2,2,1)
%     imagesc(abs(RDrev))
%     caxis([0 1e-3])
%     colormap(yuan)
%     subplot(2,2,2)
%     imagesc(imagestore(MeasIdx).img)
%     subplot(2,2,3)
%     imagesc(t, dopplerfreq, db(DSshow2))
%     grid on
%     xlabel('Time (s)')
%     ylabel('Doppler frequency (Hz)')
%     caxis([-45 -15])
%     colormap(yuan)
%     subplot(2,2,4)
%     plot(pred(1:MeasIdx,:))
%     xlim([1 360])
%     ylim([-.5 1.5])
% %     legend('p_{none}','p_{walk}','p_{sit}','p_{wave}')
%     pause(.0000001)
end
DS=DS(:,2:end);
DS=abs(DS)./max(max(abs(DS)));
t = 40*128e-4:128e-4:(size(DS,2)-1)*128e-4;
figure(1)
subplot(2,1,1)
imagesc(t,dopplerfreq,db(DS(:,41:end)))
grid on
xlabel('Time (s)')
ylabel('Doppler frequency (Hz)')
caxis([-45 -15])
colormap(yuan)

subplot(2,1,2)
plot(pred)
xlim([1 760])
ylim([-.5 1.5])