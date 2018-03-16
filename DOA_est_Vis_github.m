close all;
clear;
clc;

load('DOA_test_OP.mat');                        % Change the filename to your output filename
DOA_estimates = permute(Output, [2 1]);
target = h5read('DOA_test.hdf5','/targets');    % Loads the targets


sample = 5;                                     % There are total 9 samples in the dataset, each correponding to a specifc angle


% Get the indices within the file for the specific sample
sample_beg = 809*(sample)+1;
factor = 808;

% Extract the specific sample from the files
plot_target = target(:,sample_beg:sample_beg+factor);
plot_data = DOA_estimates(:,sample_beg:sample_beg+factor);

true_angle = (find(plot_target(:,10))-1)*5;
% Visualization

figure;

subplot(211)
imagesc(1:size(plot_target,2),0:5:180,plot_data(:,:));colormap(hot); 
colorbar; 
xlabel('Time frames','FontSize',16,'Interpreter','latex'); ylabel('DOA ($^\circ$)','FontSize',16,'Interpreter','latex');
axis xy;
caxis([0 1]);
set(gca,'FontSize',16);
title('Estimates')
hold on 
subplot(212)
imagesc(1:size(plot_target,2),0:5:180,plot_target(:,:));colormap(hot);
colorbar; 
axis xy; 
caxis([0 1]);
xlabel('Time frames','FontSize',16,'Interpreter','latex'); ylabel('DOA ($^\circ$)','FontSize',16,'Interpreter','latex');
set(gca,'FontSize',16);
title(sprintf('True DOA - %1.0f',true_angle))