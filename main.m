clc;
clear all;


addpath(genpath(pwd));
   
load('MSCCA_Data.mat');
load('data_1_to_5.mat');
VBM_1(:,[9:26, 109:116]) = [];
FDG_1(:,[9:26, 109:116]) = [];
VBM_5(:,[9:26, 109:116]) = [];
FDG_5(:,[9:26, 109:116]) = [];
VBM = [VBM_1;VBM_5];
FDG = [FDG_1;FDG_5];
Score(:,[1,2,3,4]) = [];
Score(Score == 2) = [];
Score(Score == 3) = [];
Score(Score == 4) = [];
Score(Score == 5) = 2;
label = sort(Score);
MRI = VBM;
PET = FDG;

MRI = bsxfun(@rdivide, bsxfun(@minus, MRI, mean(MRI)), std(MRI));
MRI = bsxfun(@rdivide, MRI, sqrt(sum(MRI.^2, 2)));
PET = bsxfun(@rdivide, bsxfun(@minus, PET, mean(PET)), std(PET));
PET = bsxfun(@rdivide, PET, sqrt(sum(PET.^2, 2)));

m = length(label);                                      
                               
%acc = zeros(10,10);                                    
all_truelabel = [];                                     
all_pvalue = [];                                        
all_predlabel = [];
all_W = [];


lambda1 =0.01;
lambda2 = [1e-10 1e-9 1e-8 1e-7 1e-6 1e-5 1e-4 1e-3 1e-2 1e-1 0 ]; 
lambda =[1e-10 1e-9 1e-10 1e-7 1e-6 1e-5 1e-4 1e-3 1e-2 1e-1 0 ];



all_acc = zeros(length(lambda1),length (lambda2));  

sigma = 10;                                         	
point_num = 7;                               			
exp_num = 10;                                 			
groups = ismember(label,1);                         	

tic
for ii = 1:length(lambda)
    for jj = 1:length(lambda2)

    Indices = crossvalind('Kfold', groups, 10);        
   
    pvalue = [];                                        
    truelabel = [];                                     
    predlabel = [];                                     
    W = [];
    acc_i = zeros(10,1); 
    for i = 1:10                                      
        val_set_index = (Indices == i);                 
       
        testlabel = label(val_set_index);               
       
        train_set_index =~ val_set_index;
        MRItr = MRI(train_set_index,:);                 
        PETtr = PET(train_set_index,:);                         
        trainlabel = label(train_set_index);            
       
        X = {MRItr, PETtr};                             
        Y = {trainlabel, trainlabel};                  
       
        [H1,Lh1] = cons_hypergraph(MRItr, point_num);      
        [H2,Lh2] = cons_hypergraph(PETtr, point_num);       	
       
        opts = struct('init',[0],'tFlag',[1],'tol',[10^-4],'maxIter',[600]);
       
        maxi = 0;                                       
        opts = struct('init',[0],'tFlag',[1],'tol',[10^-4],'maxIter',[1000],'rho1',[lambda(ii)],'rho_L3',lambda1,'rho4',[lambda2(jj)]);
        [w, funcval] = h_MTM_APG(X, Y, opts, Lh1, Lh2);
        opts = struct('init',[1],'tFlag',[1],'tol',[10^-4],'maxIter',[1000],'rho1',[lambda(ii)],'rho_L3',lambda1,'rho4',[lambda2(jj)],'W0',[w]);
        MRItr_selec = MRItr(:,abs(w(:,1))>1e-2); 
        PETtr_selec = PETtr(:,abs(w(:,2))>1e-2);
        [temp,ps] = mapstd(MRItr_selec',0,1);    
        MRItr_selec = temp';
        [temp,ps] = mapstd(PETtr_selec',0,1);    
        PETtr_selec = temp';
        MRIte_selec = MRI(val_set_index, abs(w(:,1))>1e-2); 
        PETte_selec =  PET(val_set_index, abs(w(:,2))>1e-2);
        [temp,ps] = mapstd(MRIte_selec',0,1);   
        MRIte_selec = temp';
        [temp,ps] = mapstd(PETte_selec',0,1);   
        PETte_selec = temp';
        MRIK = calckernel('linear', sigma, MRItr_selec);
        MRIKT = calckernel('linear', sigma, MRItr_selec, MRIte_selec);
        PETK = calckernel('linear', sigma, PETtr_selec);
        PETKT = calckernel('linear', sigma, PETtr_selec, PETte_selec);
        [best_acc, beta, best_label, best_pvalue] = gridSearchMKL(MRIK, MRIKT, PETK, PETKT, label, train_set_index, val_set_index);
        acc_i( i ) = best_acc;
    end
        all_acc(ii,jj) = mean(acc_i); 
 end
   
    end
  
toc

disp(all_acc);                                      
