function info = fcnTestBBX(varargin)

%direction = '/home/dena/Projects/FCN-TextDetection/data/ch4_test_images'; outputString='/home/dena/Projects/FCN-SynthText/Scores/COCO-Text/%s.csv';

%direction = '/home/dena/Projects/FCN-TextDetection/data/ch4_training_images'; netPath='data/fcn32s-synthText/net-epoch-100.mat';

%direction = '/home/fcn/dena/ICDAR_VAL/input/'; outputString='/home/fcn/dena/ICDAR_VAL/FCN_Synth_100_ep/%s.csv'; netPath='data/fcn32s-synthText/net-epoch-100.mat';
%direction = '/home/fcn/dena/icdar_ch4_val/input/'; outputString='/home/fcn/dena/icdar_ch4_val/hm_ICDAR_FCN_400_epoch/%s.png'; netPath='/home/dena/Projects/FCN-TextDetection2/data/fcn32s-icdar/net-epoch-400.mat' ;

%direction = '/home/fcn/dena/COCO_Text_VAL/input'; outputString='/home/fcn/dena/COCO_Text_VAL/hm_Synth_FCN_100_epoch/%s.csv'; netPath='data/fcn32s-synthText/net-epoch-100.mat';
%direction = '/home/fcn/dena/COCO_Text_VAL/input'; outputString='/home/fcn/dena/COCO_Text_VAL/hm_ICDAR_FCN_400_epoch/%s.png'; netPath= '/home/dena/Projects/FCN-TextDetection2/data/fcn32s-icdar/net-epoch-400.mat' ;
direction = '/home/dena/datasets/ICDAR/ICDAR_VAL' ; outputString= '/home/dena/Projects/DigitDetection/BBX-Rectangle/fig/%s.jpg'   ; netPath= '/home/dena/Projects/FCN-SynthText/data/fcn32s-synthText/net-epoch-100.mat' ;

%read just the 500 images at the test level and find the blob over them
% -------------------------------------------------------------------------
%run matconvnet/matlab/vl_setupnn ;
run /home/dena/Software/matconvnet-1.0-beta17/matlab/vl_setupnn;
% -------------------------------------------------------------------------

% -------------------------------------------------------------------------
% experiment and data paths
% -------------------------------------------------------------------------
%opts.expDir = 'data/fcn32s-icdar' ;
opts.modelPath = netPath; %ANGUELOS 'data/fcn32s-synthText/net-epoch-100.mat';
opts.modelFamily = 'matconvnet' ;
[opts, varargin] = vl_argparse(opts, varargin) ;
% -------------------------------------------------------------------------
% experiment setup
% -------------------------------------------------------------------------
%opts.imdbPath = fullfile(opts.expDir, 'imdb.mat') ;

opts.gpus = [1] ;
opts = vl_argparse(opts, varargin) ;

%resPath = fullfile(opts.expDir, 'results.mat') ;
%if exist(resPath)
%  info = load(resPath) ;
%  return ;
%end

if ~isempty(opts.gpus)
  gpuDevice(opts.gpus(1))
end

% -------------------------------------------------------------------------
% Setup data
% -------------------------------------------------------------------------

% D = dir(['/home/dena/Projects/FCN-TextDetection/data/ch4_test_images','/*.jpg']);
% Num = length(D(not([D.isdir])))
% imdb = icdarSetup;
% load('imdb.mat')

% ------------------------------------------------------------------------- 
% Get validation subset
% -------------------------------------------------------------------------
%val = find(imdb.images.set == 2 & imdb.images.segmentation) ;


% -------------------------------------------------------------------------
% Setup model
% -------------------------------------------------------------------------

opts.modelFamily = 'matconvnet' ;

switch opts.modelFamily
  case 'matconvnet'
    opts.modelPath = netPath; % 'data/fcn32s-synthText/net-epoch-100.mat' ;
    net = load(opts.modelPath) ;
    net = dagnn.DagNN.loadobj(net.net) ;
    net.mode = 'test' ;
    for name = {'objective', 'accuracy'}
      net.removeLayer(name) ;
    end
    net.meta.normalization.averageImage = reshape(net.meta.normalization.rgbMean,1,1,3) ;
    predVar = net.getVarIndex('prediction') ;
    inputVar = 'input' ;
    imageNeedsToBeMultiple = true ;

  case 'ModelZoo'
    net = dagnn.DagNN.loadobj(load(opts.modelPath)) ;
    net.mode = 'test' ;
    predVar = net.getVarIndex('upscore') ;
    inputVar = 'data' ;
    imageNeedsToBeMultiple = false ;

  case 'TVG'
    net = dagnn.DagNN.loadobj(load(opts.modelPath)) ;
    net.mode = 'test' ;
    predVar = net.getVarIndex('coarse') ;
    inputVar = 'data' ;
    imageNeedsToBeMultiple = false ;
end
opts.gpus = [1] ;
if ~isempty(opts.gpus)
  gpuDevice(opts.gpus(1)) ;
  net.move('gpu') ;
end
net.mode = 'test' ;

% -------------------------------------------------------------------------
% Train
% -------------------------------------------------------------------------

numGpus = 0 ;

confusion = zeros(2) ;


%direction = '/home/dena/Projects/FCN-TextDetection/data/ch4_test_images';
%direction = '/home/dena/MEGAsyncDownloads/coco-text-legible-val/val2014-onlyLegibleText';
DD = dir([direction,'/*.jpg']);
NumImgs = length(DD(not([DD.isdir])));

%for ImgNum = 380:380
for ImgNum = 1:NumImgs
%for ImgNum = 1:500
    ImgNameFrmt = DD(ImgNum).name;
    imnm = strsplit(ImgNameFrmt, '.');
    ImgName = imnm{1,1};
    %ImgName = 'img_853'; %'img_804';%'img_888';%'img_956';%'img_802';%'img_873';%'img_921';%'img_956'; %'img_953';   %'img_993'; % 'COCO_train2014_000000072453';
    rgbPath = sprintf('%s/%s.jpg',direction,ImgName);%A
    %rgbPath = sprintf('/home/dena/MEGAsyncDownloads/coco-text-legible-val/val2014-onlyLegibleText/%s.jpg',ImgName);
    inpImg = imread (rgbPath);
    %imshow (inpImg)
  % Load an image and gt segmentation
    rgb = vl_imreadjpeg({rgbPath}) ;
    rgb = rgb{1} ;

%for i = 1:numel(val)
%for i = 51:51
%for i = 13
%   imId = val(i) ;
%   name = imdb.images.name{imId} ;
%   rgbPath = sprintf(imdb.paths.image, name) ;
%   labelsPath = sprintf(imdb.paths.classSegmentation, name) ;

  % Load an image and gt segmentation
%   rgb = vl_imreadjpeg({rgbPath}) ;
%   rgb = rgb{1} ;
%   anno = imread(labelsPath) ;
%   lb = single(anno) ;
%   lb = mod(lb + 1, 256) ; % 0 = ignore, 1 = bkg

  % Subtract the mean (color)
    im = bsxfun(@minus, single(rgb), net.meta.normalization.averageImage) ;

  % Some networks requires the image to be a multiple of 32 pixels
  if imageNeedsToBeMultiple
    sz = [size(im,1), size(im,2)] ;
    sz_ = round(sz / 32)*32 ;
    im_ = imresize(im, sz_) ;
  else
    im_ = im ;
  end

  if ~isempty(opts.gpus)
    im_ = gpuArray(im_) ;
  end

    net.eval({inputVar, im_}) ;
    scores_ = gather(net.vars(predVar).value);
     
    [~,pred_] = max(scores_,[],3) ;  
    if imageNeedsToBeMultiple        
        pred = imresize(pred_, sz, 'method', 'nearest') ;
    else
        pred = pred_ ;
    end
    
% ************** Clustering *********************************
[r, c] = size(pred);
I = zeros(r,c);
Ncluster = 0 ;
for i = 1:r
    for j = 1:c
        if pred(i,j) ~= 1
            if I(i,j) == 0%all the previous indices               
                
                %if ( (pred(i-1, j) == pred(i,j)) || (pred(i+1,j)== pred(i,j)) || (pred(i,j+1)== pred(i,j)) || (pred (i,j-1)== pred(i,j)) ) && ()
                if (j~=1) && (I(i,j-1) ~=0) && (pred(i,j-1) == pred(i,j)) 
                    I(i,j) = I(i,j-1) ;
                elseif (i~=1) && (I(i-1,j) ~=0) && (pred(i-1,j) == pred(i,j))
                     I (i,j) = I (i-1,j);
                elseif (i~=1) && (j~=1) && (I(i-1,j-1) ~=0) && (pred(i-1,j-1) == pred(i,j))
                     I (i,j) = I(i-1, j-1);
                elseif (i~=1) && (j~=c) && (I(i-1,j+1) ~=0) && (pred(i-1,j+1) == pred(i,j))
                     I (i,j) = I(i-1, j+1);
                elseif (i~=r) && (j~=1) && (I(i+1,j-1) ~=0) && (pred(i+1,j-1) == pred(i,j)) 
                     I(i,j) = I(i+1,j-1) ;
                elseif (i~=r) && (j~=c) && (I(i+1,j+1) ~=0) && (pred(i+1,j+1) == pred(i,j)) 
                     I(i,j) = I(i+1,j+1) ;
                elseif (j~=c) && (I(i,j+1) ~=0) && (pred(i,j+1) == pred(i,j)) 
                    I(i,j) = I(i,j+1) ;
                elseif (i~=r) && (I(i+1,j) ~=0) && (pred(i+1,j) == pred(i,j))
                     I (i,j) = I (i+1,j);
                
                
                else %if there is no neighbor equal to it which was definded as another cluster before
                Ncluster = Ncluster+1;
                I(i,j) = Ncluster;
                
                %check all the neighbors horizontally (i, j+1)
                jj = j;
                while (jj <= c) && (pred(i,jj) == pred(i,j)) && ((I(i,jj) == 0) || (I(i,jj) == Ncluster))           
                      I(i,jj) = Ncluster;
                      
                      %Diagonal up right i-1, j+1                
                      iii = i;
                      jjj = jj;
                      while (jjj <= c) && (iii>=1) && (pred(iii,jjj) == pred(i,j)) && ((I(iii,jjj) == 0) || (I(iii,jjj) == Ncluster))
                      I(iii,jjj) = Ncluster;
                      iii = iii-1;
                      jjj = jjj+1;
                      end  
                      %Diagonal up left i-1, j-1
                      iii = i;
                      jjj = jj;
                      while (jjj >= 1) && (iii>=1) && (pred(iii,jjj) == pred(i,j)) && ((I(iii,jjj) == 0) || (I(iii,jjj) == Ncluster))
                      I(iii,jjj) = Ncluster;
                      iii = iii-1;
                      jjj = jjj-1;
                      end 
                      %Diagonal down right i+1, j+1
                      iii = i;
                      jjj = jj;
                      while (jjj <= c) && (iii <= r) && (pred(iii,jjj) == pred(i,j)) && ((I(iii,jjj) == 0) || (I(iii,jjj) == Ncluster))
                      I(iii,jjj) = Ncluster;
                      iii = iii+1;
                      jjj = jjj+1;
                      end
                      %Diagonal down left i+1 , j-1
                      iii = i;
                      jjj = jj;
                      while (jjj >= 1) && (iii <= r) && (pred(iii,jjj) == pred(i,j)) && ((I(iii,jjj) == 0) || (I(iii,jjj) == Ncluster))
                      I(iii,jjj) = Ncluster;
                      iii = iii+1;
                      jjj = jjj-1;
                      end
                      %Vertical up i-1, j
                      iii = i;
                      jjj = jj;
                      while (iii >= 1) && (pred(iii,jjj) == pred(i,j)) && ((I(iii,jjj) == 0) || (I(iii,jjj) == Ncluster))
                      I(iii,jjj) = Ncluster;
                      iii = iii-1;
                      end
                      %vertical down i+1, j
                      iii = i;
                      jjj = jj;
                      while (iii <= r) && (pred(iii,jjj) == pred(i,j)) && ((I(iii,jjj) == 0) || (I(iii,jjj) == Ncluster))
                      I(iii,jjj) = Ncluster;
                      iii = iii+1;
                      end
                      
                      jj = jj+1;
                end

                %check all the neighbors vertically (j, i+1)
                ii = i;
                while (ii <= r) && (pred(ii,j) == pred(i,j)) && ((I(ii,j) == 0) || (I(ii,j) == Ncluster))          
                      I(ii,j) = Ncluster;
                      
                      
                      %Diagonal up right i-1, j+1                
                      iii = ii;
                      jjj = j;
                      while (jjj <= c) && (iii>=1) && (pred(iii,jjj) == pred(i,j)) && ((I(iii,jjj) == 0) || (I(iii,jjj) == Ncluster))
                      I(iii,jjj) = Ncluster;
                      iii = iii-1;
                      jjj = jjj+1;
                      end  
                      %Diagonal up left i-1, j-1
                      iii = ii;
                      jjj = j;
                      while (jjj >= 1) && (iii>=1) && (pred(iii,jjj) == pred(i,j)) && ((I(iii,jjj) == 0) || (I(iii,jjj) == Ncluster))
                      I(iii,jjj) = Ncluster;
                      iii = iii-1;
                      jjj = jjj-1;
                      end 
                      %Diagonal down right i+1, j+1
                      iii = ii;
                      jjj = j;
                      while (jjj <= c) && (iii <= r) && (pred(iii,jjj) == pred(i,j)) && ((I(iii,jjj) == 0) || (I(iii,jjj) == Ncluster))
                      I(iii,jjj) = Ncluster;
                      iii = iii+1;
                      jjj = jjj+1;
                      end
                      %Diagonal down left i+1 , j-1
                      iii = ii;
                      jjj = j;
                      while (jjj >= 1) && (iii <= r) && (pred(iii,jjj) == pred(i,j)) && ((I(iii,jjj) == 0) || (I(iii,jjj) == Ncluster))
                      I(iii,jjj) = Ncluster;
                      iii = iii+1;
                      jjj = jjj-1;
                      end
                      %Horizontal right i, j+1
                      iii = ii;
                      jjj = j;
                      while (jjj <= c) && (pred(iii,jjj) == pred(i,j)) && ((I(iii,jjj) == 0) || (I(iii,jjj) == Ncluster))
                      I(iii,jjj) = Ncluster;
                      jjj = jjj+1;
                      end
                      %Horizontal left i, j-1
                      iii = ii;
                      jjj = j;
                      while (jjj >= 1) && (pred(iii,jjj) == pred(i,j)) && ((I(iii,jjj) == 0) || (I(iii,jjj) == Ncluster))
                      I(iii,jjj) = Ncluster;
                      jjj = jjj-1;
                      end
 
                      
                      ii = ii+1;
                end
 
                 
                % check all the neighbors diagonally (i+1, j+1)
                ii =i; jj=j;
             
                while (ii <= r) && (jj <= c) && (pred(ii, jj) == pred(i,j)) && ((I(ii, jj) == 0) || (I(ii, jj) == Ncluster))
                     I(ii, jj) = Ncluster;
                    
                     
                     %Diagonal up right i-1, j+1                
                      iii = ii;
                      jjj = jj;
                      while (jjj <= c) && (iii>=1) && (pred(iii,jjj) == pred(i,j)) && ((I(iii,jjj) == 0) || (I(iii,jjj) == Ncluster))
                      I(iii,jjj) = Ncluster;
                      iii = iii-1;
                      jjj = jjj+1;
                      end  
                      %Diagonal down left i+1 , j-1
                      iii = ii;
                      jjj = jj;
                      while (jjj >= 1) && (iii <= r) && (pred(iii,jjj) == pred(i,j)) && ((I(iii,jjj) == 0) || (I(iii,jjj) == Ncluster))
                      I(iii,jjj) = Ncluster;
                      iii = iii+1;
                      jjj = jjj-1;
                      end
                      %Vertical up i-1, j
                      iii = ii;
                      jjj = jj;
                      while (iii >= 1) && (pred(iii,jjj) == pred(i,j)) && ((I(iii,jjj) == 0) || (I(iii,jjj) == Ncluster))
                      I(iii,jjj) = Ncluster;
                      iii = iii-1;
                      end
                      %vertical down i+1, j
                      iii = ii;
                      jjj = jj;
                      while (iii <= r) && (pred(iii,jjj) == pred(i,j)) && ((I(iii,jjj) == 0) || (I(iii,jjj) == Ncluster))
                      I(iii,jjj) = Ncluster;
                      iii = iii+1;
                      end 
                      %Horizontal right i, j+1
                      iii = ii;
                      jjj = jj;
                      while (jjj <= c) && (pred(iii,jjj) == pred(i,j)) && ((I(iii,jjj) == 0) || (I(iii,jjj) == Ncluster))
                      I(iii,jjj) = Ncluster;
                      jjj = jjj+1;
                      end
                      %Horizontal left i, j-1
                      iii = ii;
                      jjj = jj;
                      while (jjj >= 1) && (pred(iii,jjj) == pred(i,j)) && ((I(iii,jjj) == 0) || (I(iii,jjj) == Ncluster))
                      I(iii,jjj) = Ncluster;
                      jjj = jjj-1;
                      end
                      
                     
                     ii = ii+1;
                     jj = jj+1;
                end
                                 
                
                % check all the neighbors Rediagonally (i+1, j-1)
                ii =i; jj=j;
                
                while (ii <= r) && (jj >= 1) && (pred(ii, jj) == pred(i,j)) && ((I(ii, jj) == 0) || (I(ii, jj) == Ncluster)) 
                      I(ii, jj) = Ncluster; 

                      %Diagonal up left i-1, j-1
                      iii = ii;
                      jjj = jj;
                      while (jjj >= 1) && (iii>=1) && (pred(iii,jjj) == pred(i,j)) && ((I(iii,jjj) == 0) || (I(iii,jjj) == Ncluster))
                      I(iii,jjj) = Ncluster;
                      iii = iii-1;
                      jjj = jjj-1;
                      end
                      %Diagonal down right i+1, j+1
                      iii = ii;
                      jjj = jj;
                      while (jjj <= c) && (iii <= r) && (pred(iii,jjj) == pred(i,j)) && ((I(iii,jjj) == 0) || (I(iii,jjj) == Ncluster))
                      I(iii,jjj) = Ncluster;
                      iii = iii+1;
                      jjj = jjj+1;
                      end
                     %Vertical up i-1, j
                      iii = ii;
                      jjj = jj;
                      while (iii >= 1) && (pred(iii,jjj) == pred(i,j)) && ((I(iii,jjj) == 0) || (I(iii,jjj) == Ncluster))
                      I(iii,jjj) = Ncluster;
                      iii = iii-1;
                      end
                      %vertical down i+1, j
                      iii = ii;
                      jjj = jj;
                      while (iii <= r) && (pred(iii,jjj) == pred(i,j)) && ((I(iii,jjj) == 0) || (I(iii,jjj) == Ncluster))
                      I(iii,jjj) = Ncluster;
                      iii = iii+1;
                      end 
                      %Horizontal right i, j+1
                      iii = ii;
                      jjj = jj;
                      while (jjj <= c) && (pred(iii,jjj) == pred(i,j)) && ((I(iii,jjj) == 0) || (I(iii,jjj) == Ncluster))
                      I(iii,jjj) = Ncluster;
                      jjj = jjj+1;
                      end
                      %Horizontal left i, j-1
                      iii = ii;
                      jjj = jj;
                      while (jjj >= 1) && (pred(iii,jjj) == pred(i,j)) && ((I(iii,jjj) == 0) || (I(iii,jjj) == Ncluster))
                      I(iii,jjj) = Ncluster;
                      jjj = jjj-1;
                      end
                 
                     ii = ii+1;
                     jj = jj-1;
                end
               
                
                end %if one of the neighbors is equal to sample point    
            end %if I is not equal to one
        end % if pred is not equal to one
    end % for all the columns
end % for all the rows

%************************** Rectangle **************************

	fh = figure(100);
	%imshow( im/255 , 'border', 'tight' ); %//show your image
    imshow( inpImg , 'border', 'tight' ); %//show your image
	hold on

 for id = 1:Ncluster
    hold on
	%[rr, cc] = find(pred ==id);
    [rr, cc] = find(I ==id);
    if isempty(rr) == 0 && isempty(cc) == 0
	minR = min(rr);
	minC = min(cc);
	maxR = max(rr);
	maxC = max(cc); 
  %img = imread('/home/dena/datasets/SVHN/train/30111.jpg');

   rectangle('Position', [minC, minR, (maxC-minC) ,(maxR-minR)], 'LineWidth',4, 'EdgeColor','g'); %// draw rectangle on image
	%rectangle('Position', [maxC, minR, (maxC-minC) ,(maxR-minR)], 'LineWidth',4, 'EdgeColor','g'); %// draw rectangle on image
        %rectangle('Position', [minR, minC, (maxR-minR), (maxC-minC)], 'LineWidth',4, 'EdgeColor','g'); %// draw rectangle on image
    
%     if id == 11
% 	text(maxC, minR-5, '0' , 'Color','g' , 'FontSize', 28); %//Write a name for rectangle
%     else
%     RactName = sprintf ('%d' , (pred(minR,minC)-1));
% 	text(maxC, minR-5, RactName , 'Color','g' , 'FontSize', 28); %//Write a name for rectangle
%     end
         
   end
   frm = getframe( fh ); %// get the image+rectangle
   %ImgName = sprintf('/home/dena/Projects/DigitDetection/FCN/data/fcn32s-digits/Figs/Rectangle/test/%d.png', i+30000 );
   ImgRectName = sprintf(outputString,ImgName);
   imwrite( frm.cdata, ImgRectName ); %// save to file

        
 end



end
    

    
    
    
    
