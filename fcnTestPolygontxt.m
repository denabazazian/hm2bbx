%function info = fcnTestPolygontxt(varargin)

function fcnTestPolygontxt(direction,netPath,outPutTXT,outPutImg)

%direction = '/home/fcn/dena/COCO_Text_VAL/input'; outputString='/home/fcn/dena/COCO_Text_VAL/hm_ICDAR_FCN_400_epoch/%s.png'; netPath= '/home/dena/Projects/FCN-TextDetection2/data/fcn32s-icdar/net-epoch-400.mat' ;
%direction = '/home/dena/datasets/ICDAR/ICDAR_VAL' ; outputString= '/home/dena/Projects/DigitDetection/BBX-Rectangle/fig/%s.jpg'   ; netPath= '/home/dena/Projects/FCN-SynthText/data/fcn32s-synthText/net-epoch-100.mat' ;
%direction = '/home/dena/Projects/FCN-TextDetection/data/ch4_test_images' ; outputString= '/home/dena/Projects/DigitDetection/BBX-Rectangle/fig/ICDAR-Test/%s.jpg'   ; netPath= '/home/dena/Projects/Fine-Tune-ICDAR-Synthetic/data/fcn32s-icdar/net-epoch-400.mat' ;


% -------------------------------------------------------------------------
%run matconvnet/matlab/vl_setupnn ;
run /home/dena/Software/matconvnet-1.0-beta17/matlab/vl_setupnn;
% -------------------------------------------------------------------------

% -------------------------------------------------------------------------
% experiment and data paths
% -------------------------------------------------------------------------;
opts.modelPath = netPath; %'data/fcn32s-synthText/net-epoch-100.mat';
opts.modelFamily = 'matconvnet' ;

% -------------------------------------------------------------------------
% experiment setup
% -------------------------------------------------------------------------

opts.gpus = [1] ;

if ~isempty(opts.gpus)
  gpuDevice(opts.gpus(1))
end



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
% Setup Data
% -------------------------------------------------------------------------

numGpus = 0 ;

confusion = zeros(2) ;

DD = dir([direction,'/*.jpg']);
NumImgs = length(DD(not([DD.isdir])));

for ImgNum = 1:NumImgs
%for ImgNum = 1:500
    ImgNameFrmt = DD(ImgNum).name;
    imnm = strsplit(ImgNameFrmt, '.');
    ImgName = imnm{1,1};
    rgbPath = sprintf('%s/%s.jpg',direction,ImgName);
    inpImg = imread (rgbPath);
    %imshow (inpImg)
  % Load an image and gt segmentation
    rgb = vl_imreadjpeg({rgbPath}) ;
    rgb = rgb{1} ;

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

% text file of rectangles
     %outPutTXT = /home/dena/Projects/DigitDetection/BBX-Rectangle/TextFiles/Polygon
    fileID = fopen(sprintf('%s/res_%s.txt',outPutTXT,ImgName),'w');
   
 for id = 1:Ncluster
    hold on
	                                        
    [rr, cc] = find(I ==id);
    if isempty(rr) == 0 && isempty(cc) == 0
	[minR, idxminR] = min(rr);
	[minC, idxminC] = min(cc);
	[maxR, idxmaxR] = max(rr);
	[maxC, idxmaxC] = max(cc); 
    minRC = cc(idxminR);
    minCR = rr(idxminC);
    maxRC = cc(idxmaxR);
    maxCR = rr(idxmaxC);

    if (minC ~= maxC) && (minR ~= maxR) && (minC ~= minRC ) && (minC ~= maxRC) && (minR ~= minCR) && (minR ~= maxCR)
      if maxC == c
         maxC = c-1;
      end
      if maxR == r
         maxR = r-1;
      end

      % Writing the bounding boxes as a text file
      fprintf(fileID,'%d,%d,%d,%d,%d,%d,%d,%d\r\n',minRC,minR,maxC,maxCR,maxRC,maxR,minC,minCR);
      end

                                       %rectangle('Position', [minC, minR, (maxC-minC) ,(maxR-minR)], 'LineWidth',4, 'EdgeColor','g'); %// draw rectangle on image
	                                   %rectangle('Position', [maxC, minR, (maxC-minC) ,(maxR-minR)], 'LineWidth',4, 'EdgeColor','g'); %// draw rectangle on image
                                       %rectangle('Position', [minR, minC, (maxR-minR), (maxC-minC)], 'LineWidth',4, 'EdgeColor','g'); %// draw rectangle on image

                                       
                                       %       XX = [minC maxC minRC maxRC];
                                       %       YY = [minCR maxCR minR maxR];
                                       %       XX = [maxRC maxC minC  minRC ];
                                       %       YY = [maxR maxCR minCR  minR ];
                                       
%       XX = [minRC maxC maxRC  minC  minRC ];
%       YY = [minR maxCR maxR  minCR  minR ];
%       PLT = plot(XX, YY , 'g');
%       set(PLT,{'LineWidth'},{3})
    
                                           %BW = roipoly(inpImg,XX,YY);
                                           %CombineIm = CombineIm | BW;                               
    
                                           %BW = insertShape(inpImg,'Polygon',[minC minCR maxC maxCR minRC minR maxRC maxR],'LineWidth',5);
                                           %imshow (BW)
                                           %hold on
                                           %CombineIm = CombineIm | BW;
    
                                       %     if id == 11
                                       % 	text(maxC, minR-5, '0' , 'Color','g' , 'FontSize', 28); %//Write a name for rectangle
                                       %     else
                                       %     RactName = sprintf ('%d' , (pred(minR,minC)-1));
                                       % 	text(maxC, minR-5, RactName , 'Color','g' , 'FontSize', 28); %//Write a name for rectangle
                                       %     end
  


   

   end
   %frm = getframe( fh ); %// get the image+rectangle
   %ImgPolyName = sprintf('/home/dena/Projects/DigitDetection/BBX-Rectangle/fig/Polygon/%s.png', ImgName );
   %ImgPolyName = sprintf('%s/%s.png',outPutImg, ImgName );
                                          %ImgRectName = sprintf(outputString,ImgName);
   %imwrite( frm.cdata, ImgPolyName ); %// save to file
   %ended of writing text file
        
 end
 
   fclose(fileID);
 % Writing the bounding boxes as a text file

ImgName
end
    
    
  




%   %check all the neighbors horizontally 
%                 jj = j;
%                 while (jj <= c) && (pred(i,jj) == pred(i,j)) && ((I(i,jj) == 0) || (I(i,jj) == Ncluster))           
%                       I(i,jj) = Ncluster;
%                       jj = jj+1;
%                 end
%                 Hi = i;
%                 Hj = jj;
%                 
%                 %check all the neighbors vertically
%                 ii = i;
%                 while (ii <= r) && (pred(ii,j) == pred(i,j)) && ((I(ii,j) == 0) || (I(ii,j) == Ncluster))          
%                       I(ii,j) = Ncluster;
%                       ii = ii+1;
%                 end
%                 Vi = ii;
%                 Vj = j;   
%                  
%                 % check all the neighbors diagonally
%                 iii =i; jjj=j;
%                 if i==r 
%                     m=r-1;
%                 else
%                     m=r-1;
%                 end
%                 if j==c
%                    n=c-1;
%                 else
%                    n=c-1;
%                 end
%              
%                 while (iii <= m) && (jjj<=n) && (pred(iii+1, jjj+1) == pred(i,j)) && ((I(iii+1, jjj+1) == 0) || (I(iii+1, jjj+1) == Ncluster))
%                      I(iii+1, jjj+1) = Ncluster; 
%                      iii = iii+1;
%                      jjj = jjj+1;
%                 end
%                 Di = iii;
%                 Dj = jjj;                   
%                                        
%                 % check all the neighbors Rediagonally
%                 iiii =i; jjjj=j;
%                 if i==r 
%                    m=r-1;
%                 else
%                    m=r-1;
%                 end
%                 if j == 1
%                    n = 2;
%                 else
%                    n = 2;
%                 end
%                 
%                 while  (iiii <= m) && (jjjj>=n) && (pred(iiii+1, jjjj-1) == pred(i,j)) && ((I(iiii+1, jjjj-1) == 0) || (I(iiii+1, jjjj-1) == Ncluster)) 
%                      I(iiii+1, jjjj-1) = Ncluster; 
%                      iiii = iiii+1;
%                      jjjj = jjjj-1;
%                 end
%                 RDi = iiii;
%                 rDj = jjjj;
    
    
    
    
