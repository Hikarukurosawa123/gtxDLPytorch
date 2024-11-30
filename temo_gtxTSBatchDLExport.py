function gtxTSBatchDLExport(folderDL,kDL,F,OP,DepthMapAll,QF,RE,exportFolder,nExportedMAT)
% Export data for DL
% Michael Daly
% GTx Program (UHN-TECHNA)
% May 2022

%--------------------------- Settings for Export -------------------------%
% Export folder for deep learning training/testing
dataPath = gtxDataPath;

if strcmp(folderDL,'ResizeForDL')
    fluorforwardPath = strcat(dataPath, 'SL\SFDIData', filesep);
elseif strcmp(folderDL,'ResizeForDLRemoveBckg')
    fluorforwardPath = strcat(dataPath, 'SL\SFDIData', filesep);
else
    fluorforwardPath = strcat(dataPath, 'DL', filesep);
end

if ~isempty(exportFolder)
    % Append exportFolder if provided
    fluorforwardPath = [fluorforwardPath,exportFolder,filesep];
end
    
% Append subfolder (e.g., trainingData, testData)
fluorforwardPath = [fluorforwardPath,folderDL,filesep];

% Create folder if not present
%if ~exist(fluorforwardPath, 'dir')
%    mkdir(fluorforwardPath)
%end

% Basename for exported file
basenameExport = [fluorforwardPath];

%-------------------------- Export Data ----------------------------------%

% Convert from double to single precision
% Read as float32 in Python, so save memory on exported .mat files
F = single(F);
RE = single(RE);
OP = single(OP);
QF = single(QF);

if length(DepthMapAll) == 1
DF = single(DepthMapAll{1});
else
DF_ice = single(DepthMapAll{1});
DF_sub = single(DepthMapAll{2});
end

% Get number of data sets to export
nSets = length(kDL); 

if strcmp(nExportedMAT,'Multiple')
    for ss = 1:nSets
        % Append deep learning iteration number (kDL)
        filenameExport = [basenameExport,'_Images',num2str(kDL(ss),'%0.5d')]; % e.g., _Images00001
        % Export .mat for EACH data set
        % exportMAT(filenameExport,F(ss,:,:,:),OP(ss,:,:,:),DF(ss,:,:),QF(ss,:,:));
        FF = squeeze(F(ss,:,:,:));
        REE = squeeze(RE(ss,:,:,:));
        OPP = squeeze(OP(ss,:,:,:));
        DFF = squeeze(DF(ss,:,:));
        QFF = squeeze(QF(ss,:,:));
        exportMAT(filenameExport,FF,OPP,REE,DFF,QFF, folderDL);
    end   
elseif strcmp(nExportedMAT,'Single')
    % Append total number of DL data sets
    filenameExport = [basenameExport,'_nImages',num2str(nSets,'%d')]; 
    if ~exist(basenameExport, 'dir')
       mkdir(basenameExport)
    end
    % Export .mat containing ALL data sets
<<<<<<< Updated upstream
    exportMAT(filenameExport,F,OP,RE,DF,QF, folderDL);
end

% Local function to export data
function exportMAT(filenameExport,F,OP,RE,DF,QF, folderDL)
=======
    exportMAT(filenameExport,F,OP,RE,DepthMapAll,QF)

  
end

% Local function to export data
function exportMAT(filenameExport,F,OP,RE,DepthMapAll,QF)
>>>>>>> Stashed changes

% Export as format -v7.3 for python reading
if contains(filenameExport,'StefCourseProject')
    % Don't export OP & QF
    save(filenameExport,'F','DF','-v7.3');
elseif contains(filenameExport,'ScottCourseProject')
    % Don't export DF
    save(filenameExport,'F','OP','QF','-v7.3');
% elseif contains(filenameExport,'Refl')
%     % Don't export OP
%     save(filenameExport,'F','RE','DF','QF','-v7.3')
elseif contains(folderDL, 'Ronly')
    % Don't export F or QF
    save(filenameExport, 'OP', 'RE', 'DF', '-v7.3');
else
    if length(DepthMapAll) == 1
        DF = DepthMapAll{1};
    else 
        DF_ice = DepthMapAll{1}; % DepthMaps across fx all the same
        DF_sub = DepthMapAll{2}; % DepthMaps across fx all the same
    end 
    % Export all
    if length(DepthMapAll) == 1
        
        save(filenameExport,'F','OP','RE','DF','QF','-v7.3');
    else
    save(filenameExport,'F','OP','RE','DF_sub','DF_ice', 'QF','-v7.3');
    end 
end