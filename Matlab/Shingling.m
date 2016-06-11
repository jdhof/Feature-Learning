function FBE = Shingling(FBEs, sh, mode)
% Author: Jan-David Hof    
%

%Input: 
%   (matrix) FBEs :     matrix of log-scaled mel-spectrograms
%                       
%   (scalar) sh   :     Shingling Anzahl (Menge der konkatenierten Frames)


%Output:
%   (Matrix) FBE  :     shingled data 


% 
% mode : shingling mode
%   1  : mit Nullen auffüllen
%   2  : überflüsseige Frames weglassen
%   3  : mit Mittelwertframes auffüllen
%
    
    if sh ~=0
        [r frameNumb] = size(FBEs);
        rest = mod(frameNumb,sh); % berechnen, ob Shingling aufgeht
        if rest == 0        %Falls Shingling aufgeht
            colNumber = frameNumb/sh;   %Spaltenanzahl von neuer FBE-Matrix
            numb = 1;   % aktuelle Spalte neuer Matrix
            FBE = zeros(r*sh,colNumber);    %später neu zusammengesetzte FBE MAtrix
            Shingle = zeros(1,sh*r);    %neuer Frame durch Shingling zusammengesetzt
            l = 1;
            while l<=frameNumb %Spaltenvektoren, die geshingled werden
                k = 1;
                tempStart = 1;
                temp = r;
                while k <= sh %Shingled sh Frames
                    Shingle(1,tempStart:temp) = FBEs(:,l);  %neuer Shingle-Frame wird zusammengebaut
                    tempStart = temp +1;    
                    temp = temp + r;
                    k = k + 1;
                    l = l + 1; 
                end
                FBE(:,numb) = Shingle'; %Neue Matrix wird aus erstellten Shingles zusammengefasst
                numb = numb + 1;    %Spaltennummer der Matrix immer erhöhen, wenn sh Frames zusammengefasst wurden
            end  
        else        %Falls Frames bei Shingling übrigblleiben würden
            
            switch mode
                case 1
                    frameNumb = frameNumb + (sh - rest);   %adjusted Framequantity. fehlende Frames werden erstellt
                    fillingFrames = zeros(r, (sh - rest)); %Füllende Frames werden erstellt
                    FBEs = [FBEs fillingFrames];     %Wenn Shingle nicht aufgeht, dann mit 0 auffüllen
                case 2
                    frameNumb = frameNumb - rest;               %adjusted Framequantity. Nicht aufgehende Frames werden weggelassen
                    FBEs = FBEs(:, 1:frameNumb);
                case 3
                    fillingFrames = mean(FBEs,2);   %Mittelwert jeder Zeile berechnen == Füllende Frames werden erstellt
                    frameNumb = frameNumb + (sh-rest);   %adjusted Framequantity. fehlende Frames werden erstellt
                    for i = 1:(sh-rest)
                        FBEs = [FBEs fillingFrames];     %Mittelwert aller anderen Frames berechnen und daraus fehlende Frames bilden
                    end
            end
            
            [r frameNumb] = size(FBEs); %Werte nach veränderung aktualisieren
            colNumber = frameNumb/sh;   %Spaltenanzahl von neuer FBE-Matrix
            numb = 1;   % aktuelle Spalte neuer Matrix
            FBE = zeros(r*sh,colNumber);    %später neu zusammengesetzte FBE MAtrix
            Shingle = zeros(1,sh*r);    %neuer Frame durch Shingling zusammengesetzt
            l = 1;
            while l<=frameNumb %Spaltenvektoren, die geshingled werden
                k = 1;
                tempStart = 1;
                temp = r;
                while k <= sh %Shingled sh Frames
                    Shingle(1,tempStart:temp) = FBEs(:,l);  %neuer Shingle-Frame wird zusammengebaut
                    tempStart = temp +1;
                    temp = temp + r;
                    k = k + 1;
                    l = l + 1; 
                end
                FBE(:,numb) = Shingle'; %Neue Matrix wird aus erstellten Shingles zusammengefasst
                numb = numb + 1;    %Spaltennummer der Matrix immer erhöhen, wenn sh Frames zusammengefasst wurden
            end
        end
        
    else

    FBE = FBEs;
    
    end
end