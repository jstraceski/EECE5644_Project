clf

%% Load the data from the raw source and filter/format it in a useful way
glassesData = readtable('glasses.csv');
report = readtable('report.csv');

div = 100;
last = 1;

dSize = ceil(height(glassesData(:, 1))/div);
arr = zeros(dSize, 9);
sampleSize = 1;

shiftFactor = div / sampleSize;

for i = 1:div:(height(glassesData(:, 1)) - div)
    
    avg = zeros(1, 6);    
    for shift = 0:(sampleSize - 1)
        sTotal =  ceil(i + (shift * shiftFactor));
        
        avg(1, 1) = avg(1, 1) + glassesData{sTotal, 3} / sampleSize;
        avg(1, 2) = avg(1, 2) + glassesData{sTotal, 4} / sampleSize;
        avg(1, 3) = avg(1, 3) + glassesData{sTotal, 5} / sampleSize;
        avg(1, 4) = avg(1, 4) + glassesData{sTotal, 6} / sampleSize;
        avg(1, 5) = avg(1, 5) + glassesData{sTotal, 7} / sampleSize;
        avg(1, 6) = avg(1, 6) + glassesData{sTotal, 8} / sampleSize;
    end
    
    dIndex = ceil(i/div);
    formatIn = 'yyyy/mm/dd HH:MM:SS.FFF';
    arr(dIndex, 1) = datenum(glassesData{i, 2},formatIn);
    
    arr(dIndex, 2) = avg(1, 1);
    arr(dIndex, 3) = avg(1, 2);
    arr(dIndex, 4) = avg(1, 3);
    arr(dIndex, 5) = avg(1, 4);
    arr(dIndex, 6) = avg(1, 5);
    arr(dIndex, 7) = avg(1, 6);
    disp(dIndex)
end

arr(:, 8) = 0;

C = categorical(report{:,2});
types = categories(C);

for ti = 1:length(types)
    t = types{ti};
    for i = 1:height(report)
        if strcmp(report{i,2}, t)
            from = datenum(report{i,4});
            to = datenum(report{i,5});
            filter = (arr(:, 1) <= to & arr(:, 1) >= from);
            arr(filter, 8) = 4;
        end
    end
end

filteredData = arr(arr(:, 8) ~= 0, :);

for i = 1:height(report)
    
    if strcmp(report{i,2}, 'Walk')
        from = datenum(report{i,4});
        to = datenum(report{i,5});
        filter = (arr(:, 1) <= to & arr(:, 1) >= from);
        arr(filter, 8) = 2;
    end
    
    if strcmp(report{i,2}, 'Eat')
        from = datenum(report{i,4});
        to = datenum(report{i,5});
        filter = (arr(:, 1) <= to & arr(:, 1) >= from);
        arr(filter, 8) = 1;
    end
    
    if strcmp(report{i,2}, 'Meeting')
        from = datenum(report{i,4});
        to = datenum(report{i,5});
        filter = (arr(:, 1) <= to & arr(:, 1) >= from);
        arr(filter, 8) = 3;
    end
    
    if strcmp(report{i,2}, 'In computer')
        from = datenum(report{i,4});
        to = datenum(report{i,5});
        filter = (arr(:, 1) <= to & arr(:, 1) >= from);
        arr(filter, 8) = 3;
    end
    
    if strcmp(report{i,2}, 'In vehicle')
        from = datenum(report{i,4});
        to = datenum(report{i,5});
        filter = (arr(:, 1) <= to & arr(:, 1) >= from);
        arr(filter, 8) = 5;
    end
    
    if strcmp(report{i,2}, 'In bus')
        from = datenum(report{i,4});
        to = datenum(report{i,5});
        filter = (arr(:, 1) <= to & arr(:, 1) >= from);
        arr(filter, 8) = 5;
    end
    
    if strcmp(report{i,2}, 'Train')
        from = datenum(report{i,4});
        to = datenum(report{i,5});
        filter = (arr(:, 1) <= to & arr(:, 1) >= from);
        arr(filter, 8) = 6;
    end
    
end

figure(1) % Scatter plot
clf
x = 1:length(arr);
hold on
scatter(arr(arr(:, 8) == 2,4),  arr(arr(:, 8) == 2, 5), 'c', '.');
scatter(arr(arr(:, 8) == 3,4), arr(arr(:, 8) == 3, 5), 'r', '.');
legend('= Walking', '= Metting/Using Computer');

fArr = arr((arr(:, 8) == 2 | arr(:, 8) == 3), [4 5 8]); 