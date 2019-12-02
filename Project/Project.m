clf

glassesData = readtable('glasses.csv');
report = readtable('report.csv');
% 
div = 100;
% 
reduction = ceil(height(glassesData(:, 1))/div);
arr = zeros(reduction, 9);
 
for i = 1:div:height(glassesData(:, 1))
    formatIn = 'yyyy/mm/dd HH:MM:SS.FFF';
    arr(ceil(i/div), 1) = datenum(glassesData{i, 2},formatIn);
    arr(ceil(i/div), 2) = glassesData{i, 3};
    arr(ceil(i/div), 3) = glassesData{i, 4};
    arr(ceil(i/div), 4) = glassesData{i, 5};
    arr(ceil(i/div), 5) = glassesData{i, 6};
    arr(ceil(i/div), 6) = glassesData{i, 7};
    arr(ceil(i/div), 7) = glassesData{i, 8};
     
    arr(ceil(i/div), 9) = sqrt(glassesData{i, 3}^2 + glassesData{i, 4}^2 + glassesData{i, 5}^2);
end
%}
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
        arr(filter, 8) = 5;
    end
    
end

figure()
x = 1:length(arr);
hold on
scatter(x(arr(:, 8) == 4), arr(arr(:, 8) == 4, 9), 'black', '.')
scatter(x(arr(:, 8) == 5), arr(arr(:, 8) == 5, 9), 'yellow', '.')
scatter(x(arr(:, 8) == 3), arr(arr(:, 8) == 3, 9), 'green', '.')
scatter(x(arr(:, 8) == 2), arr(arr(:, 8) == 2, 9), 'red', '.')
scatter(x(arr(:, 8) == 1), arr(arr(:, 8) == 1, 9), 'blue', '.')

figure() % Scatter plot
x = 1:length(arr);
hold on
scatter(arr(arr(:, 8) == 2,4),  arr(arr(:, 8) == 2, 5), 'blue', '.');
scatter(arr(arr(:, 8) == 3,4), arr(arr(:, 8) == 3, 5), 'red', '.');
legend('Walk','Meeting/Computer');