clf

%% load in filtered data
trainG = load('training_GYRO_X.mat');
trainA = load('training_ACC_Z.mat');
testingG = load('testing_GYRO_X.mat');
testingA = load('testing_ACC_Z.mat');

%% format filtered data to work with the branching tree
trainT = [trainA.training_ACC_Z(:, 1) trainG.training_GYRO_X(:, :)]; 
testT = [testingA.testing_ACC_Z(:, 1) testingG.testing_GYRO_X(:, :)];

%% List of tests to try
tests = {@(x,y,f)y > f + x, @(x,y,f)y > f - x, @(x,y,f)x > f, @(x,y,f)y > f};

%% Run the program with classes 1 and 2 
b = withAdaBoost(trainT, testT, [1 2], tests);

%% Function Definitions %%

%% Main Function
function roots = withAdaBoost(trainT, testT, classes, tests)
    iter = 7; % iterations of adaboost
    T = trainT; % set the training group to the training dataset
    
    % store the tree roots, error values, and weights of the adaboost test
    roots = cell(iter, 1);
    errs = zeros(iter, 1);
    ams = zeros(iter, 1);
    % set all the weights of the observations to 1/N
    obvs = ones(length(T(:, 1)), 1)/length(T(:, 1));

    % loop through 7 iterations of adaboost
    for n = 1:iter
        % generate a weighted sample of the dataset with the same size as
        % the total data set
        [samp, idxs] = datasample(T, length(T(:, 1)), 'Weights', obvs);
        
        % generate a branching tree of the data sample
        root = genTree(classes, samp, tests);
        
        % generate class labels for the data using the data tree
        c = classify(root, samp);
        
        % create an array of zeros to calculate the weighted error of the
        % current tree
        errs2 = zeros(length(idxs), 1);
        wts = zeros(length(idxs), 1);
        
        % loop through each datapoint of the sample
        for i = 1:length(idxs)
            idx = idxs(i); % get the index of the sample in the master list
            % if the sample is not classified correctly add it to the total
            % error
            errs2(i) = (c(i) ~= samp(i, 3)) * obvs(idx);
            % store the total weight of the dataset
            wts(i) = obvs(idx);
        end
        
        % calculate the weighted error
        err = sum(errs2)/sum(wts);
        % intialize the tree weigth to be the learning coeficent
        am = 0.01;
        if (err ~= 0)
            % if the error of the tree isnt zero calculate the actual
            % weight
            am = am * log((1 - err)/err);
        
            % adjust the weights of the datapoints 
            for i = 1:length(idxs)
                idx = idxs(i);
                obvs(idx) = obvs(idx) * exp(am * (c(i) ~= samp(i, 3)));
            end
        end
        
        % normalize the weights
        for i = 1:length(obvs)
            obvs(i) = obvs(i)/sum(obvs);
        end
        
        % set the weight data
        ams(n) = am;
        roots{n} = root;
        errs(n) = err;
    end
    
    labels = {'Walking', 'Metting/Using Computer', 'Errors', 'Boundary'};
    colors = {'b', 'r', 'k'};
    
    figure(4);
    clf
    hold on
    grid on
    axis square
    % generate a classification from the tree group
    c = classifyCompound(trainT, classes, roots, ams);
    % display the data
    displayData(trainT, roots, ams, c, classes, labels, colors, ...
        'Descision Tree on Training Data');
    xlabel('Acceleration-Z')
    ylabel('Gyroscope-X')
    
    figure(5);
    clf
    hold on
    grid on
    axis square
    c = classifyCompound(testT, classes, roots, ams);
    displayData(testT, roots, ams, c, classes, labels, colors, ... 
        'Descision Tree on Testing Data');
    xlabel('Acceleration-Z')
    ylabel('Gyroscope-X')
    
    % print out a confusion matrix of the testing data
    disp(genConf(classes, testT, c));
end

%% Deptrecated tree generation without adaboost
function o = noAdaBoost(trainT, testT, classes, tests)
%     iter = 5;
%     roots = cell(iter, 1);
%     errs = zeros(iter, 1);
%     
%     for n = 1:iter
%         [samp, idxs] = datasample(trainT, length(trainT(:, 1)));
%         root = genTree(classes, samp, tests);
%         
%         c = classify(root, samp);
%         err = sum(c(:) ~= samp(:, 3))/length(idxs);
%         
%         roots{n} = root;
%         errs(n) = err;
%     end
%     
%     [~, idx] = min(errs);
%     o = roots{idx};

    o = genTree(classes, trainT, tests);
    
    figure(1);
    clf
    hold on
    drawTree(o);
    
    figure(2);
    clf
    hold on
    grid on
    axis square

    c = classify(o, trainT);
    displayData(trainT, o, {}, c, classes, {'= Walking', '= Metting/Using Computer'}, ...
        {'c', 'r', 'k'}, 'Descision Tree on Training Data');
    
    figure(3);
    clf
    hold on
    grid on
    axis square
    c = classify(o, testT);
    displayData(testT, o, {}, c, classes, {'= Walking', '= Metting/Using Computer'}, ...
        {'c', 'r', 'k'}, 'Descision Tree on Testing Data');
end

%% Display the tree data
% Displays the classified data and overalys a shaded tree boundary
function displayData(T, rTree, ams, c, classes, labels, colors, tStr)
    L = T(:, 3);
    div = 400;

    c1 = classes(1);
    c2 = classes(2);
    
    neg = T(L == c1, 1:2);
    pos = T(L == c2, 1:2);
    err = T(c ~= L, 1:2);

    scatter(pos(:, 1), pos(:, 2), colors{1}, 'x');
    scatter(neg(:, 1), neg(:, 2), colors{2}, '.');
    scatter(err(:, 1), err(:, 2), colors{3}, 'o');

    sx = min(T(:, 1));
    lx = max(T(:, 1));
    sy = min(T(:, 2));
    ly = max(T(:, 2));
    
    dx = (lx - sx)/div;
    dy = (ly - sy)/div;
    
    [xvals, yvals] = meshgrid(sx:dx:lx, sy:dy:ly);
    zvals = (xvals * 0) + 1;
    
    if (~iscell(rTree))
        zvals = zvals.*drawBounds(rTree, xvals, yvals, zvals, c2);
        zvals(zvals == 1) = NaN;
    else
        zvalsA = (zvals * 0);
        zvalsB = (zvals * 0);
        
        for n = 1:length(ams)
            d = classifyGrid(rTree{n}, xvals, yvals);
            zvalsA = zvalsA + (d == c2) * ams(n);
            zvalsB = zvalsB + (d == c1) * ams(n);
        end
        
        zvals(zvalsA >= zvalsB) = NaN;
        zvals(zvalsA < zvalsB) = 1;
    end
    
    surf(xvals, yvals, zvals, 'FaceColor', 'c', 'EdgeColor', 'c', 'FaceAlpha', 0.4,'EdgeAlpha', 0.4)
    view(0,90)
    legend(labels);
    errorP = length(err(:, 1))/length(T(:, 1));
    title(sprintf('%s\nError Rate: %.2f%%', tStr, errorP * 100))
end

%% Classify data using a weighted group of trees
function c = classifyCompound(T, classes, roots, ams)
    cl1 = classes(1);
    cl2 = classes(2);
    
    c = zeros(length(T), 1);
    c1 = zeros(length(T), 1);
    c2 = zeros(length(T), 1);
    
    for n = 1:length(roots)
        d = classify(roots{n}, T);
        c1 = c1 + (d == cl2) * ams(n);
        c2 = c2 + (d == cl1) * ams(n);
    end
    
    c(c1 >= c2) = cl2;
    c(c1 < c2) = cl1;
end

%% Generate a confusion matrix
function conf = genConf(classes, data, c)
    L = data(:, 3);
    conf = zeros(2);

    for aL = 1:2
        for cL = 1:2
            aC = classes(aL);
            cC = classes(cL);

            conf(cL, aL) = length(data(c == aC & L == cC, 1));
        end
    end
end

%% Draw a tree with all descision branches
function drawTree(root)
    syms x y f
    [nodes, labels] = gNodes(root, 1);

    nodes = [0, nodes];
    labels = [sprintf('%s, f=%f', ...
        root.test(x, y, f), root.value), labels];
    
    treeplot(nodes);
    [x, y] = treelayout(nodes);

    for i=1:length(x)
        text(x(i),y(i), labels{i})
    end
end

%% Draw the bounds of a descision matrix
function fOut = drawBounds(tree, X, Y, F, c)
    if (~isempty(tree.left))
        o = (X * 0) + 1;
        fLeft = F & tree.test(X, Y, o * tree.value);
        fRight = F & ~tree.test(X, Y, o * tree.value);
        
        fLeftO = drawBounds(tree.left, X, Y, fLeft, c);
        fRightO = drawBounds(tree.right, X, Y, fRight, c);
        
        fOut = fLeftO | fRightO;
    elseif c == tree.class
        fOut = F;
    else
        fOut = F * 0;
    end
end

%% Classify data using a single tree
function o = classify(tree, data)
    o = ones(length(data(:, 1)), 1);
    if (~isempty(tree.left))
        d = tree.test(data(:, 1), data(:, 2), o * tree.value);
        o(d == 1) = classify(tree.left, data(d == 1, :));
        o(d == 0) = classify(tree.right, data(d == 0, :));
    else 
        o = o * tree.class;
    end
end

%% Classify a grid of values given a descision tree
function o = classifyGrid(tree, X, Y)
    o = X * 0 + 1;
    
    if (~isempty(tree.left))
        d = tree.test(X, Y, o * tree.value);
        o(d == 1) = classifyGrid(tree.left, X(d == 1), Y(d == 1));
        o(d == 0) = classifyGrid(tree.right, X(d == 0), Y(d == 0));
    else 
        o = o * tree.class;
    end
end

%% Generated a list of the nodes in a tree formated for matlab to draw them
function [nodes, labels] = gNodes(tree, parent)
    syms x y f
    
    nodes = [];
    labels = {};
    
    if (~isempty(tree.left))
        nodes = [nodes, parent];
        [nodesA, labelA] = gNodes(tree.left, parent + 1);
        nodes = [nodes, nodesA];
        
        if (~isempty(tree.left.test))
            labels = [labels sprintf('%s, f=%f', ...
                tree.left.test(x, y, f), tree.left.value) labelA];
        else
            labels = [labels sprintf('%d', tree.left.class) labelA];
        end
        
        
        nodes = [nodes, parent];
        shift = length(nodes) - 1;
        
        [nodesB, labelB] = gNodes(tree.right, parent + 1);
        nodesB = nodesB + shift;
        nodes = [nodes, nodesB];
        
        if (~isempty(tree.right.test))
            labels = [labels sprintf('%s, f=%f', ...
                tree.right.test(x, y, f), tree.right.value) labelB];
        else
            labels = [labels sprintf('%d', tree.right.class) labelB];
        end
    end 
end

%% Generate a descision tree from a set of data, classes, and tests
function [tOut] = genTree(classes, data, tests)
    root = tree;
    impurityVal = 1;
    nodes = 1;
    leaf = root;
    leaf.classes = classes;
    leaf.data = data;
    
    sVal = min(data(:, 1:2), [], [1 2]);
    eVal = max(data(:, 1:2), [], [1 2]);
    
    while nodes < 15 && impurityVal > 0.01
        e = entropy(classes, data);
        
        [oData, fData] = gain(e, sVal, eVal, classes, data, tests);
        [mval, idx] = max(oData, [], [1,2], 'linear');
        [vIdx, tIdx] = ind2sub(size(oData), idx);

        cTest = tests{tIdx};
        cVal = fData(vIdx, tIdx);
        
        d = cTest(data(:, 1), data(:, 2), cVal * ones(length(data(:, 1)), 1));

        leaf.test = cTest;
        leaf.value = cVal;
        leaf.gain = mval;
        
        leaf.left = tree;
        leaf.left.data = data(d == 1, :);
        leaf.left.parent = leaf;
        leaf.left.class = mode(data(d == 1, 3));
        
        leaf.right = tree;
        leaf.right.data = data(d == 0, :);
        leaf.right.parent = leaf;
        leaf.right.class = mode(data(d == 0, 3));
        
        nodes = nodes + 2;
        
        [leafs, impurity] = fLeafs(root, classes, sVal, eVal, tests);
        [impurityVal, idx] = max(impurity);
        
        leaf = leafs(idx);
        data = leaf.data;
    end
    tOut = root;
end

%% Find the leaf in the tree with the highest possible gain
function [leafs, imps] = fLeafs(tree, classes, sVal, eVal, tests)
    leafs = [];
    imps = [];
    
    if (~isempty(tree.left))
        [leafA, impsA] = fLeafs(tree.left, classes, sVal, eVal, tests);
        [leafB, impsB] = fLeafs(tree.right, classes, sVal, eVal, tests);
        
        leafs = [leafs, leafA, leafB];
        imps = [imps, impsA, impsB];
    else
        if(isempty(tree.gain))
            e = entropy(classes, tree.data);
            [oData, ~] = gain(e, sVal, eVal, classes, tree.data, tests);
            [mval, ~] = max(oData, [], [1,2], 'linear');
            tree.gain = mval;
        end
        
        leafs = (tree);
        imps = (tree.gain);
    end
end

%% Calcualte the best possible gain for a given node
function [o, v] = gain(e, sVal, eVal, classes, data, tests)
    syms x y f 
    
    divs = 400;
    delim = (eVal - sVal)/divs;
    
    gVal = zeros(divs, length(tests));
    vVal = zeros(divs, length(tests));
    
    dLen = double(length(data(:, 1)));
    
    for item = 1:divs
        val = sVal + (item * delim);
        
        for index = 1:length(tests)
            l = tests{index}(data(:, 1), data(:, 2), val * ones(dLen, 1));
            total = 0;
            
            if (~isempty(data(l == 1)))
                nRatio = sum(l == 1)/dLen;
                total = total + nRatio * entropy(classes, data(l==1, :));
            end
            
            if (~isempty(data(l == 0)))
                nRatio = sum(l == 0)/dLen;
                total = total + nRatio * entropy(classes, data(l==0, :));
            end
            
            
            gVal(item, index) = e - total;
            vVal(item, index) = val;
        end
    end
    
    o = gVal;
    v = vVal;
end

%% Calcualte the entropy of a tree node
function o = entropy(classes, data)
    total = 0;
    
    for class = classes
        f = sum(data(:, 3) == class) / length(data(:, 3));
        
        if (f ~= 0)
            total = total + (f * log10(f));
        end
    end
    
    o = -total;
end

