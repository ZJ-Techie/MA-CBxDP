function Y = getNormalization(X, type)

if nargin < 2
    type = 'std';
end

[~, p] = size(X);
Y = X;

if strcmpi(type, 'std')
    for i = 1 : p
        Xv = X(:, i);
        Xvn = (Xv - mean(Xv)) / std(Xv);
        Y(:, i) = Xvn;
    end
elseif strcmpi(type, 'maxmin')
    for i = 1 : p
        Xv = X(:, i);
        Xvn = (Xv - min(Xv)) / (max(Xv) - min(Xv));
        Y(:, i) = Xvn;
    end
elseif strcmpi(type, 'dmax')
    for i = 1 : p
        Xv = X(:, i);
        Xvn = Xv / max(Xv);
        Y(:, i) = Xvn;
    end
elseif strcmpi(type, 'unit')
    for i = 1 : p
        Xv = X(:, i);
        Xvn = Xv / norm(Xv);
        Y(:, i) = Xvn;
    end
elseif strcmpi(type, 'centered')
    for i = 1 : p
        Xv = X(:, i);
        Xvn = Xv - mean(Xv);
        Y(:, i) = Xvn;
    end
elseif strcmpi(type, 'normalize')
    for i = 1 : p
        Xv = X(:, i);
        Xv = Xv - mean(Xv);
        Xvn = Xv / (norm(Xv) + eps);
        Y(:, i) = Xvn;
    end
end
