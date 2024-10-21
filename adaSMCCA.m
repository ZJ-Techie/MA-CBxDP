function [W, u, v, w, obj] = adaSMCCA(X, Y, Z, opts)
% --------------------------------------------------------------------
%% Problem
% --------------------------------------------------------------------
p = size(X, 2);
q = size(Y, 2);
r = size(Z, 2);

X = getNormalization(X, 'normalize');
Y = getNormalization(Y, 'normalize');
Z = getNormalization(Z, 'normalize');

% Calculate coverance between X and Y
XY = X'* Y;
YX = Y'* X;
XZ = X'* Z;
ZX = Z'* X;
YZ = Y'* Z;
ZY = Z'* Y;

u0 = ones(p, 1);
v0 = ones(q, 1);
w0 = ones(r, 1);
u = u0;
v = v0;
w = w0;
iu = 1; iv = 2; iw = 3;
W{iu} = u; W{iv} = v; W{iw} = w;
% % scale u and v
scale = sqrt(u' * u);
u = u ./ scale;
scale = sqrt(v' * v);
v = v ./ scale;
scale = sqrt(w' * w);
w = w ./ scale;
% set parameters
lambda.u1 = opts.lambda.u1; % L1
lambda.v1 = opts.lambda.v1; % L1
lambda.w1 = opts.lambda.w1; % L1
% set stopping criteria
max_Iter = 100;
i = 0;
tol = 1e-5;
obj = [];
tv = inf;
tu = inf;
tw = inf;
tall = inf;
res1 = [];
res2 = [];
res3 = [];

Xu = X * u;
Yv = Y * v;
Zw = Z * w;
omega12 = 1 / (norm(Xu) * norm(Yv));
omega13 = 1 / (norm(Xu) * norm(Zw));
omega23 = 1 / (norm(Yv) * norm(Zw));

while (i < max_Iter && tall > tol) % default 100 times of iteration
    i = i+1;
    
    % update loss weight
    omega12 = 1 / (norm(Xu) * norm(Yv));
    omega13 = 1 / (norm(Xu) * norm(Zw));
    omega23 = 1 / (norm(Yv) * norm(Zw));
    
    % update u
    u_old = u;
    % -------------------------------------
    % solve u
    u = omega12 * XY * v + omega13 * XZ * w;
    u = soft_Threshold(u, lambda.u1);
    u = u ./ norm(u);
%     omega12 = 1 / (norm(Xu) * norm(Yv));
%     omega13 = 1 / (norm(Xu) * norm(Zw));
    
    % update v
    v_old = v;
    % -------------------------------------
    v = omega12 * YX * u + omega23 * YZ * w;
    v = soft_Threshold(v, lambda.v1);
    v = v ./ norm(v);
%     omega12 = 1 / (norm(Xu) * norm(Yv));
%     omega23 = 1 / (norm(Yv) * norm(Zw));
    
    % update w
    w_old = w;
    % -------------------------------------
    w = omega13 * ZX * u + omega23 * ZY * v;
    w = soft_Threshold(w, lambda.w1);
    w = w ./ norm(w);
%     omega13 = 1 / (norm(Xu) * norm(Zw));
%     omega23 = 1 / (norm(Yv) * norm(Zw));
    
    % Running objective
    % ------------------------------
    % stopping condition
    % Condition 1:
    res1(end + 1) = max(abs(u - u_old));
    res2(end + 1) = max(abs(v - v_old));
    res3(end + 1) = max(abs(w - w_old));
    if i > 1        
        tu = max(abs(u - u_old));
        tv = max(abs(v - v_old));
        tw = max(abs(w - w_old));
    else
        tu = tol * 10;
        tv = tol * 10;
        tw = tol * 10;
    end
    tall = max([tu, tv, tw]);
    
    % objective
    obj(i) = -omega12 * u' * XY * v -omega13 * u' * XZ * w - omega23 * v' * YZ * w + lambda.u1 * sum(abs(u)) ...
        + lambda.v1 * sum(abs(v)) + lambda.w1 * sum(abs(w));
end
W{iu} = u; W{iv} = v; W{iw} = w;
end