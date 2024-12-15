function [U1, V1, Q] = MA_CBxDP(data, opts)
    X = data.X;
    Y{1} = data.Y{1};
    Y{2} = data.Y{2};
    Y{3} = data.Y{3};
    Y{4} = data.Y{4};

    E = data.E;
    z = data.z;
    n = size(X, 1);
    p = size(X, 2);
    d = size(E, 2);
    q1 = size(Y{1}, 2);
    q2 = size(Y{2}, 2);
    q3 = size(Y{3}, 2);
    q4 = size(Y{4}, 2);

    U1 = ones(p, 4);
    V1{1} = ones(q1, 1);
    V1{2} = ones(q2, 1);
    V1{3} = ones(q3, 1);
    V1{4} = ones(q4, 1);

    lambda_u1 = opts.lambda.u1;
    lambda_u2 = opts.lambda.u2;
    lambda_u3 = opts.lambda.u3;
    lambda_v1 = opts.lambda.v1;
    lambda_v2 = opts.lambda.v2;
    lambda_v3 = opts.lambda.v3;
    lambda_v4 = opts.lambda.v4;
    
    max_Iter = 100;
    t = 0;
    tol = 1e-5;
    tu1 = inf;
    tv1 = inf;
    
    for i = 1 : 4
        U = U1(:, i) ;
        scale = sqrt(U' * X' * X * U);
        U1(:, i) = U1(:, i) ./ scale;
        V = V1{i};
        scale = sqrt(V' * Y{i}' *  Y{i} * V);
        V1{i} = V1{i} ./ scale;
    end

    W = ones(n,1);
    block = 200;
    nblock = ceil(p / block);
    Xall = X;
    Yv1 = Y{1} * V1{1};
    Yv2 = Y{2} * V1{2};
    Yv3 = Y{3} * V1{3};
    Yv4 = Y{4} * V1{4};
    
    random_state = 0;
    DC = decorrelation(X, random_state);
    Q_ini = rand(p, d);
    Qvec = vectorization(Q_ini);
    Qvec = Qvec./norm(Qvec);
    Q_ini = de_vectorization(Qvec, p, d);
    Q = Q_ini;
    step = 0.01;
    alpha = 0.1;

  while (t < max_Iter && (tu1 > tol|| tv1 > tol))
        t = t + 1;
        U1_old = U1;
        V1_old = V1;
        Yv = [Yv1 Yv2 Yv3 Yv4];
        ut = [];
        uq = [];
        su1 = 0;
        su2 = 0;
        su3 = 0;
        su4 = 0;

        for iu = 1 : nblock
        if iu * block <= p
            X = Xall(:, 1 + (iu - 1) * block : iu * block);
            sub_u = U1(1 + (iu - 1) * block : iu * block,:);
            Qb = Q(1 + (iu - 1) * block : iu * block, :);
        else
            X = Xall(:, 1 + (iu - 1) * block : end);
            sub_u = U1(1 + (iu - 1) * block : end, :);
            Qb = Q(1 + (iu - 1) * block : end, :);
        end
        W  = DC;
        XX = X' * diag(W) * X;

        for i = 1 : 4
            XY{i} = X' * diag(W) * Y{i};
            YY{i} = Y{i}' * diag(W) * Y{i};
            YX{i} = XY{i}';
        end

        d1 = updateD(sub_u);
        D1 = diag(d1);
        fd1 = updateD_FGL21(sub_u(:, 1), sub_u(:, 2), sub_u(:, 3), sub_u(:, 4));
        fD1 = diag(fd1);  

        for i = 1 : 4
            du1 = updateD(sub_u(:,i));
            Du1{i} = diag(du1);
        end

        for i = 1 : 4
        F1 =  XX + alpha * X' * X  + lambda_u1 * D1 + lambda_u2 * fD1 + lambda_u3 * Du1{i};
        b1 =  XY{i} * V1{i} - X' * diag(W) * diag(Xall * Q * E') - alpha * X' * diag(Xall * Q * E') + alpha * X' * z -  alpha * X' * Y{i} * V1{i};
        sub_u(:,i) = F1 \ b1;
        end

        ut = [ut; sub_u];
        [su1, su2, su3, su4] = cal_scale(sub_u, XX, su1, su2, su3, su4);

        m = size(X,1);
        pb = size(X,2);
        grad_Q = zeros(pb,d);
    
        for i = 1:n
            for j = 1:4
                xi = X(i,:);
                ei = E(i,:);
                zi = z(i,:);
                yVi = Y{1}(i, :) * V1{1} + Y{2}(i, :) * V1{2} + Y{3}(i, :) * V1{3} + Y{4}(i, :) * V1{4};
                Xtemp1 = (4 * xi * sub_u(:, j) + xi * Qb * ei' - yVi);
                Xtemp2 = (4 * xi * sub_u(:, j) + xi * Qb * ei' - zi);
                grad_Q = grad_Q + W(i) * (xi' * Xtemp1) * ei +  alpha * (xi' * Xtemp2) * ei;
            end
        end
        
        Qb = Qb - step * grad_Q;
        Qvec = vectorization(Qb);
        Qvec = Qvec./norm(Qvec);
        lambdaQ = 0.01;
        Qs = soft_Threshold (Qvec, lambdaQ);
        Qs = Qs./norm(Qs);
        Qb = de_vectorization(Qs, pb, d);
        uq = [uq; Qb];
    end

        U1(:, 1) = ut(:,1) / sqrt(su1);
        U1(:, 2) = ut(:,2) / sqrt(su2);
        U1(:, 3) = ut(:,3) / sqrt(su3);
        U1(:, 4) = ut(:,4) / sqrt(su4);
        Q = uq;

        for i = 1 : 4
            XY{i} = Xall' * diag(W) * Y{i};
            YY{i} = Y{i}' * diag(W) * Y{i};
            YX{i} = XY{i}';
        end
 
    d21 = updateD(V1{1});
    D21 = diag(d21);
    F1 =  YY{1} + alpha * Y{1}' * Y{1} + lambda_v1 * D21;
    b1 =  YX{1} * U1(:, 1)  + Y{1}' * diag(W) * diag(Xall * Q * E') + alpha * Y{1}' * z - alpha * Y{1}' * Xall * U1(:, 1) - alpha * Y{1}' * diag(Xall * Q * E');
    V1{1} = F1 \ b1;
    V = V1{1};
    scale = sqrt(V' * Y{1}' * Y{1} * V);
    V1{1} = V1{1} ./ scale;

    d22 = updateD(V1{2});
    D22 = diag(d22);
    F2 = YY{2} + alpha * Y{2}' * Y{2} + lambda_v2 * D22;
    b2 = YX{2} * U1(:, 2) + Y{2}' * diag(W) * diag(Xall * Q * E') + alpha * Y{2}' * z - alpha * Y{2}' * Xall * U1(:, 2) - alpha * Y{2}' * diag(Xall * Q * E');
    V1{2} = F2 \ b2;
    V = V1{2};
    scale = sqrt(V' * Y{2}' * Y{2} * V);
    V1{2} = V1{2} ./ scale;

    d23 = updateD(V1{3});
    D23 = diag(d23);
    F2 = YY{3} + alpha * Y{3}' * Y{3} + lambda_v3 * D23;
    b2 = YX{3} * U1(:, 3) + Y{3}' * diag(W) * diag(Xall * Q * E') + alpha * Y{3}' * z - alpha * Y{3}' * Xall * U1(:, 3) - alpha * Y{3}' * diag(Xall * Q * E');
    V1{3} = F2 \ b2;
    V = V1{3};
    scale = sqrt(V' * Y{3}' * Y{3} * V);
    V1{3} = V1{3} ./ scale;

    d24 = updateD(V1{4});
    D24 = diag(d24);
    F2 = YY{4} + alpha * Y{4}' * Y{4} + lambda_v4 * D24;
    b2 = YX{4} * U1(:, 4 ) + Y{4}' * diag(W) * diag(Xall * Q * E') + alpha * Y{4}' * z - alpha * Y{4}' * Xall * U1(:, 4) - alpha * Y{4}' * diag(Xall * Q * E');
    V1{4} = F2 \ b2;
    V = V1{4};
    scale = sqrt(V' * Y{4}' * Y{4} * V);
    V1{4} = V1{4} ./ scale;

    tu1 = max(max(abs(U1 - U1_old)));
    t1 = max(max(abs(V1{1} - V1_old{1})));
    t2 = max(max(abs(V1{2} - V1_old{2})));
    t3 = max(max(abs(V1{3} - V1_old{3})));
    t4 = max(max(abs(V1{4} - V1_old{4})));
    tv1 =max([t1,t2,t3,t4]);
    end
end

function D = updateD(W, group)
    [n_features, n_tasks] = size(W);
        for i = 1 : n_features
            d(i) = sqrt(sum(W(i, :) .^ 2) + eps);
        end
     D = 0.5 ./ d;
end

function [D] = updateD_FGL21(u1, u2, u3, u4)
    ulen = length(u1);
    for i = 1 : ulen
        if i == 1
        d(i) = sqrt(u1(i).^2 + u2(i).^2 + u3(i).^2 + u4(i).^2 + u1(i+1).^2 + u2(i+1).^2 + u3(i+1).^2 + u4(i+1).^2 + eps);
        d(i) = 0.5 ./ d(i);
        elseif i == ulen
        d(i) = sqrt(u1(i-1).^2 + u2(i-1).^2 + u3(i-1).^2 + u4(i-1).^2  + u1(i).^2 + u2(i).^2 + u3(i).^2 + u4(i).^2 + eps);
        d(i) = 0.5 ./ d(i);
        else
        d(i) = 0.5./(sqrt(u1(i-1).^2 + u2(i-1).^2 + u3(i-1).^2 + u4(i-1).^2 + u1(i).^2 + u2(i).^2 + u3(i).^2 + u4(i).^2 + eps)) + 0.5./(sqrt(u1(i).^2 + u2(i).^2 + u3(i).^2 + u4(i).^2  + u1(i+1).^2 + u2(i+1).^2 + u3(i+1).^2 + u4(i+1).^2 + eps));
        end
     D = d;
    end
end

function weights = decorrelation(x, random_state)
    [n, p] = size(x);
    x_decorrelation = column_wise_resampling(x, 'random_state', random_state);
    P = array2table(x);
    Q = array2table(x_decorrelation);
    P.src = ones(n, 1); 
    Q.src = zeros(n, 1);
    P = table2array(P);
    Q = table2array(Q);
    Z = [P; Q];
    labels = Z(:,end);
    data = Z(:,1:end-1);
    svm = fitcsvm(data, labels, 'Standardize', true, 'KernelFunction', 'linear', 'BoxConstraint', 1, 'NumPrint', 0);
    supportVectors = svm.SupportVectors;  
    coefficients = svm.Alpha;  
    bias = svm.Bias;  
    probabilities = binaryClassification(data, labels, supportVectors, coefficients, bias);
    proba = probabilities(1:n,:);
    weights = (1./proba) - 1.; 
    weights = weights ./ mean(weights); 
end

function z = vectorization(Q)
    p = size(Q, 1);
    d = size(Q, 2);
    z = zeros(p * d, 1);
    z(1:end) = reshape(Q, p * d, 1);
end

function [Q] = de_vectorization(z, p, d)
    Q = reshape(z(1 : end, 1), p, d);
end


function x = soft_Threshold(x, lambda)
    p = length(x);
    for i = 1:p
        if x(i) > lambda
        x(i) = x(i)-lambda;
        elseif x(i) < -lambda
        x(i) = x(i)+lambda;
        else
        x(i) = 0;
        end
    end
end


function x_decorrelation = column_wise_resampling(x, replacement, random_state, varargin)
    rng = RandStream('mt19937ar', 'Seed', random_state);
    [n, p] = size(x);
    if any(strcmp(varargin, 'sensitive_variables'))
        sensitive_variables = varargin{find(strcmp(varargin, 'sensitive_variables')) + 1};
    else
        sensitive_variables = 1:p;
    end
    x_decorrelation = zeros(n, p);
    for i = sensitive_variables
        var = x(:, i);
        if replacement  
            x_decorrelation(:, i) = var(datasample(rng, 1 : n, n, 'Replace', true));
        else  
            x_decorrelation(:, i) = var(randperm(rng, n));
        end
    end
end


function probabilities = binaryClassification(Z, labels, supportVectors, coefficients, bias)
    n = size(Z, 1);  
    nSV = size(supportVectors, 1);  
    probabilities = zeros(n, 1);  
    for i = 1:n
        prediction = 0;
        for j = 1:nSV
            kernel = dot(supportVectors(j, :), Z(i, :));  
            prediction = prediction + coefficients(j) * kernel;  
        end
        prediction = prediction + bias;  
        probabilities(i) = 1 / (1 + exp(-prediction));
    end
end

function [sum1 sum2 sum3 sum4] = cal_scale(beta, XX, sum1, sum2, sum3, sum4)
      [p,ntask] = size(beta);
    if nargin <3
        for i = 1:ntask
        temp1 = beta(:,i);
        temp2 = temp1';
        eval(['sum', str2double(i),' = temp2 * XX * temp1;'])
        end
    else
        for i = 1:ntask
        temp1 = beta(:,i);
        temp2 = temp1';
        eval(['sum',num2str(i),' = sum', num2str(i),'+ temp2 * XX * temp1;']);
        end
    end
end
