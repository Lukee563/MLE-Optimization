function [theta, hist] = solve_newton(problem, theta0, opts)

% Newton method for regularized logistic regression

X   = problem.Xtr;
y   = problem.ytr(:);
mu  = problem.mu;
obj = problem.obj;

theta = theta0(:);

maxIter    = opts.maxIter;
tolGrad    = opts.tolGrad;
printEvery = opts.printEvery;

hist.iter       = zeros(maxIter+1,1);
hist.f          = zeros(maxIter+1,1);
hist.gn         = zeros(maxIter+1,1);
hist.theta_norm = zeros(maxIter+1,1);

for k = 0:maxIter

    % mode = 3 must return gradient AND Hessian
    [g,H] = obj(theta, X, y, mu, 3);

    f  = obj(theta, X, y, mu, 0);
    gn = norm(g);

    hist.iter(k+1)       = k;
    hist.f(k+1)          = f;
    hist.gn(k+1)         = gn;
    hist.theta_norm(k+1) = norm(theta);

    if gn <= tolGrad
        break;
    end

    % Newton direction
    p = -(H \ g);

    % Step length
    if opts.useLineSearch
        alpha = steplength(problem, theta, p, opts.ls);
    else
        alpha = 1;
    end

    theta = theta + alpha*p;

    if printEvery>0 && mod(k,printEvery)==0
        fprintf('Newton iter %4d: f=%.3e ||g||=%.3e alpha=%.2e\n',k,f,gn,alpha);
    end
end

hist.iter       = hist.iter(1:k+1);
hist.f          = hist.f(1:k+1);
hist.gn         = hist.gn(1:k+1);
hist.theta_norm = hist.theta_norm(1:k+1);
end