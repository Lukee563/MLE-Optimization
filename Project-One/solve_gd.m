function [theta, hist] = solve_gd(problem, theta0, opts)


%SOLVE_GD Gradient Descent (fixed step size)
%
%   [theta, hist] = solve_gd(problem, theta0, opts)
%
% Implements gradient descent with a FIXED step size:
%     theta_{k+1} = theta_k - alpha * grad f(theta_k)
%
% INPUTS
%   problem.Xtr : N x d training data (rows are samples)
%   problem.ytr : N x 1 labels in {0,1}
%   problem.mu  : regularization parameter
%   problem.obj : objective function handle
%                [f,g] = obj(theta, X, y, mu, mode) with mode=2 returning f,g
%
%   theta0      : initial parameter vector (d+1) x 1, theta = [w; b]
%
%   opts.maxIter     : maximum number of iterations
%   opts.tolGrad     : stopping tolerance on gradient norm
%   opts.alpha_fixed : fixed step size alpha > 0 (MUST be scalar)
%   opts.printEvery  : print frequency (set 0 to disable)
%
% OUTPUTS
%   theta : final iterate
%
%   hist  : iteration history struct with fields
%       hist.iter       : iteration indices, starting at 0
%       hist.f          : objective values f(theta_k)
%       hist.gn         : gradient norms ||grad f(theta_k)||_2
%       hist.theta_norm : parameter norms ||theta_k||_2

    % ----- unpack problem -----
    X   = problem.Xtr;
    y   = problem.ytr(:);
    mu  = problem.mu;
    obj = problem.obj;

    % ensure theta is a column vector
    theta = theta0(:);

    % ----- unpack options -----
    maxIter    = opts.maxIter;
    tolGrad    = opts.tolGrad;
    alpha      = opts.alpha_fixed;
    printEvery = opts.printEvery;

    % ----- validate alpha -----
    if isempty(alpha) || ~isscalar(alpha) || ~isfinite(alpha) || alpha <= 0
        error('opts.alpha_fixed must be a finite, positive scalar. Got: size=%s, value=%s.', ...
              mat2str(size(alpha)), mat2str(alpha));
    end

    % ----- initialize history (preallocate to max possible length) -----
    hist.iter       = zeros(maxIter+1,1);
    hist.f          = zeros(maxIter+1,1);
    hist.gn         = zeros(maxIter+1,1);
    hist.theta_norm = zeros(maxIter+1,1);

    % ----- k = 0 evaluation -----
    [f,g] = obj(theta, X, y, mu, 2);
    gn = norm(g);

    hist.iter(1)       = 0;
    hist.f(1)          = f;
    hist.gn(1)         = gn;
    hist.theta_norm(1) = norm(theta);

    if printEvery > 0
        fprintf('GD iter %6d: f = %.6e, ||g|| = %.3e\n', 0, f, gn);
    end

    % ----- main GD loop -----
    for k = 1:maxIter

        % Descent direction
        p = -g;

        % GD update (use .* to avoid dimension issues if alpha is accidentally non-scalar)
        theta = theta + alpha .* p;   % equivalent: theta = theta - alpha.*g;

        % Evaluate objective and gradient at new iterate
        [f,g] = obj(theta, X, y, mu, 2);
        gn = norm(g);

        % Record history (index k+1 corresponds to iteration k)
        hist.iter(k+1)       = k;
        hist.f(k+1)          = f;
        hist.gn(k+1)         = gn;
        hist.theta_norm(k+1) = norm(theta);

        if printEvery > 0 && mod(k, printEvery) == 0
            fprintf('GD iter %6d: f = %.6e, ||g|| = %.3e\n', k, f, gn);
        end

        % stopping criterion
        if gn <= tolGrad
            hist.iter       = hist.iter(1:k+1);
            hist.f          = hist.f(1:k+1);
            hist.gn         = hist.gn(1:k+1);
            hist.theta_norm = hist.theta_norm(1:k+1);
            return;
        end
    end

    % trim history if maxIter reached
    hist.iter       = hist.iter(1:maxIter+1);
    hist.f          = hist.f(1:maxIter+1);
    hist.gn         = hist.gn(1:maxIter+1);
    hist.theta_norm = hist.theta_norm(1:maxIter+1);
end