% ========================================================================
% main_BinaryClassification_Part6.m
%
% AM 230 – Numerical Optimization
% Project I: Logistic Regression
% Part 6: Newton's Method (local/global convergence)
%
% This script is CLEANED to run ONLY Part 6 moving forward.
%
% Requirements (from the project pdf):
%   - Fix mu = 1e-4
%   - Stop when ||grad Lbar(theta_k)||_2 <= 1e-9
%   - Compare pure Newton (alpha=1) vs damped Newton (strong Wolfe line search)
%   - Test theta0 = [1;1;1], [2;2;2], and several random initial conditions
%   - Compare iteration counts with gradient descent (from previous section)
%
% Folder structure assumed:
%   ./data    : Project1_train.mat
%   ./models  : logistic_objective.m    (must support mode=2 and mode=3)
%   ./solvers : solve_gd.m, solve_newton.m
%   ./utils   : steplength.m  (strong Wolfe)
% ========================================================================

clear; close all; clc;

%% ------------------------------------------------------------------------
%  Path setup
% -------------------------------------------------------------------------
restoredefaultpath;
rehash toolboxcache;

root = fileparts(mfilename('fullpath'));
addpath(root);
addpath(fullfile(root,'data'));
addpath(fullfile(root,'models'));
addpath(fullfile(root,'solvers'));
addpath(fullfile(root,'utils'));

%% ------------------------------------------------------------------------
%  Load training data
% -------------------------------------------------------------------------
Tr = load(fullfile(root,'data','Project1_train.mat'));
X = Tr.X;          % N x d
y = Tr.y(:);       % N x 1 in {0,1}
[N,d] = size(X);

%% ------------------------------------------------------------------------
%  Problem struct
% -------------------------------------------------------------------------
problem = struct();
problem.Xtr = X;
problem.ytr = y;
problem.mu  = 1e-4;                  % Part 6 fixed
problem.obj = @logistic_objective;   % must support mode=2 and mode=3

%% ------------------------------------------------------------------------
%  Shared stopping tolerance for Part 6
% -------------------------------------------------------------------------
tol_newton = 1e-9;

%% ------------------------------------------------------------------------
%  Newton options
% -------------------------------------------------------------------------
optsN = struct();
optsN.maxIter       = 200;    % Newton should converge quickly if it converges
optsN.tolGrad       = tol_newton;
optsN.printEvery    = 1;
optsN.useLineSearch = false;  % pure Newton by default

% Line search sub-opts (used only when optsN.useLineSearch = true)
optsN.ls = struct();
optsN.ls.c1 = 1e-4;
optsN.ls.c2 = 0.9;
optsN.ls.alpha0 = 1;          % will be changed in 6.3(c)
optsN.ls.alphaMax = 50;
optsN.ls.maxIter = 30;

%% ------------------------------------------------------------------------
%  Part 6.2(a): Pure Newton (alpha = 1)
% -------------------------------------------------------------------------
theta0_list = {};
theta0_list{end+1} = [1;1;1];
theta0_list{end+1} = [2;2;2];

rng(0);
num_rand = 5;
for i = 1:num_rand
    theta0_list{end+1} = randn(d+1,1);
end

theta_pure = cell(numel(theta0_list),1);
hist_pure  = cell(numel(theta0_list),1);

fprintf('\n===== Part 6.2(a): PURE Newton (alpha = 1) =====\n');
optsN.useLineSearch = false;

for j = 1:numel(theta0_list)
    theta0 = theta0_list{j};

    fprintf('\n--- Pure Newton run %d/%d ---\n', j, numel(theta0_list));
    fprintf('theta0 = [% .3f % .3f % .3f]^T\n', theta0(1), theta0(2), theta0(3));

    [theta_pure{j}, hist_pure{j}] = solve_newton(problem, theta0, optsN);

    fprintf('Finished: iters = %d, final ||g|| = %.3e\n', ...
        hist_pure{j}.iter(end), hist_pure{j}.gn(end));
end

% Plot gradient norms (pure Newton)
figure('Name','Part 6.2(a): Pure Newton');
hold on; grid on;
for j = 1:numel(theta0_list)
    semilogy(hist_pure{j}.iter, hist_pure{j}.gn, 'LineWidth', 2);
end
xlabel('Iteration k');
ylabel('$\|\nabla \bar L(\theta_k)\|_2$','Interpreter','latex');
title('Pure Newton (alpha=1): Gradient norm vs iteration');
legendStrings = cell(numel(theta0_list),1);
legendStrings{1} = 'theta0=[1;1;1]';
legendStrings{2} = 'theta0=[2;2;2]';
for j = 3:numel(theta0_list)
    legendStrings{j} = sprintf('rand %d', j-2);
end
legend(legendStrings,'Location','southwest');

%% ------------------------------------------------------------------------
%  Part 6.2(b): Compare iterations with GD (use "best" constant step from Hessian at theta*)
%  We'll compute theta* using DAMPED Newton (safer), then compute alpha*,
%  then run GD to same tol.
% -------------------------------------------------------------------------
fprintf('\n===== Part 6.2(b): Compare Newton vs GD iterations (same tol) =====\n');

% Get a reliable theta* via damped Newton (alpha0=1)
optsN.useLineSearch = true;
optsN.ls.alpha0 = 1;
theta0_ref = zeros(d+1,1);

fprintf('\nComputing reference minimizer theta* via damped Newton (alpha0=1)...\n');
[theta_star, hist_star] = solve_newton(problem, theta0_ref, optsN);

% Hessian eigenvalues at theta*
[g_star, H_star] = problem.obj(theta_star, X, y, problem.mu, 3);
eigH = eig(H_star);
lam_min = min(eigH);
lam_max = max(eigH);

alpha_star = 2/(lam_min + lam_max);   % optimal constant step for quadratic model

fprintf('theta* computed. Hessian eig range: [%.3e, %.3e]\n', lam_min, lam_max);
fprintf('Using alpha_star = 2/(lam_min+lam_max) = %.6e for GD comparison.\n', alpha_star);

% Run GD to same tolerance
optsG = struct();
optsG.maxIter    = 200000;      % GD may need many iterations for tight tol
optsG.tolGrad    = tol_newton;
optsG.alpha_fixed = alpha_star;
optsG.printEvery = 5000;

theta0_gd = zeros(d+1,1);
[theta_gd, hist_gd] = solve_gd(problem, theta0_gd, optsG);

fprintf('\nIteration counts to reach ||g||<=1e-9:\n');
fprintf('  Damped Newton (alpha0=1): %d iterations\n', hist_star.iter(end));
fprintf('  GD (alpha_star):          %d iterations\n', hist_gd.iter(end));

figure('Name','Part 6.2(b): Newton vs GD');
grid on; hold on;
semilogy(hist_star.iter, hist_star.gn, 'LineWidth', 2);
semilogy(hist_gd.iter,   hist_gd.gn,   'LineWidth', 2);
xlabel('Iteration k');
ylabel('$\|\nabla \bar L(\theta_k)\|_2$','Interpreter','latex');
title('Newton vs GD: Gradient norm vs iteration (same tol)');
legend({'Damped Newton (Wolfe)','GD (alpha\_star)'},'Location','southwest');

%% ------------------------------------------------------------------------
%  Part 6.3(b): Damped Newton with strong Wolfe line search (alpha0 = 1)
% -------------------------------------------------------------------------
fprintf('\n===== Part 6.3(b): Damped Newton (Wolfe), alpha0 = 1 =====\n');
optsN.useLineSearch = true;
optsN.ls.alpha0 = 1;

theta_damped = cell(numel(theta0_list),1);
hist_damped  = cell(numel(theta0_list),1);

for j = 1:numel(theta0_list)
    theta0 = theta0_list{j};

    fprintf('\n--- Damped Newton run %d/%d (alpha0=1) ---\n', j, numel(theta0_list));
    fprintf('theta0 = [% .3f % .3f % .3f]^T\n', theta0(1), theta0(2), theta0(3));

    [theta_damped{j}, hist_damped{j}] = solve_newton(problem, theta0, optsN);

    fprintf('Finished: iters = %d, final ||g|| = %.3e\n', ...
        hist_damped{j}.iter(end), hist_damped{j}.gn(end));
end

figure('Name','Part 6.3(b): Damped Newton alpha0=1');
hold on; grid on;
for j = 1:numel(theta0_list)
    semilogy(hist_damped{j}.iter, hist_damped{j}.gn, 'LineWidth', 2);
end
xlabel('Iteration k');
ylabel('$\|\nabla \bar L(\theta_k)\|_2$','Interpreter','latex');
title('Damped Newton (Wolfe), alpha0=1: Gradient norm vs iteration');
legend(legendStrings,'Location','southwest');

%% ------------------------------------------------------------------------
%  Part 6.3(c): Damped Newton with alpha0 = 0.1 (do NOT try alpha=1 first)
% -------------------------------------------------------------------------
fprintf('\n===== Part 6.3(c): Damped Newton (Wolfe), alpha0 = 0.1 =====\n');
optsN.useLineSearch = true;
optsN.ls.alpha0 = 0.1;

theta_damped01 = cell(numel(theta0_list),1);
hist_damped01  = cell(numel(theta0_list),1);

for j = 1:numel(theta0_list)
    theta0 = theta0_list{j};

    fprintf('\n--- Damped Newton run %d/%d (alpha0=0.1) ---\n', j, numel(theta0_list));
    fprintf('theta0 = [% .3f % .3f % .3f]^T\n', theta0(1), theta0(2), theta0(3));

    [theta_damped01{j}, hist_damped01{j}] = solve_newton(problem, theta0, optsN);

    fprintf('Finished: iters = %d, final ||g|| = %.3e\n', ...
        hist_damped01{j}.iter(end), hist_damped01{j}.gn(end));
end

figure('Name','Part 6.3(c): Damped Newton alpha0=0.1');
hold on; grid on;
for j = 1:numel(theta0_list)
    semilogy(hist_damped01{j}.iter, hist_damped01{j}.gn, 'LineWidth', 2);
end
xlabel('Iteration k');
ylabel('$\|\nabla \bar L(\theta_k)\|_2$','Interpreter','latex');
title('Damped Newton (Wolfe), alpha0=0.1: Gradient norm vs iteration');
legend(legendStrings,'Location','southwest');

fprintf('\nDONE: Part 6 runs completed.\n');