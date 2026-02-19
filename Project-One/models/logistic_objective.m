function varargout = logistic_objective(theta, X, y, mu, mode)
%LOGISTIC_OBJECTIVE Regularized logistic regression objective (stable).
%
% Project spec interface:
%   mode = 0: f
%   mode = 1: g   (BUT: steplength.m in your project calls [~,g]=obj(...,1))
%   mode = 2: [f,g]
%   mode = 3: [g,H]
%
% This implementation is compatible with BOTH the PDF spec and steplength.m:
%   - mode=1, nargout==1 -> returns g
%   - mode=1, nargout==2 -> returns [f,g]

    y = y(:);
    [N,d] = size(X);
    theta = theta(:);

    % Add bias column
    Xb = [X, ones(N,1)];   % N x (d+1)

    % Scores
    s = Xb * theta;        % N x 1

    % Stable log(1+exp(s))
    log1pexp = log1p(exp(-abs(s))) + max(s,0);

    % Sigmoid
    p = 1 ./ (1 + exp(-s));

    % Loss
    f = (1/N) * sum(log1pexp - y .* s) + (mu/2) * (theta' * theta);

    % Gradient
    g = (1/N) * (Xb' * (p - y)) + mu * theta;

    % Hessian
    w = p .* (1 - p);                    % N x 1
    H = (1/N) * (Xb' * (Xb .* w)) + mu * eye(d+1);

    switch mode
        case 0
            varargout{1} = f;

        case 1
            % Compatibility shim for steplength.m:
            % If caller asks for 2 outputs, give [f,g]
            if nargout >= 2
                varargout{1} = f;
                varargout{2} = g;
            else
                varargout{1} = g;
            end

        case 2
            varargout{1} = f;
            varargout{2} = g;

        case 3
            varargout{1} = g;
            varargout{2} = H;

        otherwise
            error('Invalid mode. Use mode in {0,1,2,3}.');
    end
end