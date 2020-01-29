function [S, U, sv, tol] = hosvd(T, dim, tol, signConsistence)
%HOSVD High Order SVD of a multidimensional array
%	[S U sv tol] = HOSVD(T)
%	[S U sv tol] = HOSVD(T, dim)
%	[S U sv tol] = HOSVD(T, dim, tol)
%
%	T    - multidimensional array
%	dim  - bools: hosvd is applied in these dimensions (default: ones())
%	tol  - tolerance for each dimensions (default: eps * ones())
%
%	S    - decomposed core tensor so that T==tprod(S, U)
%	U    - matrices for each dimension (U{n}==[] if dim(n) was 0)
%	sv   - n-mode singular values (or [] if dim(n) was 0)
%	tol  - larges dropped singular value (hosvd truncates small sv)
%
%	eg. [S, U, sv, tol] = hosvd(ones(3,4,5), [1 0 1])
%
%	See also TPROD, SVDTRUNC

M = size(T);
P = length(M);
if nargin < 2 || isempty(dim)
    dim = ones(1,P);
end
if nargin < 3 || isempty(tol)
    tol = eps;
end
if nargin < 4
    signConsistence = false;
end
if numel(tol) == 1
    tol = tol*ones(1,P);
end
U  = cell(1,P);
UT = cell(1,P);
sv = cell(1,P);
for i = 1:P
    if dim(i)
        A = ndim_unfold(T, i);
        % SVD based reduction of the current dimension (i)
        [Ui, svi, toli] = svdtrunc(A, tol(i));
        if signConsistence
            signMat = diag(sign(Ui(1,:)));
            U{i} =  Ui * signMat;
        else
            U{i} = Ui;
        end
        UT{i} = U{i}';
        sv{i} = svi;
        tol(i) = toli;
    else
        U{i} = [];
        UT{i} = [];
        sv{i} = [];
        tol(i) = 0;
    end
end
S = tprod(T, UT);
end