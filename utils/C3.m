clc;
clear;
tic;
pvar x1 x2;
vars = [x1, x2];
f = [x1*x2 - 2*x1; x1*x2 - x2];

prog = sosprogram(vars);
[prog, B] = sospolyvar(prog, monomials([x1; x2],[1,2]), 'wscoeff1');

inv = [2250000-((x1-(0.0))^2+(x2-(0.0))^2)];

[prog, Q1] = sospolyvar(prog, monomials([x1; x2],[0,1,2]), 'wscoeff2');
prog = sosineq(prog,Q1);
[prog, S1] = sospolyvar(prog, monomials([x1; x2],[0,1,2]), 'wscoeff3');
prog = sosineq(prog,S1);
B_U = B - inv(1) * Q1;
prog = sosineq(prog,B_U);
DB = diff(B, x1) * f(1) + diff(B, x2) * f(2);
DB = - DB - inv(1) * S1;
prog = sosineq(prog, DB);
solver_opt.solver = 'mosek';
prog = sossolve(prog, solver_opt);
SOLB = sosgetsol(prog,B)
toc;
