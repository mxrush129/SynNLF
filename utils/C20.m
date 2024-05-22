clc;
clear;
tic;
pvar x1 x2 x3 x4 x5 x6;
vars = [x1, x2, x3, x4, x5, x6];
f = [-x1^3 + x2*x4; -3*x1*x4 - x2^3; -3*x1*x4^3 - x3; x1*x3 - x4; -x5 + x6^3; x3^4 - x5 - x6];

prog = sosprogram(vars);
[prog, B] = sospolyvar(prog, monomials([x1; x2; x3; x4; x5; x6],[1,2]), 'wscoeff1');

inv = [1000000-((x1-(0.0))^2+(x2-(0.0))^2+(x3-(0.0))^2+(x4-(0.0))^2+(x5-(0.0))^2+(x6-(0.0))^2)];

[prog, Q1] = sospolyvar(prog, monomials([x1; x2; x3; x4; x5; x6],[0,1,2]), 'wscoeff2');
prog = sosineq(prog,Q1);
[prog, S1] = sospolyvar(prog, monomials([x1; x2; x3; x4; x5; x6],[0,1,2]), 'wscoeff3');
prog = sosineq(prog,S1);
B_U = B - inv(1) * Q1;
prog = sosineq(prog,B_U);
DB = diff(B, x1) * f(1) + diff(B, x2) * f(2) + diff(B, x3) * f(3) + diff(B, x4) * f(4) + diff(B, x5) * f(5) + diff(B, x6) * f(6);
DB = - DB - inv(1) * S1;
prog = sosineq(prog, DB);
solver_opt.solver = 'mosek';
prog = sossolve(prog, solver_opt);
SOLB = sosgetsol(prog,B)
toc;
