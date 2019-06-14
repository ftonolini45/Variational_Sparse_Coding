function In = add_eyes(I,s,v1,v2,v3)

if s == 1
    
    c = [16.5,14];
    
    c1 = c+[-4-v1,+v2];
    c2 = c+[4+v1,+v2];
    r1 = 2+v3;
    r2 = 2+v3;
    
    X1 = ones(32,1)*(1:32)-c1(1);
    Y1 = (1:32)'*ones(1,32)-c1(2);
    R1 = sqrt(X1.^2+Y1.^2);
    X2 = ones(32,1)*(1:32)-c2(1);
    Y2 = (1:32)'*ones(1,32)-c2(2);
    R2 = sqrt(X2.^2+Y2.^2);
    
    In = I;
    In(R1<=r1) = 0;
    In(R2<=r2) = 0;
    
else
    
    In = I;
    
end