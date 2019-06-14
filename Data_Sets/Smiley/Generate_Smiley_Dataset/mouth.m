function In = mouth(I,s,v1,v2,v3,v4,v5)

if s == 1
    
    lx = 4+v1;
    ly = 2+v2;
    p = [16+0.5*v3,22+0.5*v4];
    stx = round(p(1)-lx/2);
    sty = round(p(2)-ly/2);
    enx = round(p(1)+lx/2);
    eny = round(p(2)+ly/2);
    
    In = I;
    In(sty:eny,stx:enx) = 0;
    
    sy1 = round(p(2)+1*v5);
    sy2 = round(p(2)+1.4*v5);
    sx1 = enx+1;
    sx2 = enx+2;
    bx1 = stx-1;
    bx2 = stx-2;
    
    In(sy1,sx1) = 0;
    In(sy2,sx2) = 0;
    In(sy1,bx1) = 0;
    In(sy2,bx2) = 0;
    
else
    In = I;
end