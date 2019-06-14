function In = make_hat(I,s,v1,v2,v3,v4,v5,v6)

if s == 1
    
    c = [16.5+v1,6+v2];
    sz = [15+3*v3,2+2*v4];
    
    stx = round(c(1)-sz(1)/2);
    enx = round(c(1)+sz(1)/2);
    sty = round(c(2)-sz(2)/2)+5;
    eny = round(c(2)+sz(2)/2)+5;
    
    sz2 = [10+2*v5,2+v6];
    
    In = [zeros(5,size(I,2));I];
    In(sty:eny,stx:enx) = 1;
    stny = max(round(sty-sz2(2)),1);
    In(stny:sty,round(c(1)-sz2/2):round(c(1)+sz2/2)) = 1;
    
else
    
    In = I;
    
end

In = In(end-32+1:end,end-32+1:end);