function In = add_bowtie(I,s,bt,v1,v2,v3,v4)

if s == 1
    
    c = [16.5+0.5*v1,29+v2];
    sz = [12+3*v3,5+2*v4];
    
    bt = round(mat2gray(imresize(bt,[sz(2),sz(1)])));
    szb = size(bt);
    
    stx = round(c(1)-sz(1)/2);
    enx = stx+szb(2)-1;
    sty = round(c(2)-sz(2)/2);
    eny = sty+szb(1)-1;
    
    In = [I;zeros(5,size(I,2))];
    In(sty:eny,stx:enx) = In(sty:eny,stx:enx)+bt;
    In(In>1)=1;
    
else
    
    In = I;
    
end
In = In(1:32,1:32);