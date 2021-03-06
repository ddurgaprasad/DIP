function    butterworth_filter( img,cutoff, filter_order)
 
f=imread(img);
[M,N]=size(f);
F=fft2(double(f));
u=0:(M-1);
v=0:(N-1);
idimg=find(u>M/2);
u(idimg)=u(idimg)-M;
idy=find(v>N/2);
v(idy)=v(idy)-N;
[V,U]=meshgrid(v,u);
D=sqrt(U.^2+V.^2);
H = (1./(1 + D./cutoff).^(2*filter_order));  

G=H.*F;

g=real(ifft2(double(G)));
imshow(f),figure,imshow(g,[ ]);
end    