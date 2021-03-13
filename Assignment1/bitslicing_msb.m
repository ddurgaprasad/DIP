% clearing the output screen 
clc; 

% reading image's pixel in c 
A = imread('C:/SAI/IIIT/2019_Monsoon/DIP/Assignment1/A1_resources/DIP_2019_A1/lena.jpg'); 
A = rgb2gray(A);


% c1=bitget(A,1);
% figure,
% subplot(2,2,1);imshow(logical(c1));title('Bit plane 1');
% 
% c2=bitget(A,2);
% subplot(2,2,2);imshow(logical(c2));title('Bit plane 2');
% 
% c3=bitget(A,3);
% subplot(2,2,3);imshow(logical(c3));title('Bit plane 3');
% 
% c4=bitget(A,4);
% subplot(2,2,4);imshow(logical(c4));title('Bit plane 4');
% c5=bitget(A,5);
% figure,
% subplot(2,2,1);imshow(logical(c5));title('Bit plane 5');
% 
% c6=bitget(A,6);
% subplot(2,2,2);imshow(logical(c6));title('Bit plane 6');
% 
% c7=bitget(A,7);
% subplot(2,2,3);imshow(logical(c7));title('Bit plane 7');
% 
% c8=bitget(A,8);
% subplot(2,2,4);imshow(logical(c8));title('Bit plane 8');
% 

B=zeros(size(A));
B=bitset(B,8,bitget(A,8)); % MSB
B=bitset(B,7,bitget(A,7));
B=bitset(B,6,bitget(A,6));
B=bitset(B,5,bitget(A,5));
B=bitset(B,4,bitget(A,4));
B=bitset(B,3,bitget(A,3));
% B=bitset(B,2,bitget(A,2));
% B=bitset(B,1,bitget(A,1)); %Comment this to set LSB TO ZERO
B=uint8(B);
%figure,imshow(B);

figure,
subplot(2,2,1);imshow(A);title('Original');
subplot(2,2,2);imhist(A);title('Original');

subplot(2,2,3);imshow(B);title('Combined');
subplot(2,2,4);imhist(B);title('Combined');




