%% Parametric Model Embedding (PME)
% developed by A. Serani and M. Diez, 06/12/2022

%% NACA 0012 example
% Parameterization by Bezier curves

load("pme_dataset.mat");

M = 14;                                     % Number of original design variables
K = (size(Daux,1) - M)/2;                   % Number of elements in each coordinate direction 
S = size(Daux,2);                           % Number of MC samples, first element is the parent geometry
CI = 0.95;                                  % Dimensionality reduction confidence interval

%% Build and centering data matrix

for j=1:S 
    delta(:,j) = Daux(:,j)-Daux(:,1);       % Modification vectors
end

delta_m = mean(delta,2);                    % Mean vector
Dc = delta - delta_m;                       % Data centering

%% Variance estimate

sigma2_g = zeros(S,1);
sigma2_x = zeros(S,1);

for j=1:S
    for k=1:K
        sigma2_g(j) = sigma2_g(j) + (delta(k  ,j) - delta_m(k  ))^2 ...
                                  + (delta(k+K,j) - delta_m(k+K))^2;      % Geometric variance
    end
    for m=1:M
        sigma2_x(j) = sigma2_x(j) + (delta(m+2*K,j) - delta_m(m+2*K))^2;  % Variables variance 
    end
end

for k=1:K
     rho_g(k) = 1./(sum(sigma2_g)/S);         % Geometric weights
end
disp(['---> Geometric variance = ' num2str(sum(sigma2_g)/S)])
disp(['---> Variables variance = ' num2str(sum(sigma2_x)/S)])

for m=1:M
   rho_x(m) = 0./(sum(sigma2_x)/S);         % Variables weights
end

%% Build weight matrix

W = zeros(2*K+M);
for k=1:K
    W(k  ,k  ) = rho_g(k);
    W(k+K,k+K) = rho_g(k);
end
for m=1:M
    W(m+2*K,m+2*K) = rho_x(m);
end

A  = (Dc*Dc')/S;                        % Autocovariance matrix
AW = A*W;                               % Weighted autocovarance

%% PCA and descending sorting

Zs = zeros(2*K+M);
[Z,Laux] = eig(AW);
[L,I] = sort(diag(Laux),'descend');
for i=1:2*K+M
    Zs(:,i) = Z(:,I(i)); 
end

%% Eigenvector normalization

an = Zs'*W*Zs;
for j=1:size(Zs,1)
    for i=1:size(Zs,1)
        Zn(i,j) = Zs(i,j)/sqrt(an(j,j));
    end
end

%% Dimensionality reduction

for k=1:2*K+M
    if(cumsum(real(L(1:k)))<=CI) 
       nconf = k;
    end
end

disp(['---> ' num2str(nconf+1) ' eigenvalues conver ' num2str(sum(L(1:nconf+1))*100) '% of problem variance'])
disp(['---> ' num2str((1-(nconf+1)/M)*100) '% dimensionality reduction achieved'])

Zr = Zn(:,1:nconf+1);
Lr = L(1:nconf+1);

%% Reduced design variables bounds

ak = Dc'*W*Zn;
for j=1:M %nconf+1
    alfak(j,1) = min(real(ak(:,j)));
    alfak(j,2) = max(real(ak(:,j)));
    alfak(j,3) = sum(real(ak(:,j)).^2)/S;
end

%% Geometric modes

xnew_u = zeros(K,nconf+1);
ynew_u = zeros(K,nconf+1);
xnew_l = zeros(K,nconf+1);
ynew_l = zeros(K,nconf+1);

for j=1:nconf+1
    for k =1:K
      xnew_u(k,j)=(Daux(k,1  )+delta_m(k  ))+alfak(j,2)*Zr(k  ,j);
      ynew_u(k,j)=(Daux(k+K,1)+delta_m(k+K))+alfak(j,2)*Zr(k+K,j);
      xnew_l(k,j)=(Daux(k,1  )+delta_m(k  ))+alfak(j,1)*Zr(k  ,j);
      ynew_l(k,j)=(Daux(k+K,1)+delta_m(k+K))+alfak(j,1)*Zr(k+K,j);

    end
end

%% Reconstruction

for j=1:S
    for k=1:2*K
        geo_rec_redu(k,j) = Daux(k    ,1) + delta_m(k    ,1) + sum(real(ak(j,1:nconf+1)).*Zn(k    ,1:nconf+1));
        geo_rec_full(k,j) = Daux(k    ,1) + delta_m(k    ,1) + sum(real(ak(j,1:M      )).*Zn(k    ,1:M      ));
    end
    for k=1:K
        rec_aux_kle(k) = ((geo_rec_redu(k,j)-Daux(k,j))^2+(geo_rec_redu(k+K,j)-Daux(k+K,j))^2);
    end
    rec_err_kle(j) = sum(rec_aux_kle);
    for i=1:M
        var_rec_redu(i,j) = Daux(2*K+i,1) + delta_m(2*K+i,1) + sum(real(ak(j,1:nconf+1)).*Zn(2*K+i,1:nconf+1));
        var_rec_full(i,j) = Daux(2*K+i,1) + delta_m(2*K+i,1) + sum(real(ak(j,1:M      )).*Zn(2*K+i,1:M      ));
    end
end

disp(['---> Geometric NMSE% = ' num2str(100*(sum(rec_err_kle)/S)/(sum(sigma2_g)/S))])

N = nconf+1;

%% Plotting

figure(1)
hold on
for j=1:nconf+1
    subplot(nconf+1,1,j),hold on,plot(Daux(1:K,1),Daux(K+1:2*K,1),'k--')
    subplot(nconf+1,1,j),hold on,plot(xnew_u(:,j),ynew_u(:,j),'b-')
    subplot(nconf+1,1,j),hold on,plot(xnew_l(:,j),ynew_l(:,j),'g-')
    xlim([0,1])
    ylim([-0.1,0.1])
    ylabel('y/c [-]')
    set(gca,'XTick',[])
end
xlabel('x/c [-]')
hold off

save('pme_geomode.mat','xnew_u','ynew_u','xnew_l','ynew_l','ak')

figure(2)
hold on
for j=1:nconf+1
    subplot(nconf+1,1,j),hold on,plot(Zr(2*K+1:end,j),'go-')
    xlim([1,M])
    %ylim([-0.15,0.15])
    %ylabel('y/c [-]')
    set(gca,'XTick',[])
end
xlabel('Eigenvector variables components [-]')
hold off

save('pme_varmode.mat','Zr')

figure(3)
hold on
plot(cumsum(real(L(1:M)))*100,'go-')
plot([1,M],[95,95],'k--')
xlabel('Number of reduced design variables, N [-]')
ylabel('Variance retained %')


figure(5)
hold on
plot(Daux(2*K+1:end,100),var_rec_full(:,100),'ko')
plot(Daux(2*K+1:end,100),var_rec_redu(:,100),'bs')
xlabel('Original design variables [-]')
ylabel('Reconstructed design variables [-]')

save('pme_var-rec.mat','var_rec_full','var_rec_redu')

figure(6)
hold on
gtest = 2;
plot(Daux(1:K,gtest),Daux(K+1:2*K,gtest),'r:')
plot(geo_rec_full(1:K,gtest),geo_rec_full(K+1:2*K,gtest),'k-')
plot(geo_rec_redu(1:K,gtest),geo_rec_redu(K+1:2*K,gtest),'g-')
xlabel('x/c [-]')
ylabel('y/c [-]')

save('pme_geo-rec.mat','geo_rec_full','geo_rec_redu')

