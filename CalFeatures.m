% calculate seismic features from JiangCS's catalog.
clc; clear;
workpath = 'D:\毕设\Code\data\';
% datainput = load([workpath,'Chinabig2021.dat']); 
datainput = load([workpath,'Chuandian2021.dat']); 

% select events larger than Mc
Mc=3.0;
data1 = datainput(datainput(:,10)>=Mc,:); 

year=data1(:,1);month=data1(:,2);
day=data1(:,3);hour=data1(:,4);
minute=data1(:,5);second=data1(:,6);
lat=data1(:,7);lon=data1(:,8);
depth=data1(:,9);mag=data1(:,10);

% % format of ZMAP catalog
b=fopen([workpath,'Chuandian2021_zmap.dat'],'w');
fprintf(b,'Longitude	Latitude	Year	Month	Day	Mag	Hour	Minite	Second \n');
for m=1:length(year)
    fprintf(b,'%f %f %d %d %d %f %d %d %f \n',lon(m),lat(m),year(m),month(m),day(m),mag(m),hour(m),minute(m),second(m));
end
fclose(b);

% calculate Julian days
data=[year,month,day,hour,minute,second];
m=size(data,1);
jd0=cal2jd([1970,1,1,0,0,0]);
for i=1:m
    jd(i)=cal2jd(data(i,:))-jd0;
end
cat_jd=jd';

% Parameters Initilization
% dt=30;Twindow=2*365;
% Tfore=365;
dt=30;Twindow=60;
Tfore=30;
Mag_predmax_threshold=5.0;

% loop for calculate features
Nloop=ceil((cat_jd(end)-cat_jd(1)-Twindow-Tfore)/dt);
% features array
t=zeros(Nloop,1);
Num=zeros(Nloop,1);
Mag_max=zeros(Nloop,1);
Mag_max_obs=zeros(Nloop,1);
Mag_mean=zeros(Nloop,1);
b_lsq=zeros(Nloop,1);a_lsq=zeros(Nloop,1);
b_std_lsq=zeros(Nloop,1);std_gr_lsq=zeros(Nloop,1);
b_mlk=zeros(Nloop,1);a_mlk=zeros(Nloop,1);
b_std_mlk=zeros(Nloop,1);std_gr_mlk=zeros(Nloop,1);
dM_lsq=zeros(Nloop,1);dM_mlk=zeros(Nloop,1);
Energy_sqrt=zeros(Nloop,1);
prob_x7_lsq=zeros(Nloop,1);
prob_x7_mlk=zeros(Nloop,1);
zvalue=zeros(Nloop,1);
beta=zeros(Nloop,1);
T_elaps6=zeros(Nloop,1);T_elaps65=zeros(Nloop,1);
T_elaps7=zeros(Nloop,1);T_elaps75=zeros(Nloop,1);

for i=1:Nloop
    t(i)=Twindow+cat_jd(1)+(i-1)*dt;
    index=find(cat_jd>=(t(i)-Twindow) & cat_jd<t(i));
    
    subcat=data1(index,:);
    % Number, Max/Mean magnitude of input windows
    Num(i)=length(index);
    Mag_max(i)=max(subcat(:,10));
    Mag_mean(i)=mean(subcat(:,10));
    
    % b/a value of least square regression/ maximum likelihood
    dMag=0.1;
    Mag_int=Mc:dMag:Mag_max(i);
    Mag_int=Mag_int';
    len_M=length(Mag_int);
    NumM=zeros(len_M,1);
    for j=1:len_M
        index_j=find(subcat(:,10)>=Mag_int(j));
        NumM(j)=length(index_j);
    end
    b_lsq(i)=(len_M.*sum(Mag_int.*log10(NumM))-sum(Mag_int).*sum(log10(NumM)))./(sum(Mag_int).*sum(Mag_int)-len_M.*sum(Mag_int.*Mag_int));
    a_lsq(i)=sum(log10(NumM+b_lsq(i).*Mag_int))/len_M;
    b_std_lsq(i)=2.3*b_lsq(i)*b_lsq(i)*sqrt((sum((Mag_int-Mag_mean(i)).*(Mag_int-Mag_mean(i)))./len_M./(len_M-1)));
    std_gr_lsq(i)=sum((log10(NumM)-a_lsq(i)-b_lsq(i)*Mag_int).*(log10(NumM)-a_lsq(i)-b_lsq(i)*Mag_int))/(len_M-1);
    
    b_mlk(i)=log10(exp(1))/(Mag_mean(i)-Mc);
    a_mlk(i)=log10(Num(i))+b_mlk(i)*Mc;
    b_std_mlk(i)=2.3*b_mlk(i)*b_mlk(i)*sqrt((sum((Mag_int-Mag_mean(i)).*(Mag_int-Mag_mean(i)))/len_M/(len_M-1)));
    std_gr_mlk(i)=sum((log10(NumM)-a_mlk(i)-b_mlk(i)*Mag_int).*(log10(NumM)-a_mlk(i)-b_mlk(i)*Mag_int))/(len_M-1);
    
    % Max magnitude defict
    dM_lsq(i)=Mag_max(i)-a_lsq(i)/b_lsq(i);
    dM_mlk(i)=Mag_max(i)-a_mlk(i)/b_mlk(i);
    
    % seismic energy released 
    Energy_sqrt(i)=sqrt(sum(10.^(12+1.8*subcat(:,10))));
    
    % elapse time since last M>=6.0/6.5/7.0/7.5 event
    Mag_elaps=[6.0;6.5;7.0;7.5];
    indexT_elaps=find(cat_jd<t(i));
    indexT_elaps6=find(mag(indexT_elaps)>=Mag_elaps(1));indexT_elaps65=find(mag(indexT_elaps)>=Mag_elaps(2));
    indexT_elaps7=find(mag(indexT_elaps)>=Mag_elaps(3));indexT_elaps75=find(mag(indexT_elaps)>=Mag_elaps(4));
    T_elaps6(i)=t(i)-cat_jd(max(indexT_elaps6));
    T_elaps65(i)=t(i)-cat_jd(max(indexT_elaps65));
    T_elaps7(i)=t(i)-cat_jd(max(indexT_elaps7));
    T_elaps75(i)=t(i)-cat_jd(max(indexT_elaps75));
    
    % probability of earthquake occurrence
    prob_x7_lsq(i)=exp(-3*b_lsq(i)/log10(exp(1)));
    prob_x7_mlk(i)=exp(-3*b_mlk(i)/log10(exp(1)));
    
    % seismicity rate change: beta value and zvalue
    mCat=cat_jd(index);
    fTstart=t(i)-Twindow;
    fT=t(i)-0.5*Twindow;
    fTw=0.5*Twindow;
    Tbin=0.05*Twindow;
    indexr1=find(mCat>=fTstart & mCat<fT);
    indexr2=find(mCat>=fT & mCat<(fT+fTw));
    N1=length(indexr1);N2=length(indexr2);
    R1=N1/(fT-fTstart);R2=N2/fTw;
    nR1=histc(mCat(indexr1),fTstart:Tbin:fT);
    nR2=histc(mCat(indexr2),fT:Tbin:(fT+fTw));
    S1=std(nR1);S2=std(nR2);
    zvalue(i)=(R1-R2)/sqrt(S1*S1/N1+S2*S2/N2);
    
    vR1=histc(mCat,fTstart:Tbin:(fT+fTw));
    nEq1=sum(vR1);
    nBin1=size(vR1,1);
    winlen_days=fTw/Tbin;
    fNormInvalLength=winlen_days/nBin1; % normalized interval length
    beta(i)=(N2 - nEq1.*fNormInvalLength )./sqrt(nEq1.*fNormInvalLength.*(1-fNormInvalLength));
    
    % Observed Maximum magnitude in the prediction period
    index_Max_mag_obs=find(cat_jd>=t(i) & cat_jd<(t(i)+Tfore));
    Mag_max_obs(i)=max(data1(index_Max_mag_obs,10));
    
    % introduce other features such as slip rates et. al. in the future
    
end

% output
% a=fopen([workpath,'Chinabig2021-features-dt15.dat'],'w');
a=fopen([workpath,'Chuandian2021-features-dt30Tw60Tp30.dat'],'w');
fprintf(a,'Time(Jdays) N.O. Mag_max Mag_mean b_lsq a_lsq b_std_lsq std_gr_lsq b_mlk a_mlk b_std_mlk std_gr_mlk dM_lsq dM_mlk Energy x7_lsq x7_mlk zvalue beta T_elaps6 T_elaps65 T_elaps7 T_elaps75 Mag_max_obs \n');
for k=1:Nloop
    fprintf(a,' %f %d %4.1f %6.3f %8.5f %8.5f %8.5f %8.5f %8.5f %8.5f %8.5f %8.5f %6.3f %6.3f %e %6.4f %6.4f %8.5f %8.5f %8.3f %8.3f %8.3f %8.3f %4.1f \n',...
    t(k),Num(k),Mag_max(k),Mag_mean(k),b_lsq(k),a_lsq(k),b_std_lsq(k),std_gr_lsq(k),...
    b_mlk(k),a_mlk(k),b_std_mlk(k),std_gr_mlk(k),dM_lsq(k),dM_mlk(k),Energy_sqrt(k),...
    prob_x7_lsq(k),prob_x7_mlk(k),zvalue(k),beta(k),T_elaps6(k),T_elaps65(k),T_elaps7(k),T_elaps75(k),Mag_max_obs(k));
end
fclose(a);



